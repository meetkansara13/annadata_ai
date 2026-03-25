"""
अन्नदाता AI — FastAPI Server
All routes for crop disease detection, mandi prices, schemes, voice, farmer management.

Run:
    uvicorn api:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import os
import io
import sys
from pathlib import Path
from datetime import date

import torch
import torch.nn.functional as F
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase1.model.model   import load_checkpoint, build_model
from phase1.utils.dataset import CLASS_INFO, TREATMENT_ADVICE, get_transforms
from database             import (
    SessionLocal, Farmer, Prediction, MandiPrice,
    Scheme, PriceAlert, log_prediction, get_schemes_for_state
)

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
NUM_CLASSES     = 15
DEVICE          = "cpu"

# ── State → Language map ──────────────────────────────────────────────────────
STATE_LANGUAGE_MAP = {
    "Uttar Pradesh": "hi",   "Bihar": "hi",
    "Madhya Pradesh": "hi",  "Rajasthan": "hi",
    "Uttarakhand": "hi",     "Himachal Pradesh": "hi",
    "Haryana": "hi",         "Delhi": "hi",
    "Jharkhand": "hi",       "Chhattisgarh": "hi",
    "Gujarat": "gu",         "Maharashtra": "mr",
    "Goa": "kok",            "Tamil Nadu": "ta",
    "Karnataka": "kn",       "Kerala": "ml",
    "Andhra Pradesh": "te",  "Telangana": "te",
    "West Bengal": "bn",     "Odisha": "or",
    "Assam": "as",           "Punjab": "pa",
    "Manipur": "mni",        "Meghalaya": "kha",
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "अन्नदाता AI",
    description = "Crop disease detection for Indian farmers — 15 diseases, 22 Indian languages",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Model (loaded once at startup) ────────────────────────────────────────────
model       = None
class_names = None

@app.on_event("startup")
async def load_model():
    global model, class_names
    if CHECKPOINT_PATH.exists():
        model = load_checkpoint(CHECKPOINT_PATH, num_classes=NUM_CLASSES, device=DEVICE)
        print("[अन्नदाता AI] ✅ Model loaded for inference")
    else:
        print("[अन्नदाता AI] ⚠️  No checkpoint found — running in DEMO mode")
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
    class_names = list(CLASS_INFO.keys())[:NUM_CLASSES]


# ── DB dependency ─────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Inference helper ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(image_bytes: bytes, lang: str = "hi") -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    transform  = get_transforms(augment=False)
    tensor     = transform(img).unsqueeze(0)
    logits     = model(tensor.to(DEVICE))
    probs      = F.softmax(logits, dim=1)[0]
    top5_p, top5_i = torch.topk(probs, k=min(5, NUM_CLASSES))

    top_class  = class_names[top5_i[0].item()]
    confidence = round(top5_p[0].item() * 100, 2)
    info       = CLASS_INFO.get(top_class, {"en": top_class, "hi": top_class})
    is_healthy = "healthy" in top_class.lower()
    treatment  = TREATMENT_ADVICE.get(top_class, "Consult your nearest Krishi Vigyan Kendra (KVK).") if not is_healthy else None

    top5 = [
        {
            "class":      class_names[i.item()],
            "label":      CLASS_INFO.get(class_names[i.item()], {}).get(lang, class_names[i.item()]),
            "confidence": round(p.item() * 100, 2),
        }
        for p, i in zip(top5_p, top5_i)
    ]

    return {
        "disease_class":    top_class,
        "disease_label":    info.get(lang, info["en"]),
        "confidence_pct":   confidence,
        "is_healthy":       is_healthy,
        "treatment":        treatment,
        "top5_predictions": top5,
        "language":         lang,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "app":          "अन्नदाता AI",
        "version":      "1.0.0",
    }


# ── Disease prediction ────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    file:      UploadFile = File(..., description="Crop leaf image JPG/PNG"),
    lang:      str        = Form(default="hi", description="Language: hi/gu/mr/ta/te/bn/pa"),
    farmer_id: int        = Form(default=None, description="Farmer ID (optional)"),
    db:        Session    = Depends(get_db),
):
    """
    Upload crop leaf photo → get disease name + treatment in farmer's language.

    Example response:
    {
      "disease_label": "टमाटर का अगेती झुलसा",
      "confidence_pct": 94.3,
      "is_healthy": false,
      "treatment": "Spray Copper Oxychloride 50% WP @ 3g/L"
    }
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image (JPG/PNG).")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Max 10MB.")

    result = run_inference(image_bytes, lang=lang)

    # Log to database
    log_prediction(
        disease_class = result["disease_class"],
        disease_hindi = result["disease_label"],
        confidence    = result["confidence_pct"],
        treatment     = result["treatment"] or "Healthy crop",
        farmer_id     = farmer_id,
    )

    return result


# ── Farmer registration ───────────────────────────────────────────────────────
@app.post("/farmer/register")
async def register_farmer(
    name:     str = Form(...),
    phone:    str = Form(...),
    state:    str = Form(...),
    district: str = Form(...),
    language: str = Form(default=None),
    db: Session   = Depends(get_db),
):
    """Register a new farmer. Auto-assigns language based on state if not provided."""

    # Check if already registered
    existing = db.query(Farmer).filter(Farmer.phone == phone).first()
    if existing:
        raise HTTPException(status_code=400, detail="Farmer with this phone already registered.")

    # Auto language from state
    auto_lang = STATE_LANGUAGE_MAP.get(state, "hi")
    lang      = language or auto_lang

    farmer = Farmer(
        name     = name,
        phone    = phone,
        state    = state,
        district = district,
        language = lang,
    )
    db.add(farmer)
    db.commit()
    db.refresh(farmer)

    return {
        "success":  True,
        "farmer_id": farmer.id,
        "name":     farmer.name,
        "language": farmer.language,
        "message":  f"Farmer registered successfully! Language set to '{lang}'"
    }


# ── Farmer history ────────────────────────────────────────────────────────────
@app.get("/farmer/{farmer_id}/history")
async def farmer_history(farmer_id: int, db: Session = Depends(get_db)):
    """Get all past disease predictions for a farmer."""
    farmer = db.query(Farmer).filter(Farmer.id == farmer_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found.")

    predictions = db.query(Prediction)\
                    .filter(Prediction.farmer_id == farmer_id)\
                    .order_by(Prediction.created_at.desc())\
                    .limit(20)\
                    .all()

    return {
        "farmer":      farmer.name,
        "state":       farmer.state,
        "language":    farmer.language,
        "total":       len(predictions),
        "predictions": [
            {
                "id":            p.id,
                "disease":       p.disease_class,
                "disease_local": p.disease_hindi,
                "confidence":    p.confidence,
                "treatment":     p.treatment,
                "date":          p.created_at.strftime("%d %b %Y %H:%M"),
            }
            for p in predictions
        ]
    }


# ── Mandi prices ──────────────────────────────────────────────────────────────
@app.get("/prices")
async def get_prices(
    state:     str = None,
    commodity: str = None,
    db: Session    = Depends(get_db),
):
    """Get mandi prices — filter by state and/or commodity."""
    query = db.query(MandiPrice)
    if state:
        query = query.filter(MandiPrice.state.ilike(f"%{state}%"))
    if commodity:
        query = query.filter(MandiPrice.commodity.ilike(f"%{commodity}%"))

    prices = query.order_by(MandiPrice.price_date.desc()).limit(50).all()

    return {
        "total": len(prices),
        "prices": [
            {
                "market":      p.market,
                "state":       p.state,
                "district":    p.district,
                "commodity":   p.commodity,
                "min_price":   p.min_price,
                "max_price":   p.max_price,
                "modal_price": p.modal_price,
                "date":        str(p.price_date),
            }
            for p in prices
        ]
    }


# ── Price alert ───────────────────────────────────────────────────────────────
@app.post("/prices/alert")
async def set_price_alert(
    farmer_id:    int   = Form(...),
    commodity:    str   = Form(...),
    target_price: float = Form(...),
    market:       str   = Form(...),
    db: Session         = Depends(get_db),
):
    """Set a price alert — notify farmer when commodity hits target price."""
    farmer = db.query(Farmer).filter(Farmer.id == farmer_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found.")

    alert = PriceAlert(
        farmer_id    = farmer_id,
        commodity    = commodity,
        target_price = target_price,
        market       = market,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    return {
        "success":   True,
        "alert_id":  alert.id,
        "message":   f"Alert set! You'll be notified when {commodity} reaches ₹{target_price} at {market}."
    }


# ── Government schemes ────────────────────────────────────────────────────────
@app.get("/schemes")
async def get_schemes(state: str = "ALL", db: Session = Depends(get_db)):
    """Get all government schemes available for a state."""
    schemes = db.query(Scheme)\
                .filter((Scheme.state == "ALL") | (Scheme.state == state))\
                .all()

    return {
        "state":   state,
        "total":   len(schemes),
        "schemes": [
            {
                "id":          s.id,
                "name":        s.name,
                "description": s.description,
                "eligibility": s.eligibility,
                "benefit":     s.benefit,
                "apply_url":   s.apply_url,
            }
            for s in schemes
        ]
    }


# ── Language info ─────────────────────────────────────────────────────────────
@app.get("/language/{state}")
async def get_language_for_state(state: str):
    """Get the recommended language for a given Indian state."""
    lang = STATE_LANGUAGE_MAP.get(state, "hi")
    return {
        "state":    state,
        "language": lang,
        "note":     "Farmer can override this in their profile"
    }


# ── Supported diseases ────────────────────────────────────────────────────────
@app.get("/diseases")
async def get_diseases(lang: str = "hi"):
    """List all 15 supported crop diseases."""
    return {
        "total": len(class_names),
        "diseases": [
            {
                "id":    i,
                "class": c,
                "label": CLASS_INFO.get(c, {}).get(lang, c),
            }
            for i, c in enumerate(class_names)
        ]
    }


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)