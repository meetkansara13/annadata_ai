"""
अन्नदाता AI — Single Entry Point
One command to run everything:
    python run.py

URLs:
    http://localhost:8001/         → Landing page
    http://localhost:8001/app      → Farmer Gradio App
    http://localhost:8001/docs     → API Documentation
    http://localhost:8001/predict  → Disease Detection API
    http://localhost:8001/prices   → Mandi Prices API
    http://localhost:8001/schemes  → Govt Schemes API
"""

import sys, io, os
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent))

# ── Imports ────────────────────────────────────────────────────────────────────
from phase1.model.model   import load_checkpoint, build_model
from phase1.utils.dataset import CLASS_INFO, TREATMENT_ADVICE, get_transforms
from phase2.mandi         import get_latest_prices
from phase3.voice         import text_to_speech, STATE_LANGUAGE_MAP
from phase4.schemes       import match_schemes
from database             import (
    SessionLocal, Farmer, MandiPrice, Scheme,
    PriceAlert, log_prediction
)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import gradio as gr
import requests as http_requests

# ── Load Model ─────────────────────────────────────────────────────────────────
CHECKPOINT = Path("checkpoints/best_model.pt")
NUM_CLASSES = 15
DEVICE      = "cpu"
class_names = list(CLASS_INFO.keys())[:NUM_CLASSES]

print("[अन्नदाता AI] Loading model...")
model = load_checkpoint(CHECKPOINT, num_classes=NUM_CLASSES, device=DEVICE) \
    if CHECKPOINT.exists() \
    else build_model(num_classes=NUM_CLASSES, pretrained=False)
print("[अन्नदाता AI] ✅ Model ready")

# ── Constants ──────────────────────────────────────────────────────────────────
LANG_MAP = {
    "Hindi (हिंदी)":       "hi",
    "Gujarati (ગુજરાતી)":  "gu",
    "Marathi (मराठी)":     "mr",
    "Tamil (தமிழ்)":       "ta",
    "Kannada (ಕನ್ನಡ)":    "kn",
    "Malayalam (മലയാളം)":  "ml",
    "Telugu (తెలుగు)":     "te",
    "Bengali (বাংলা)":     "bn",
    "Odia (ଓଡ଼ିଆ)":        "or",
    "Assamese (অসমীয়া)":  "as",
    "Punjabi (ਪੰਜਾਬੀ)":    "pa",
}
LANG_OPTIONS = list(LANG_MAP.keys())
STATES = sorted([
    "Gujarat","Maharashtra","Punjab","Uttar Pradesh","Rajasthan",
    "Madhya Pradesh","Karnataka","Andhra Pradesh","West Bengal",
    "Bihar","Tamil Nadu","Kerala","Telangana","Haryana",
    "Himachal Pradesh","Uttarakhand","Odisha","Assam","Delhi",
])

# ── FastAPI app ─────────────────────────────────────────────────────────────────
api = FastAPI(
    title       = "अन्नदाता AI",
    description = "Crop Disease Detection API for Indian Farmers",
    version     = "1.0.0",
    docs_url    = "/docs",
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def get_db():
    db = SessionLocal()
    try:    yield db
    finally: db.close()

# ── Core inference function ────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(image_bytes: bytes, lang: str = "hi") -> dict:
    img       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_transforms(augment=False)
    tensor    = transform(img).unsqueeze(0)
    logits    = model(tensor.to(DEVICE))
    probs     = F.softmax(logits, dim=1)[0]
    top5_p, top5_i = torch.topk(probs, k=5)

    top_class  = class_names[top5_i[0].item()]
    confidence = round(top5_p[0].item() * 100, 2)
    info       = CLASS_INFO.get(top_class, {"en": top_class, "hi": top_class})
    is_healthy = "healthy" in top_class.lower()
    label      = info.get(lang, info.get("en", top_class))
    treatment  = TREATMENT_ADVICE.get(top_class, "Consult your nearest KVK.") if not is_healthy else None

    top5 = [{
        "class":      class_names[i.item()],
        "label":      CLASS_INFO.get(class_names[i.item()], {}).get(lang, class_names[i.item()]),
        "confidence": round(p.item() * 100, 2),
    } for p, i in zip(top5_p, top5_i)]

    return {
        "disease_class":    top_class,
        "disease_label":    label,
        "confidence_pct":   confidence,
        "is_healthy":       is_healthy,
        "treatment":        treatment,
        "top5_predictions": top5,
        "language":         lang,
    }

# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Routes
# ══════════════════════════════════════════════════════════════════════════════

@api.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "version": "1.0.0"}

@api.post("/predict")
async def predict(
    file:      UploadFile = File(...),
    lang:      str        = Form(default="hi"),
    farmer_id: int        = Form(default=None),
    db:        Session    = Depends(get_db),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "File must be an image.")
    image_bytes = await file.read()
    result = run_inference(image_bytes, lang=lang)
    log_prediction(
        disease_class = result["disease_class"],
        disease_hindi = result["disease_label"],
        confidence    = result["confidence_pct"],
        treatment     = result["treatment"] or "Healthy",
        farmer_id     = farmer_id,
    )
    return result

@api.post("/farmer/register")
async def register_farmer(
    name:     str = Form(...),
    phone:    str = Form(...),
    state:    str = Form(...),
    district: str = Form(...),
    language: str = Form(default="hi"),
    db: Session   = Depends(get_db),
):
    existing = db.query(Farmer).filter(Farmer.phone == phone).first()
    if existing:
        return {"success": True, "farmer_id": existing.id,
                "message": f"Welcome back, {existing.name}!"}
    farmer = Farmer(name=name, phone=phone, state=state,
                    district=district, language=language)
    db.add(farmer); db.commit(); db.refresh(farmer)
    return {"success": True, "farmer_id": farmer.id,
            "message": "Registered successfully!"}

@api.get("/farmer/{farmer_id}/history")
async def farmer_history(farmer_id: int, db: Session = Depends(get_db)):
    from database import Prediction
    farmer = db.query(Farmer).filter(Farmer.id == farmer_id).first()
    if not farmer:
        raise HTTPException(404, "Farmer not found.")
    preds = db.query(Prediction).filter(
        Prediction.farmer_id == farmer_id
    ).order_by(Prediction.created_at.desc()).limit(20).all()
    return {
        "farmer": farmer.name, "state": farmer.state,
        "total": len(preds),
        "predictions": [{
            "disease": p.disease_class, "label": p.disease_hindi,
            "confidence": p.confidence, "date": str(p.created_at)
        } for p in preds]
    }

@api.get("/prices")
async def get_prices(state: str = None, commodity: str = None,
                     db: Session = Depends(get_db)):
    query = db.query(MandiPrice)
    if state:     query = query.filter(MandiPrice.state.ilike(f"%{state}%"))
    if commodity: query = query.filter(MandiPrice.commodity.ilike(f"%{commodity}%"))
    prices = query.order_by(MandiPrice.price_date.desc()).limit(50).all()
    return {
        "total": len(prices),
        "prices": [{
            "state": p.state, "district": p.district, "market": p.market,
            "commodity": p.commodity, "min_price": p.min_price,
            "max_price": p.max_price, "modal_price": p.modal_price,
            "date": str(p.price_date)
        } for p in prices]
    }

@api.get("/schemes")
async def get_schemes(state: str = "ALL", db: Session = Depends(get_db)):
    schemes = db.query(Scheme).filter(
        (Scheme.state == "ALL") | (Scheme.state == state)
    ).all()
    return {
        "state": state, "total": len(schemes),
        "schemes": [{
            "id": s.id, "name": s.name, "description": s.description,
            "benefit": s.benefit, "apply_url": s.apply_url
        } for s in schemes]
    }

@api.post("/prices/alert")
async def set_alert(
    farmer_id: int = Form(...), commodity: str = Form(...),
    target_price: float = Form(...), market: str = Form(...),
    db: Session = Depends(get_db),
):
    alert = PriceAlert(farmer_id=farmer_id, commodity=commodity,
                       target_price=target_price, market=market)
    db.add(alert); db.commit(); db.refresh(alert)
    return {"success": True, "alert_id": alert.id,
            "message": f"Alert set for {commodity} @ ₹{target_price}"}

@api.get("/diseases")
async def get_diseases(lang: str = "hi"):
    return {"total": len(class_names), "diseases": [
        {"id": i, "class": c, "label": CLASS_INFO.get(c, {}).get(lang, c)}
        for i, c in enumerate(class_names)
    ]}

# ── Landing page ───────────────────────────────────────────────────────────────
@api.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")

@api.get("/manifest.json")
async def manifest():
    from fastapi.responses import JSONResponse
    return JSONResponse({"name": "अन्नदाता AI", "short_name": "Annadata", "start_url": "/app"})

@api.get("/", response_class=HTMLResponse)
async def landing():
    html_path = Path("phase5/ui.html")
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        # Make buttons point to /app
        html = html.replace(
            '>Start Free</button>',
            ' onclick="window.location.href=\'/app\'">Launch App</button>'
        ).replace(
            '>Watch Demo</button>',
            ' onclick="window.location.href=\'/app\'">Open App</button>'
        )
        return html
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/app">')

# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI — calls FastAPI routes directly (no HTTP, same process)
# ══════════════════════════════════════════════════════════════════════════════

def ui_predict(image, language):
    """Calls /predict API internally."""
    if image is None:
        return "❌ Please upload an image.", "", "", None

    lang = LANG_MAP.get(language, "hi")

    # Convert PIL → bytes and POST to /predict
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    try:
        resp = http_requests.post(
            "http://localhost:8001/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
            data={"lang": lang},
            timeout=30
        )
        data = resp.json()
    except Exception as e:
        return f"❌ API Error: {e}", "", "", None

    label      = data.get("disease_label", "Unknown")
    confidence = data.get("confidence_pct", 0)
    is_healthy = data.get("is_healthy", False)
    treatment  = data.get("treatment") or "Crop is healthy!"
    top5       = data.get("top5_predictions", [])

    emoji  = "✅" if is_healthy else "⚠️"
    result = f"{emoji}  {label}\n\nConfidence: {confidence}%\nLanguage: {language}"

    top5_text = "📊 Top 5 Predictions:\n"
    for p in top5:
        top5_text += f"  • {p['label']}: {p['confidence']}%\n"

    # Voice
    Path("output/voice").mkdir(parents=True, exist_ok=True)
    audio_path = f"output/voice/result_{lang}.mp3"
    try:
        text_to_speech(f"{label}. {treatment}", lang=lang, output_path=audio_path)
    except:
        audio_path = None

    return result, f"💊 Treatment:\n{treatment}", top5_text, audio_path


def ui_prices(state, commodity):
    """Calls /prices API internally."""
    try:
        params = {}
        if state != "All States": params["state"] = state
        if commodity:             params["commodity"] = commodity
        resp   = http_requests.get("http://localhost:8001/prices", params=params, timeout=10)
        data   = resp.json()
        prices = data.get("prices", [])
        if not prices:
            return "No price data. Run: python phase2/scheduler.py"
        result = f"📈 Live Mandi Prices — {state}\n{'─'*60}\n\n"
        for p in prices:
            result += f"🌾 {p['commodity']:25} | {p['market']:30} | ₹{p['modal_price']:>8}/qtl | {p['date']}\n"
        return result
    except Exception as e:
        return f"❌ Error: {e}"


def ui_schemes(state, language):
    """Calls /schemes API internally."""
    try:
        resp    = http_requests.get("http://localhost:8001/schemes", params={"state": state}, timeout=10)
        data    = resp.json()
        schemes = data.get("schemes", [])
        lang    = LANG_MAP.get(language, "hi")
        if not schemes:
            return "No schemes found."
        result = f"🏛️ Govt Schemes for {state}\n{'─'*60}\n"
        for i, s in enumerate(schemes, 1):
            result += f"\n{i}. {s['name']}\n   💰 {s['benefit']}\n   🔗 {s['apply_url']}\n"
        return result
    except Exception as e:
        return f"❌ Error: {e}"


def ui_register(name, phone, state, district, language):
    """Calls /farmer/register API internally."""
    if not name or not phone:
        return "❌ Please fill name and phone."
    lang = LANG_MAP.get(language, "hi")
    try:
        resp = http_requests.post(
            "http://localhost:8001/farmer/register",
            data={"name": name, "phone": phone, "state": state,
                  "district": district, "language": lang},
            timeout=10
        )
        data = resp.json()
        if resp.status_code == 200:
            return f"✅ {data.get('message')}\nFarmer ID: {data.get('farmer_id')}\nLanguage: {language}"
        return f"❌ {data.get('detail', 'Error')}"
    except Exception as e:
        return f"❌ Error: {e}"


def ui_health():
    """Calls /health API."""
    try:
        resp = http_requests.get("http://localhost:8001/health", timeout=5)
        data = resp.json()
        return (f"✅ API Online\n"
                f"Model loaded : {data.get('model_loaded')}\n"
                f"Version      : {data.get('version')}\n"
                f"Status       : {data.get('status')}")
    except:
        return "❌ API Offline"

# ── CSS ─────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Outfit:wght@300;400;500;600&display=swap');

body, .gradio-container {
    background: #0b150c !important;
    font-family: 'Outfit', sans-serif !important;
    color: #f0ebe0 !important;
}
.gradio-container {
    background:
        radial-gradient(ellipse 70% 50% at 15% 15%, rgba(44,106,53,.18) 0%, transparent 65%),
        radial-gradient(ellipse 50% 70% at 85% 85%, rgba(139,195,74,.07) 0%, transparent 65%),
        #0b150c !important;
    max-width: 100% !important;
    padding: 0 !important;
}

/* Hero */
.hero { text-align:center; padding:36px 24px 22px; background:linear-gradient(180deg,rgba(44,106,53,.14) 0%,transparent 100%); border-bottom:1px solid rgba(76,175,80,.1); margin-bottom:0; }
.hero-title { font-family:'Playfair Display',serif !important; font-size:2.8rem !important; font-weight:900 !important; background:linear-gradient(120deg,#8bc34a 0%,#f5c842 50%,#8bc34a 100%); background-size:200% auto; -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important; background-clip:text !important; animation:shimmer 5s linear infinite; line-height:1.1 !important; margin:0 0 4px !important; display:block; }
@keyframes shimmer { 0%{background-position:0% center} 100%{background-position:200% center} }
.hero-sub { color:rgba(240,235,224,.35); font-size:.8rem; letter-spacing:.22em; text-transform:uppercase; margin-bottom:14px; display:block; }
.pills { display:flex; justify-content:center; gap:8px; flex-wrap:wrap; }
.pill { background:rgba(76,175,80,.09); border:1px solid rgba(76,175,80,.2); border-radius:100px; padding:4px 13px; font-size:.74rem; color:#8bc34a; animation:pf 3s ease-in-out infinite alternate; font-family:'Outfit',sans-serif; }
.pill:nth-child(2){animation-delay:.25s}.pill:nth-child(3){animation-delay:.5s}.pill:nth-child(4){animation-delay:.75s}.pill:nth-child(5){animation-delay:1s}
@keyframes pf { 0%{transform:translateY(0)} 100%{transform:translateY(-3px)} }
.back-link { display:inline-block; margin-bottom:8px; padding:4px 14px; background:rgba(76,175,80,.07); border:1px solid rgba(76,175,80,.18); border-radius:100px; font-size:.74rem; color:#8bc34a; text-decoration:none; }
.hint-text { padding:4px 0 10px; color:rgba(240,235,224,.28); font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; font-family:'Outfit',sans-serif; }
.footer { text-align:center; padding:16px; border-top:1px solid rgba(76,175,80,.07); color:rgba(240,235,224,.18); font-size:.74rem; letter-spacing:.04em; }

/* Tabs */
.tab-nav { background:rgba(0,0,0,.3) !important; border-bottom:1px solid rgba(76,175,80,.1) !important; }
.tab-nav button { font-family:'Outfit',sans-serif !important; font-weight:500 !important; color:rgba(240,235,224,.3) !important; background:transparent !important; border:none !important; border-bottom:2px solid transparent !important; padding:10px 16px !important; font-size:.78rem !important; letter-spacing:.08em !important; text-transform:uppercase !important; transition:all .25s !important; }
.tab-nav button.selected, .tab-nav button:hover { color:#8bc34a !important; border-bottom-color:#8bc34a !important; background:rgba(76,175,80,.04) !important; }

/* Remove oversized padding from Gradio blocks */
.gr-block, .gr-box, .block, .contain, .gap, .form { gap:8px !important; padding:0 !important; }
.padded { padding:8px !important; }

/* Inputs - natural size, no forced height */
input, textarea, select {
    background:rgba(255,255,255,.05) !important;
    border:1px solid rgba(76,175,80,.18) !important;
    border-radius:8px !important;
    color:#f0ebe0 !important;
    font-family:'Outfit',sans-serif !important;
    font-size:.9rem !important;
    transition:border-color .25s !important;
    padding:8px 12px !important;
    min-height:unset !important;
    height:auto !important;
}
input:focus, textarea:focus { border-color:#4caf50 !important; outline:none !important; box-shadow:0 0 0 2px rgba(76,175,80,.1) !important; }

/* Buttons - compact */
button.primary, .gr-button {
    font-family:'Outfit',sans-serif !important;
    font-weight:600 !important;
    background:linear-gradient(135deg,#235e2b,#3d9e45) !important;
    border:none !important;
    border-radius:8px !important;
    color:#fff !important;
    padding:10px 20px !important;
    cursor:pointer !important;
    transition:all .25s ease !important;
    text-transform:uppercase !important;
    letter-spacing:.08em !important;
    font-size:.8rem !important;
    min-height:unset !important;
    height:auto !important;
}
button.primary:hover, .gr-button:hover { transform:translateY(-1px) !important; box-shadow:0 6px 20px rgba(76,175,80,.3) !important; }

/* Labels - compact */
label, .label-wrap span { font-family:'Outfit',sans-serif !important; color:rgba(240,235,224,.4) !important; font-size:.72rem !important; letter-spacing:.1em !important; text-transform:uppercase !important; margin-bottom:2px !important; }

/* Image upload - natural size */
.gr-image, [data-testid="image"] { border:2px dashed rgba(76,175,80,.22) !important; border-radius:12px !important; background:rgba(76,175,80,.02) !important; }

/* Textboxes - natural height */
.gr-textbox, [data-testid="textbox"] { background:rgba(0,0,0,.25) !important; border:1px solid rgba(76,175,80,.1) !important; border-radius:10px !important; }

/* Audio */
.gr-audio { background:rgba(15,30,16,.7) !important; border:1px solid rgba(76,175,80,.14) !important; border-radius:10px !important; }

/* Dropdown */
ul[role="listbox"] { background:#112015 !important; border:1px solid rgba(76,175,80,.2) !important; border-radius:8px !important; }
ul[role="listbox"] li:hover { background:rgba(76,175,80,.12) !important; color:#8bc34a !important; }

/* Scrollbar */
::-webkit-scrollbar { width:3px; } ::-webkit-scrollbar-track { background:transparent; } ::-webkit-scrollbar-thumb { background:#2d5e33; border-radius:3px; }

/* Info card */
.info-card { margin-top:10px; padding:14px 16px; background:rgba(76,175,80,.04); border:1px solid rgba(76,175,80,.1); border-radius:10px; }
.info-card-title { font-size:.88rem; font-weight:600; color:#8bc34a; margin-bottom:8px; font-family:'Outfit',sans-serif; }
.info-card-body { color:rgba(240,235,224,.35); font-size:.8rem; line-height:1.8; font-family:'Outfit',sans-serif; }

/* ── FIX OVERSIZED CONTAINERS ── */
/* Remove extra height from row containers */
.gr-row, .row { align-items:flex-start !important; }

/* Dropdown — collapse to just the select element */
.gr-dropdown, [data-testid="dropdown"] { height:auto !important; min-height:unset !important; }
.gr-dropdown > div, [data-testid="dropdown"] > div { height:auto !important; min-height:unset !important; }

/* Collapse all block wrappers to content height */
.block, .gr-block, .svelte-1gfkn6j, .wrap, .container, .component-wrapper { 
    height:auto !important; 
    min-height:unset !important; 
    flex:0 0 auto !important;
}

/* Textbox — only as tall as lines dictate */
.gr-textbox, [data-testid="textbox"] { height:auto !important; min-height:unset !important; }
.gr-textbox textarea, [data-testid="textbox"] textarea { resize:vertical !important; }

/* Row — don't stretch children to match tallest */
.gr-form, .form { height:auto !important; align-items:flex-start !important; }

/* Specific fix for the column that holds dropdowns */
.gr-column, [data-testid="column"] { height:auto !important; align-self:flex-start !important; }

/* Gradio internal layout fixes */
div.svelte-vt1mxs { height:auto !important; min-height:unset !important; }
div.svelte-1gfkn6j { height:auto !important; }
"""



# ── Build Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="अन्नदाता AI") as gradio_app:

    gr.HTML("""
    <div class="hero">
        <a href="/" class="back-link">← Home</a>
        <div class="hero-title">🌾 अन्नदाता AI</div>
        <div class="hero-sub">Intelligent Farming Assistant for Bharat</div>
        <div class="pills">
            <div class="pill">🌱 140M+ Farmers</div>
            <div class="pill">🔬 99.81% Accuracy</div>
            <div class="pill">🗣️ 11 Languages</div>
            <div class="pill">📊 Live Prices</div>
            <div class="pill">🏛️ 6 Schemes</div>
        </div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("📸  Crop Doctor"):
            gr.HTML("<p class='hint-text'>Upload crop leaf photo → instant disease detection via API</p>")
            with gr.Row():
                with gr.Column(scale=1):
                    img_in   = gr.Image(type="pil", label="📷 Crop Leaf Photo", height=200)
                    lang_in  = gr.Dropdown(choices=LANG_OPTIONS, value="Hindi (हिंदी)", label="🗣️ Your Language")
                    detect_b = gr.Button("🔍 Detect Disease", variant="primary")
                with gr.Column(scale=1):
                    res_out  = gr.Textbox(label="🌿 Diagnosis", lines=4, interactive=False)
                    trt_out  = gr.Textbox(label="💊 Treatment Advice", lines=4, interactive=False)
                    top5_out = gr.Textbox(label="📊 Confidence Scores", lines=4, interactive=False)
                    aud_out  = gr.Audio(label="🔊 Listen in Your Language", autoplay=True)
            detect_b.click(fn=ui_predict, inputs=[img_in, lang_in],
                           outputs=[res_out, trt_out, top5_out, aud_out])

        with gr.Tab("📈  Mandi Prices"):
            gr.HTML("<p class='hint-text'>Live wholesale mandi prices via API</p>")
            with gr.Row():
                st_in   = gr.Dropdown(choices=["All States"]+STATES, value="Gujarat", label="📍 State")
                com_in  = gr.Textbox(label="🌾 Commodity (optional)", placeholder="e.g. Tomato, Potato...")
            price_b = gr.Button("📊 Get Live Prices", variant="primary")
            price_o = gr.Textbox(label="💹 Mandi Prices", lines=8, interactive=False)
            price_b.click(fn=ui_prices, inputs=[st_in, com_in], outputs=price_o)

        with gr.Tab("🏛️  Govt Schemes"):
            gr.HTML("<p class='hint-text'>Government schemes matched via API</p>")
            with gr.Row():
                sch_st  = gr.Dropdown(choices=STATES, value="Gujarat", label="📍 Your State")
                sch_lg  = gr.Dropdown(choices=LANG_OPTIONS, value="Hindi (हिंदी)", label="🗣️ Language")
            sch_b = gr.Button("🔍 Find My Schemes", variant="primary")
            sch_o = gr.Textbox(label="🏛️ Eligible Schemes", lines=8, interactive=False)
            sch_b.click(fn=ui_schemes, inputs=[sch_st, sch_lg], outputs=sch_o)

        with gr.Tab("👤  My Profile"):
            gr.HTML("<p class='hint-text'>Register via API to save history</p>")
            with gr.Row():
                with gr.Column():
                    r_name = gr.Textbox(label="👤 Full Name", placeholder="e.g. Rameshbhai Patel")
                    r_ph   = gr.Textbox(label="📱 Phone", placeholder="e.g. 9876543210")
                    r_st   = gr.Dropdown(choices=STATES, value="Gujarat", label="📍 State")
                    r_dist = gr.Textbox(label="🏘️ District", placeholder="e.g. Anand")
                    r_lg   = gr.Dropdown(choices=LANG_OPTIONS, value="Gujarati (ગુજરાતી)", label="🗣️ Language")
                    r_btn  = gr.Button("✅ Register", variant="primary")
                with gr.Column():
                    r_out  = gr.Textbox(label="📋 Status", lines=4, interactive=False)
                    gr.HTML("""
                    <div style='margin-top:12px;padding:16px;background:rgba(76,175,80,.05);border:1px solid rgba(76,175,80,.1);border-radius:12px'>
                        <div style='font-size:.9rem;font-weight:600;color:#8bc34a;margin-bottom:8px'>Why Register?</div>
                        <div style='color:rgba(253,248,240,.35);font-size:.8rem;line-height:1.8'>
                            ✓ Save all crop disease history<br>
                            ✓ Personalized scheme recommendations<br>
                            ✓ Price alerts for your crops<br>
                            ✓ Advice in your regional language
                        </div>
                    </div>""")
            r_btn.click(fn=ui_register, inputs=[r_name, r_ph, r_st, r_dist, r_lg], outputs=r_out)

        with gr.Tab("⚙️  API Status"):
            gr.HTML("<p class='hint-text'>Check API health and all active routes</p>")
            h_btn = gr.Button("🔄 Check API Health", variant="primary")
            h_out = gr.Textbox(label="Status", lines=4, interactive=False)
            h_btn.click(fn=ui_health, outputs=h_out)
            gr.HTML("""
            <div style='margin-top:14px;padding:18px;background:rgba(76,175,80,.05);border:1px solid rgba(76,175,80,.1);border-radius:12px'>
                <div style='font-size:.85rem;font-weight:600;color:#8bc34a;margin-bottom:10px'>All API Routes</div>
                <div style='font-family:monospace;font-size:.78rem;color:rgba(253,248,240,.45);line-height:2'>
                    GET  /health           → Server health check<br>
                    POST /predict          → Disease detection<br>
                    GET  /prices           → Live mandi prices<br>
                    GET  /schemes          → Govt schemes<br>
                    POST /farmer/register  → Register farmer<br>
                    GET  /farmer/{id}/history → Prediction history<br>
                    POST /prices/alert     → Set price alert<br>
                    GET  /diseases         → All 15 diseases<br>
                    GET  /docs             → Full API documentation
                </div>
            </div>""")

    gr.HTML('<div class="footer">अन्नदाता AI — Built for India\'s 140 Million Farmers &nbsp;·&nbsp; जय किसान &nbsp;·&nbsp; जय अन्नदाता 🌾</div>')

# ── Mount Gradio on FastAPI ─────────────────────────────────────────────────────
app = gr.mount_gradio_app(api, gradio_app, path="/app", css=CSS)

# ── Start server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*55)
    print("  🌾 अन्नदाता AI — Unified Server")
    print("="*55)
    print("\n  🏠 Landing Page : http://localhost:8001/")
    print("  📱 Farmer App   : http://localhost:8001/app")
    print("  📋 API Docs     : http://localhost:8001/docs")
    print("\n" + "="*55 + "\n")
    uvicorn.run("run:app", host="0.0.0.0", port=8001, reload=False)