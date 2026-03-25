"""
अन्नदाता AI — Phase 1
FastAPI inference server.

Endpoints:
  POST /predict        — upload crop image → get disease + treatment in Hindi/English
  GET  /health         — server health check
  GET  /classes        — list all 38 supported diseases

Run:
  uvicorn phase1.api.server:app --reload --port 8000

Test:
  curl -X POST http://localhost:8000/predict \
    -F "file=@/path/to/leaf.jpg" \
    -F "lang=hi"
"""

import io
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from phase1.model.model import load_checkpoint, build_model
from phase1.utils.dataset import CLASS_INFO, TREATMENT_ADVICE, get_transforms, IMG_SIZE

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="अन्नदाता AI",
    description="Crop disease detection for Indian farmers — 38 diseases, Hindi + English",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Model loading (done once at startup) ─────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
DEVICE          = "cpu"   # CPU for API server (mobile-friendly)
NUM_CLASSES     = 38

model      = None
class_names = None

@app.on_event("startup")
async def load_model():
    global model, class_names
    if CHECKPOINT_PATH.exists():
        model = load_checkpoint(CHECKPOINT_PATH, num_classes=NUM_CLASSES, device=DEVICE)
        print("[अन्नदाता AI] Model loaded for inference ✓")
    else:
        # Demo mode — random weights (for testing the API before training)
        print("[अन्नदाता AI] WARNING: No checkpoint found. Running in DEMO mode.")
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
    class_names = list(CLASS_INFO.keys())


# ── Inference helper ──────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    transform = get_transforms(augment=False)
    tensor    = transform(img).unsqueeze(0)   # shape: (1, 3, 224, 224)
    return tensor


@torch.no_grad()
def run_inference(tensor: torch.Tensor, lang: str = "en") -> dict:
    logits      = model(tensor.to(DEVICE))
    probs       = F.softmax(logits, dim=1)[0]
    top5_probs, top5_idx = torch.topk(probs, k=5)

    top_class   = class_names[top5_idx[0].item()]
    confidence  = round(top5_probs[0].item() * 100, 2)
    info        = CLASS_INFO.get(top_class, {"en": top_class, "hi": top_class})
    is_healthy  = "healthy" in top_class.lower()
    treatment   = TREATMENT_ADVICE.get(top_class, "Consult your local Krishi Vigyan Kendra (KVK).") if not is_healthy else None

    top5 = [
        {
            "class":      class_names[i.item()],
            "label":      CLASS_INFO.get(class_names[i.item()], {}).get(lang, class_names[i.item()]),
            "confidence": round(p.item() * 100, 2),
        }
        for p, i in zip(top5_probs, top5_idx)
    ]

    return {
        "disease_class": top_class,
        "disease_label": info.get(lang, info["en"]),
        "confidence_pct": confidence,
        "is_healthy": is_healthy,
        "treatment": treatment,
        "treatment_hi": _translate_treatment(treatment) if lang == "hi" and treatment else treatment,
        "top5_predictions": top5,
        "language": lang,
    }


def _translate_treatment(text: str | None) -> str | None:
    """
    Placeholder for Bhashini API translation (Phase 3).
    For now returns the English text with a note.
    """
    if text is None:
        return None
    return f"[Translation via Bhashini coming in Phase 3] {text}"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "app": "अन्नदाता AI"}


@app.get("/classes")
async def get_classes(lang: str = "en"):
    return {
        "total": len(class_names),
        "classes": [
            {"id": i, "class": c, "label": CLASS_INFO.get(c, {}).get(lang, c)}
            for i, c in enumerate(class_names)
        ],
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Crop leaf image (JPG/PNG)"),
    lang: str        = Form(default="en", description="Response language: 'en' or 'hi'"),
):
    """
    Upload a photo of a crop leaf.
    Returns: disease name, confidence, treatment advice — in Hindi or English.

    Example response (Hindi):
    {
      "disease_label": "टमाटर का अगेती झुलसा",
      "confidence_pct": 94.3,
      "is_healthy": false,
      "treatment": "Apply Copper Oxychloride 50% WP @ 3g/L ..."
    }
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image (JPG/PNG).")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Max 10MB.")

    tensor = preprocess_image(image_bytes)
    result = run_inference(tensor, lang=lang)
    return result
