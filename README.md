# 🌾 अन्नदाता AI — Phase 1: Crop Disease Detection

> "अन्नदाता" — The one who gives food. Built for India's 140 million farmers.

Detect 38 crop diseases from a single photo. Works offline. Responds in Hindi.

---

## Project Structure

```
annadata/
├── phase1/
│   ├── model/
│   │   └── model.py          ← MobileNetV3-Small (2.5MB, mobile-ready)
│   ├── utils/
│   │   └── dataset.py        ← PlantVillage loader, transforms, class info
│   ├── api/
│   │   └── server.py         ← FastAPI REST server (POST /predict)
│   ├── train.py              ← Two-phase training script
│   └── demo.py               ← Gradio browser demo (test before mobile)
├── data/
│   └── PlantVillage/         ← Dataset goes here (see Step 2)
├── checkpoints/
│   ├── best_model.pt         ← Best checkpoint (created after training)
│   └── training_log.json     ← Loss/accuracy per epoch
└── requirements.txt
```

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download PlantVillage dataset

**Option A: Kaggle CLI (recommended)**
```bash
# Install and configure kaggle
pip install kaggle
# Place kaggle.json from kaggle.com/settings → API → Create Token into ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset (~1.1GB)
kaggle datasets download -d emmarex/plantdisease -p data/ --unzip
```

**Option B: Manual download**
- Go to https://www.kaggle.com/datasets/emmarex/plantdisease
- Download and unzip into `data/PlantVillage/`

Verify structure:
```
data/PlantVillage/
  Tomato___Early_blight/   ← folder per class
  Potato___Late_blight/
  Corn_(maize)___healthy/
  ...38 folders total
```

### Step 3 — Train the model
```bash
# From project root
python phase1/train.py
```

Training takes ~20 min on GPU, ~2 hours on CPU.
Expected accuracy: **94–96%** on validation set.

Training output:
```
── Phase A: Head-only training ──
[Epoch 01/20] train_loss=0.8231 train_acc=0.7812 | val_loss=0.4201 val_acc=0.9102 | 180s
...
── Phase B: Full fine-tuning ──
[Epoch 06/20] train_loss=0.2341 train_acc=0.9521 | val_loss=0.1803 val_acc=0.9601 | 210s
✓ Best model saved (val_acc=0.9601)
```

### Step 4 — Test in browser (Gradio demo)
```bash
python phase1/demo.py
# Open http://localhost:7860
# Upload a leaf photo → see disease + treatment in Hindi
```

### Step 5 — Run the API server
```bash
uvicorn phase1.api.server:app --reload --port 8000
```

Test with curl:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@leaf.jpg" \
  -F "lang=hi"
```

Response:
```json
{
  "disease_label": "टमाटर का अगेती झुलसा",
  "confidence_pct": 94.3,
  "is_healthy": false,
  "treatment": "Spray Copper Oxychloride 50% WP @ 3g/L. Maintain plant spacing for airflow.",
  "top5_predictions": [...]
}
```

API docs: http://localhost:8000/docs

---

## Model Details

| Property         | Value                          |
|------------------|-------------------------------|
| Architecture     | MobileNetV3-Small              |
| Parameters       | ~2.5M (mobile-ready)           |
| Input size       | 224 × 224 RGB                  |
| Output classes   | 38 (PlantVillage)              |
| Training strategy| Phase A: head only → Phase B: full fine-tune |
| Target accuracy  | 94–96% val accuracy            |
| Inference time   | ~50ms on CPU, ~8ms on GPU      |

---

## Supported Crops & Diseases (38 classes)

| Crop    | Diseases Detected                              |
|---------|------------------------------------------------|
| Tomato  | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, TYLCV, Mosaic Virus, Healthy |
| Potato  | Early Blight, Late Blight, Healthy             |
| Maize   | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Apple   | Apple Scab, Black Rot, Cedar Rust, Healthy     |
| Grape   | Black Rot, Esca, Leaf Blight, Healthy          |
| + more  | Cherry, Peach, Pepper, Soybean, Strawberry...  |

---

## Roadmap

- **Phase 1** ✅ Crop disease detection (this folder)
- **Phase 2** 🔜 Mandi price prediction (AGMARKNET API + XGBoost)
- **Phase 3** 🔜 Voice AI in 22 Indian languages (Bhashini API)
- **Phase 4** 🔜 Govt scheme matcher + Soil Health Card OCR
- **Phase 5** 🔜 React Native mobile app (offline-capable)

---

*जय किसान। जय अन्नदाता।*
