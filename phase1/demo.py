"""
अन्नदाता AI — Phase 1
Gradio demo: test the model visually in your browser before building the mobile app.

Run:
  python phase1/demo.py

Opens at: http://localhost:7860
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase1.model.model import build_model, load_checkpoint
from phase1.utils.dataset import CLASS_INFO, TREATMENT_ADVICE, get_transforms

import gradio as gr

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
DEVICE = "cpu"

# Load model
if CHECKPOINT_PATH.exists():
    model = load_checkpoint(CHECKPOINT_PATH, num_classes=38, device=DEVICE)
else:
    print("No checkpoint found — using untrained model for UI demo.")
    model = build_model(num_classes=38, pretrained=False)
    model.eval()

class_names = list(CLASS_INFO.keys())
transform   = get_transforms(augment=False)


@torch.no_grad()
def predict(image: Image.Image, language: str):
    if image is None:
        return "कृपया एक फोटो अपलोड करें। / Please upload an image.", "", ""

    tensor  = transform(image).unsqueeze(0)
    logits  = model(tensor)
    probs   = F.softmax(logits, dim=1)[0]
    top5_p, top5_i = torch.topk(probs, 5)

    lang        = "hi" if language == "हिंदी" else "en"
    top_class   = class_names[top5_i[0].item()]
    confidence  = top5_p[0].item() * 100
    info        = CLASS_INFO.get(top_class, {"en": top_class, "hi": top_class})
    is_healthy  = "healthy" in top_class.lower()
    label       = info.get(lang, info["en"])
    treatment   = TREATMENT_ADVICE.get(top_class, "अपने नजदीकी कृषि विज्ञान केंद्र से मिलें। / Consult your local KVK.")

    # Result
    emoji  = "✅" if is_healthy else "⚠️"
    result = f"{emoji}  {label}\n\nविश्वास / Confidence: {confidence:.1f}%"

    # Treatment
    advice = "🌿 फसल स्वस्थ है!\nFasal swasth hai!\n\nकोई उपचार की जरूरत नहीं।\nNo treatment needed." if is_healthy else f"💊 उपचार / Treatment:\n\n{treatment}"

    # Top-5
    top5_text = "Top 5 Predictions:\n"
    for p, i in zip(top5_p, top5_i):
        c = class_names[i.item()]
        l = CLASS_INFO.get(c, {}).get(lang, c)
        top5_text += f"  • {l}: {p.item()*100:.1f}%\n"

    return result, advice, top5_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="अन्नदाता AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌾 अन्नदाता AI
    ### फसल रोग पहचान | Crop Disease Detection
    *अपनी फसल की पत्ती की फोटो खींचें और रोग जानें।*
    *Take a photo of your crop leaf and detect the disease instantly.*
    """)

    with gr.Row():
        with gr.Column():
            img_input  = gr.Image(type="pil", label="📸 फोटो अपलोड करें / Upload Photo")
            lang_input = gr.Radio(["English", "हिंदी"], value="हिंदी", label="भाषा / Language")
            btn        = gr.Button("🔍 रोग पहचानें / Detect Disease", variant="primary")
        with gr.Column():
            out_result    = gr.Textbox(label="🌿 परिणाम / Result", lines=4)
            out_treatment = gr.Textbox(label="💊 उपचार / Treatment", lines=6)
            out_top5      = gr.Textbox(label="📊 Top 5 Predictions", lines=7)

    btn.click(fn=predict, inputs=[img_input, lang_input], outputs=[out_result, out_treatment, out_top5])

    gr.Markdown("""
    ---
    **अन्नदाता AI** — Built for India's 140 million farmers.
    Phase 1: Crop Disease Detection | Phase 2: Mandi Price Predictor | Phase 3: Voice AI in 22 Languages
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
