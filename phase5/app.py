"""
अन्नदाता AI — Phase 5
Gradio UI linked to FastAPI backend on port 8001.

Architecture:
    Farmer → Gradio UI (7860) → FastAPI API (8001) → ML Model + PostgreSQL

Run:
    1. Start API first:  python -m uvicorn phase1.api.api:app --port 8001
    2. Start UI:         python phase5/app.py
"""

import sys
import io
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent.parent))

from phase3.voice import text_to_speech
import gradio as gr

# ── API config ────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8001"

# ── Language map ──────────────────────────────────────────────────────────────
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


# ── Functions calling FastAPI ─────────────────────────────────────────────────
def predict_disease(image, language):
    if image is None:
        return "❌ Please upload an image.", "", "", None

    lang = LANG_MAP.get(language, "hi")

    try:
        # Convert PIL image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # Call FastAPI /predict
        response = requests.post(
            f"{API_BASE}/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
            data={"lang": lang},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        label      = data.get("disease_label", "Unknown")
        confidence = data.get("confidence_pct", 0)
        is_healthy = data.get("is_healthy", False)
        treatment  = data.get("treatment", "Consult your nearest KVK.")
        top5       = data.get("top5_predictions", [])

        emoji  = "✅" if is_healthy else "⚠️"
        result = f"{emoji}  {label}\n\nConfidence: {confidence}%\nLanguage: {language}"

        treatment_text = f"💊 Treatment:\n{treatment}" if not is_healthy else "✅ Crop is healthy! No treatment needed."

        top5_text = "📊 Top 5 Predictions:\n"
        for p in top5:
            top5_text += f"  • {p['label']}: {p['confidence']}%\n"

        # Generate voice
        Path("output/voice").mkdir(parents=True, exist_ok=True)
        audio_path = f"output/voice/result_{lang}.mp3"
        try:
            voice_text = f"{label}. {treatment if not is_healthy else 'Your crop is healthy.'}"
            text_to_speech(voice_text, lang=lang, output_path=audio_path)
        except Exception as e:
            print(f"Voice error: {e}")
            audio_path = None

        return result, treatment_text, top5_text, audio_path

    except requests.exceptions.ConnectionError:
        return "❌ API not running! Start it with:\npython -m uvicorn phase1.api.api:app --port 8001", "", "", None
    except Exception as e:
        return f"❌ Error: {e}", "", "", None


def get_mandi_prices(state, commodity):
    try:
        params = {}
        if state != "All States": params["state"] = state
        if commodity:             params["commodity"] = commodity

        response = requests.get(f"{API_BASE}/prices", params=params, timeout=10)
        response.raise_for_status()
        data   = response.json()
        prices = data.get("prices", [])

        if not prices:
            return "No price data found. Run: python phase2/scheduler.py"

        result = f"📈 Live Mandi Prices — {state}\n{'─'*60}\n\n"
        for p in prices:
            result += f"🌾 {p['commodity']:25} | {p['market']:30} | ₹{p['modal_price']:>8}/qtl | {p['date']}\n"
        return result

    except requests.exceptions.ConnectionError:
        return "❌ API not running! Start it with:\npython -m uvicorn phase1.api.api:app --port 8001"
    except Exception as e:
        return f"❌ Error: {e}"


def get_schemes(state, language):
    try:
        response = requests.get(f"{API_BASE}/schemes", params={"state": state}, timeout=10)
        response.raise_for_status()
        data    = response.json()
        schemes = data.get("schemes", [])
        lang    = LANG_MAP.get(language, "hi")

        if not schemes:
            return "No schemes found."

        result = f"🏛️ Govt Schemes for {state}\n{'─'*60}\n"
        for i, s in enumerate(schemes, 1):
            result += f"\n{i}. {s['name']}\n"
            result += f"   💰 {s['benefit']}\n"
            result += f"   🔗 {s['apply_url']}\n"
        return result

    except requests.exceptions.ConnectionError:
        return "❌ API not running! Start it with:\npython -m uvicorn phase1.api.api:app --port 8001"
    except Exception as e:
        return f"❌ Error: {e}"


def register_farmer(name, phone, state, district, language):
    if not name or not phone:
        return "❌ Please fill name and phone."

    lang = LANG_MAP.get(language, "hi")
    try:
        response = requests.post(
            f"{API_BASE}/farmer/register",
            data={"name": name, "phone": phone, "state": state,
                  "district": district, "language": lang},
            timeout=10,
        )
        data = response.json()
        if response.status_code == 200:
            return f"✅ {data.get('message', 'Registered!')}\nFarmer ID: {data.get('farmer_id')}\nLanguage: {language}"
        else:
            return f"❌ {data.get('detail', 'Registration failed.')}"

    except requests.exceptions.ConnectionError:
        return "❌ API not running! Start it with:\npython -m uvicorn phase1.api.api:app --port 8001"
    except Exception as e:
        return f"❌ Error: {e}"


def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        data = r.json()
        return f"✅ API Online\nModel loaded: {data.get('model_loaded')}\nVersion: {data.get('version')}"
    except:
        return "❌ API Offline — Run: python -m uvicorn phase1.api.api:app --port 8001"


# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Outfit:wght@300;400;500;600&display=swap');
body,.gradio-container{background:#0e1a0f!important;font-family:'Outfit',sans-serif!important;color:#fdf8f0!important}
.gradio-container{background:radial-gradient(ellipse 80% 60% at 10% 20%,rgba(44,106,53,.15) 0%,transparent 70%),radial-gradient(ellipse 60% 80% at 90% 80%,rgba(245,200,66,.06) 0%,transparent 70%),#0e1a0f!important}
.annadata-hero{text-align:center;padding:48px 24px 32px;background:linear-gradient(180deg,rgba(44,106,53,.12) 0%,transparent 100%);border-bottom:1px solid rgba(76,175,80,.1);margin-bottom:8px}
.annadata-title{font-family:'Playfair Display',serif!important;font-size:3rem!important;font-weight:900!important;background:linear-gradient(135deg,#8bc34a,#f5c842,#8bc34a);background-size:200% auto;-webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;background-clip:text!important;animation:shimmer 4s linear infinite;line-height:1.1!important;margin-bottom:8px!important}
@keyframes shimmer{0%{background-position:0% center}100%{background-position:200% center}}
.annadata-sub{color:rgba(253,248,240,.4);font-size:.9rem;letter-spacing:.2em;text-transform:uppercase;margin-bottom:20px}
.stat-pills{display:flex;justify-content:center;gap:12px;flex-wrap:wrap}
.stat-pill{background:rgba(76,175,80,.08);border:1px solid rgba(76,175,80,.2);border-radius:100px;padding:6px 16px;font-size:.8rem;color:#8bc34a;animation:pillFloat 3s ease-in-out infinite alternate}
.stat-pill:nth-child(2){animation-delay:.3s}.stat-pill:nth-child(3){animation-delay:.6s}.stat-pill:nth-child(4){animation-delay:.9s}.stat-pill:nth-child(5){animation-delay:1.2s}
@keyframes pillFloat{0%{transform:translateY(0)}100%{transform:translateY(-4px)}}
.tab-nav button{font-family:'Outfit',sans-serif!important;font-weight:500!important;color:rgba(253,248,240,.4)!important;background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;padding:14px 20px!important;font-size:.85rem!important;letter-spacing:.08em!important;text-transform:uppercase!important;transition:all .3s ease!important}
.tab-nav button.selected,.tab-nav button:hover{color:#8bc34a!important;border-bottom-color:#8bc34a!important}
input,textarea,select,.gr-input,.gr-textbox textarea{background:rgba(255,255,255,.04)!important;border:1px solid rgba(76,175,80,.15)!important;border-radius:10px!important;color:#fdf8f0!important;font-family:'Outfit',sans-serif!important;transition:border-color .3s!important}
input:focus,textarea:focus,select:focus{border-color:#4caf50!important;box-shadow:0 0 0 3px rgba(76,175,80,.1)!important}
.gr-button,button.primary{font-family:'Outfit',sans-serif!important;font-weight:600!important;background:linear-gradient(135deg,#2d6a35,#4caf50)!important;border:none!important;border-radius:10px!important;color:white!important;padding:12px 28px!important;cursor:pointer!important;transition:all .3s ease!important;text-transform:uppercase!important;letter-spacing:.08em!important;font-size:.85rem!important}
.gr-button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(76,175,80,.3)!important}
label,.gr-label,.label-wrap span{font-family:'Outfit',sans-serif!important;color:rgba(253,248,240,.5)!important;font-size:.8rem!important;letter-spacing:.1em!important;text-transform:uppercase!important}
.gr-image,.image-container{border:2px dashed rgba(76,175,80,.2)!important;border-radius:16px!important;background:rgba(76,175,80,.02)!important;transition:border-color .3s!important}
.gr-image:hover{border-color:#4caf50!important}
.gr-textbox{border:1px solid rgba(76,175,80,.12)!important;border-radius:12px!important;background:rgba(0,0,0,.25)!important}
.gr-audio{background:rgba(20,36,21,.6)!important;border:1px solid rgba(76,175,80,.15)!important;border-radius:12px!important}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:#2d6a35;border-radius:4px}
.annadata-footer{text-align:center;padding:24px;border-top:1px solid rgba(76,175,80,.08);color:rgba(253,248,240,.2);font-size:.8rem;letter-spacing:.05em}
.api-status{display:inline-block;padding:6px 14px;border-radius:100px;font-size:.75rem;background:rgba(76,175,80,.1);border:1px solid rgba(76,175,80,.2);color:#8bc34a;margin-bottom:12px}
"""

# ── Build UI ───────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="अन्नदाता AI") as app:

    gr.HTML("""
    <div class="annadata-hero">
        <div class="annadata-title">🌾 अन्नदाता AI</div>
        <div class="annadata-sub">Intelligent Farming Assistant for Bharat</div>
        <div class="stat-pills">
            <div class="stat-pill">🌱 140M+ Farmers</div>
            <div class="stat-pill">🔬 99.81% Accuracy</div>
            <div class="stat-pill">🗣️ 11 Languages</div>
            <div class="stat-pill">📊 Live Prices</div>
            <div class="stat-pill">🏛️ 6 Schemes</div>
        </div>
    </div>
    """)

    with gr.Tabs():

        # Tab 1: Disease Detection
        with gr.Tab("📸  Crop Doctor"):
            gr.HTML("<div style='padding:6px 0 16px;color:rgba(253,248,240,.35);font-size:.8rem;letter-spacing:.15em;text-transform:uppercase'>Upload crop leaf photo for instant AI disease detection</div>")
            with gr.Row():
                with gr.Column(scale=1):
                    img_input  = gr.Image(type="pil", label="📷 Crop Leaf Photo", height=260)
                    lang_input = gr.Dropdown(choices=LANG_OPTIONS, value="Hindi (हिंदी)", label="🗣️ Your Language")
                    detect_btn = gr.Button("🔍 Detect Disease", variant="primary")
                with gr.Column(scale=1):
                    result_out    = gr.Textbox(label="🌿 Diagnosis", lines=4, interactive=False)
                    treatment_out = gr.Textbox(label="💊 Treatment Advice", lines=4, interactive=False)
                    top5_out      = gr.Textbox(label="📊 Confidence Scores", lines=7, interactive=False)
                    audio_out     = gr.Audio(label="🔊 Listen in Your Language", autoplay=True)
            detect_btn.click(
                fn=predict_disease,
                inputs=[img_input, lang_input],
                outputs=[result_out, treatment_out, top5_out, audio_out]
            )

        # Tab 2: Mandi Prices
        with gr.Tab("📈  Mandi Prices"):
            gr.HTML("<div style='padding:6px 0 16px;color:rgba(253,248,240,.35);font-size:.8rem;letter-spacing:.15em;text-transform:uppercase'>Live wholesale mandi prices from across India</div>")
            with gr.Row():
                state_input     = gr.Dropdown(choices=["All States"]+STATES, value="Gujarat", label="📍 State")
                commodity_input = gr.Textbox(label="🌾 Commodity (optional)", placeholder="e.g. Tomato, Potato...")
            price_btn = gr.Button("📊 Get Live Prices", variant="primary")
            price_out = gr.Textbox(label="💹 Mandi Prices", lines=20, interactive=False)
            price_btn.click(fn=get_mandi_prices, inputs=[state_input, commodity_input], outputs=price_out)

        # Tab 3: Govt Schemes
        with gr.Tab("🏛️  Govt Schemes"):
            gr.HTML("<div style='padding:6px 0 16px;color:rgba(253,248,240,.35);font-size:.8rem;letter-spacing:.15em;text-transform:uppercase'>Find government schemes you are eligible for</div>")
            with gr.Row():
                scheme_state = gr.Dropdown(choices=STATES, value="Gujarat", label="📍 Your State")
                scheme_lang  = gr.Dropdown(choices=LANG_OPTIONS, value="Hindi (हिंदी)", label="🗣️ Language")
            scheme_btn = gr.Button("🔍 Find My Schemes", variant="primary")
            scheme_out = gr.Textbox(label="🏛️ Eligible Schemes", lines=22, interactive=False)
            scheme_btn.click(fn=get_schemes, inputs=[scheme_state, scheme_lang], outputs=scheme_out)

        # Tab 4: Farmer Profile
        with gr.Tab("👤  My Profile"):
            gr.HTML("<div style='padding:6px 0 16px;color:rgba(253,248,240,.35);font-size:.8rem;letter-spacing:.15em;text-transform:uppercase'>Register to save history and get personalized advice</div>")
            with gr.Row():
                with gr.Column():
                    reg_name     = gr.Textbox(label="👤 Full Name", placeholder="e.g. Rameshbhai Patel")
                    reg_phone    = gr.Textbox(label="📱 Phone Number", placeholder="e.g. 9876543210")
                    reg_state    = gr.Dropdown(choices=STATES, value="Gujarat", label="📍 State")
                    reg_district = gr.Textbox(label="🏘️ District", placeholder="e.g. Anand")
                    reg_lang     = gr.Dropdown(choices=LANG_OPTIONS, value="Gujarati (ગુજરાતી)", label="🗣️ Preferred Language")
                    reg_btn      = gr.Button("✅ Register", variant="primary")
                with gr.Column():
                    reg_out = gr.Textbox(label="📋 Status", lines=6, interactive=False)
                    gr.HTML("""
                    <div style='margin-top:16px;padding:20px;background:rgba(76,175,80,.05);border:1px solid rgba(76,175,80,.12);border-radius:12px'>
                        <div style='font-size:1rem;font-weight:600;color:#8bc34a;margin-bottom:10px'>Why Register?</div>
                        <div style='color:rgba(253,248,240,.4);font-size:.85rem;line-height:1.8'>
                            ✓ Save all crop disease history<br>
                            ✓ Personalized scheme recommendations<br>
                            ✓ Price alerts for your crops<br>
                            ✓ Advice in your regional language<br>
                            ✓ Track farm health over time
                        </div>
                    </div>
                    """)
            reg_btn.click(
                fn=register_farmer,
                inputs=[reg_name, reg_phone, reg_state, reg_district, reg_lang],
                outputs=reg_out
            )

        # Tab 5: API Health
        with gr.Tab("⚙️  API Status"):
            gr.HTML("<div style='padding:6px 0 16px;color:rgba(253,248,240,.35);font-size:.8rem;letter-spacing:.15em;text-transform:uppercase'>Check FastAPI backend connection status</div>")
            health_btn = gr.Button("🔄 Check API Health", variant="primary")
            health_out = gr.Textbox(label="API Status", lines=5, interactive=False)
            health_btn.click(fn=check_api_health, outputs=health_out)
            gr.HTML("""
            <div style='margin-top:16px;padding:20px;background:rgba(76,175,80,.05);border:1px solid rgba(76,175,80,.12);border-radius:12px'>
                <div style='font-size:.9rem;font-weight:600;color:#8bc34a;margin-bottom:10px'>How to run both servers:</div>
                <div style='font-family:monospace;font-size:.8rem;color:rgba(253,248,240,.5);line-height:2'>
                    # Terminal 1 — FastAPI (port 8001)<br>
                    python -m uvicorn phase1.api.api:app --port 8001<br><br>
                    # Terminal 2 — Gradio UI (port 7860)<br>
                    python phase5/app.py
                </div>
            </div>
            """)

    gr.HTML('<div class="annadata-footer">अन्नदाता AI — Built for India\'s 140 Million Farmers &nbsp;·&nbsp; जय किसान &nbsp;·&nbsp; जय अन्नदाता 🌾</div>')


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)