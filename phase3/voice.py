"""
अन्नदाता AI — Phase 3
Voice AI — Text to Speech in 22 Indian Languages

Currently uses gTTS (Google TTS) — free, no API key needed.
When Bhashini API is approved, swap in bhashini_tts() below.

Supported languages:
  hi  = Hindi, gu = Gujarati, mr = Marathi
  ta  = Tamil, te = Telugu,   bn = Bengali
  pa  = Punjabi, kn = Kannada, ml = Malayalam
  or  = Odia,   as = Assamese

Run:
    python phase3/voice.py
"""

import os
import sys
from pathlib import Path
from gtts import gTTS

sys.path.append(str(Path(__file__).resolve().parent.parent))

# ── State → Language map ──────────────────────────────────────────────────────
STATE_LANGUAGE_MAP = {
    "Uttar Pradesh":    "hi",
    "Bihar":            "hi",
    "Madhya Pradesh":   "hi",
    "Rajasthan":        "hi",
    "Uttarakhand":      "hi",
    "Himachal Pradesh": "hi",
    "Haryana":          "hi",
    "Delhi":            "hi",
    "Jharkhand":        "hi",
    "Chhattisgarh":     "hi",
    "Gujarat":          "gu",
    "Maharashtra":      "mr",
    "Tamil Nadu":       "ta",
    "Karnataka":        "kn",
    "Kerala":           "ml",
    "Andhra Pradesh":   "te",
    "Telangana":        "te",
    "West Bengal":      "bn",
    "Odisha":           "or",
    "Assam":            "as",
    "Punjab":           "pa",
    "Goa":              "mr",
}

# ── Language display names ────────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "hi": "Hindi (हिंदी)",
    "gu": "Gujarati (ગુજરાતી)",
    "mr": "Marathi (मराठी)",
    "ta": "Tamil (தமிழ்)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "te": "Telugu (తెలుగు)",
    "bn": "Bengali (বাংলা)",
    "or": "Odia (ଓଡ଼ିଆ)",
    "as": "Assamese (অসমীয়া)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
}

# ── Disease messages in each language ────────────────────────────────────────
DISEASE_MESSAGES = {
    "Tomato_Early_blight": {
        "hi": "आपके टमाटर में अगेती झुलसा रोग है। कॉपर ऑक्सीक्लोराइड का छिड़काव करें।",
        "gu": "તમારા ટામેટામાં અગ્રીમ સુકારો રોગ છે. કોપર ઓક્સીક્લોરાઈડ છાંટો.",
        "mr": "तुमच्या टोमॅटोला अगेती करपा रोग आहे. कॉपर ऑक्सीक्लोराइड फवारा.",
        "pa": "ਤੁਹਾਡੇ ਟਮਾਟਰ ਵਿੱਚ ਅਗੇਤੀ ਝੁਲਸ ਰੋਗ ਹੈ। ਕਾਪਰ ਆਕਸੀਕਲੋਰਾਈਡ ਦਾ ਛਿੜਕਾਅ ਕਰੋ।",
        "ta": "உங்கள் தக்காளியில் ஆரம்பகால கருகல் நோய் உள்ளது. காப்பர் ஆக்சிகுளோரைடு தெளிக்கவும்.",
        "te": "మీ టమాటాలో ముందస్తు తెగులు ఉంది. కాపర్ ఆక్సీక్లోరైడ్ పిచికారీ చేయండి.",
        "bn": "আপনার টমেটোতে আগাম ধ্বসা রোগ আছে। কপার অক্সিক্লোরাইড স্প্রে করুন।",
        "kn": "ನಿಮ್ಮ ಟೊಮೇಟೊಗೆ ಮೊದಲ ಸುಟ್ಟ ರೋಗ ಇದೆ. ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಸಿಂಪಡಿಸಿ.",
        "ml": "നിങ്ങളുടെ തക്കാളിക്ക് ആദ്യകാല കരിച്ചിൽ രോഗം ഉണ്ട്. കോപ്പർ ഓക്സിക്ലോറൈഡ് തളിക്കുക.",
    },
    "Potato___Early_blight": {
        "hi": "आपके आलू में अगेती झुलसा रोग है। मैंकोजेब का छिड़काव करें।",
        "gu": "તમારા બટાકામાં અગ્રીમ સુકારો રોગ છે. મેન્કોઝેબ છાંટો.",
        "mr": "तुमच्या बटाट्याला अगेती करपा रोग आहे. मॅन्कोझेब फवारा.",
        "pa": "ਤੁਹਾਡੇ ਆਲੂ ਵਿੱਚ ਅਗੇਤੀ ਝੁਲਸ ਰੋਗ ਹੈ। ਮੈਂਕੋਜ਼ੇਬ ਦਾ ਛਿੜਕਾਅ ਕਰੋ।",
        "ta": "உங்கள் உருளைக்கிழங்கில் ஆரம்பகால கருகல் நோய் உள்ளது. மேன்கோசெப் தெளிக்கவும்.",
        "te": "మీ బంగాళాదుంపలో ముందస్తు తెగులు ఉంది. మాంకోజెబ్ పిచికారీ చేయండి.",
        "bn": "আপনার আলুতে আগাম ধ্বসা রোগ আছে। ম্যানকোজেব স্প্রে করুন।",
        "kn": "ನಿಮ್ಮ ಆಲೂಗಡ್ಡೆಗೆ ಮೊದಲ ಸುಟ್ಟ ರೋಗ ಇದೆ. ಮ್ಯಾಂಕೋಜೆಬ್ ಸಿಂಪಡಿಸಿ.",
        "ml": "നിങ്ങളുടെ ഉരുളക്കിഴങ്ങിന് ആദ്യകാല കരിച്ചിൽ രോഗം ഉണ്ട്. മാൻകോസെബ് തളിക്കുക.",
    },
    "healthy": {
        "hi": "बधाई हो! आपकी फसल स्वस्थ है। कोई उपचार की जरूरत नहीं।",
        "gu": "અભિનંદન! તમારી ફસલ સ્વસ્થ છે. કોઈ સારવારની જરૂર નથી.",
        "mr": "अभिनंदन! तुमची पीक निरोगी आहे. कोणत्याही उपचाराची गरज नाही.",
        "pa": "ਵਧਾਈ ਹੋ! ਤੁਹਾਡੀ ਫਸਲ ਸਿਹਤਮੰਦ ਹੈ। ਕਿਸੇ ਇਲਾਜ ਦੀ ਲੋੜ ਨਹੀਂ।",
        "ta": "வாழ்த்துகள்! உங்கள் பயிர் ஆரோக்கியமாக உள்ளது. சிகிச்சை தேவையில்லை.",
        "te": "అభినందనలు! మీ పంట ఆరోగ్యంగా ఉంది. చికిత్స అవసరం లేదు.",
        "bn": "অভিনন্দন! আপনার ফসল সুস্থ আছে। কোনো চিকিৎসার প্রয়োজন নেই।",
        "kn": "ಅಭಿನಂದನೆಗಳು! ನಿಮ್ಮ ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.",
        "ml": "അഭിനന്ദനങ്ങൾ! നിങ്ങളുടെ വിള ആരോഗ്യകരമാണ്. ചികിത്സ ആവശ്യമില്ല.",
    },
}


# ── Core TTS function ─────────────────────────────────────────────────────────
def text_to_speech(
    text:      str,
    lang:      str  = "hi",
    output_path: str = "output/voice.mp3",
    slow:      bool = False,
) -> str:
    """
    Convert text to speech in specified Indian language.
    Returns path to generated MP3 file.

    Args:
        text        : Text to speak
        lang        : Language code (hi/gu/mr/ta/te/bn/pa/kn/ml/or/as)
        output_path : Where to save the MP3
        slow        : Speak slowly (good for elderly farmers)

    Returns:
        Path to MP3 file
    """
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(output_path)
        print(f"✅ Voice generated → {output_path} (lang={lang})")
        return output_path
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None


def speak_disease_result(
    disease_class: str,
    lang:          str = "hi",
    output_dir:    str = "output/voice",
) -> str:
    """
    Generate voice message for a disease detection result.
    Auto-selects message in farmer's language.
    Returns path to MP3 file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find message
    is_healthy = "healthy" in disease_class.lower()
    key        = "healthy" if is_healthy else disease_class

    messages   = DISEASE_MESSAGES.get(key, {})
    text       = messages.get(lang, messages.get("hi", ""))

    if not text:
        # Fallback — generic message in Hindi
        text = "कृपया अपने नजदीकी कृषि केंद्र से संपर्क करें।"
        lang = "hi"

    output_path = f"{output_dir}/{disease_class}_{lang}.mp3"
    return text_to_speech(text=text, lang=lang, output_path=output_path)


def get_language_for_state(state: str) -> str:
    """Get language code for an Indian state."""
    return STATE_LANGUAGE_MAP.get(state, "hi")


def speak_price_update(
    commodity:   str,
    market:      str,
    price:       float,
    lang:        str = "hi",
    output_path: str = "output/voice/price.mp3",
) -> str:
    """Generate voice message for mandi price update."""

    messages = {
        "hi": f"{commodity} का आज का भाव {market} मंडी में {price} रुपये प्रति क्विंटल है।",
        "gu": f"{commodity} નો આજનો ભાવ {market} માં {price} રૂપિયા પ્રતિ ક્વિન્ટલ છે.",
        "mr": f"{commodity} चा आजचा भाव {market} मंडीत {price} रुपये प्रति क्विंटल आहे.",
        "pa": f"{commodity} ਦਾ ਅੱਜ ਦਾ ਭਾਅ {market} ਮੰਡੀ ਵਿੱਚ {price} ਰੁਪਏ ਪ੍ਰਤੀ ਕੁਇੰਟਲ ਹੈ।",
        "ta": f"{commodity} இன் இன்றைய விலை {market} சந்தையில் {price} ரூபாய் ஒரு குவிண்டாலுக்கு.",
        "te": f"{commodity} నేటి ధర {market} మార్కెట్‌లో {price} రూపాయలు క్వింటాలుకు.",
        "bn": f"{commodity} এর আজকের দাম {market} বাজারে {price} টাকা প্রতি কুইন্টাল।",
    }

    text = messages.get(lang, messages["hi"])
    return text_to_speech(text=text, lang=lang, output_path=output_path)


# ── Run directly to test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  अन्नदाता AI — Phase 3: Voice AI")
    print("="*55 + "\n")

    # Test 1: Disease result in Hindi
    print("── Test 1: Tomato Early Blight in Hindi ──")
    path = speak_disease_result("Tomato_Early_blight", lang="hi")
    print(f"  File: {path}\n")

    # Test 2: Same disease in Gujarati
    print("── Test 2: Tomato Early Blight in Gujarati ──")
    path = speak_disease_result("Tomato_Early_blight", lang="gu")
    print(f"  File: {path}\n")

    # Test 3: Healthy crop in Punjabi
    print("── Test 3: Healthy crop in Punjabi ──")
    path = speak_disease_result("Tomato_healthy", lang="pa")
    print(f"  File: {path}\n")

    # Test 4: Mandi price in Hindi
    print("── Test 4: Mandi price update in Hindi ──")
    path = speak_price_update(
        commodity   = "टमाटर",
        market      = "अहमदाबाद",
        price       = 1200,
        lang        = "hi",
        output_path = "output/voice/price_hi.mp3"
    )
    print(f"  File: {path}\n")

    # Test 5: Mandi price in Gujarati
    print("── Test 5: Mandi price update in Gujarati ──")
    path = speak_price_update(
        commodity   = "ટામેટા",
        market      = "અમદાવાદ",
        price       = 1200,
        lang        = "gu",
        output_path = "output/voice/price_gu.mp3"
    )
    print(f"  File: {path}\n")

    print("🎉 Voice AI complete!")
    print("\n📁 Check output/voice/ folder for MP3 files!")
    print("   Play them to hear अन्नदाता AI speaking to farmers!")