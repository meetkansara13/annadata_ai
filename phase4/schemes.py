"""
अन्नदाता AI — Phase 4
Government Scheme Matcher

Matches farmer profile to eligible government schemes.
Uses data already seeded in PostgreSQL.

Run:
    python phase4/schemes.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from database import SessionLocal, Scheme, Farmer

# ── Scheme eligibility rules ──────────────────────────────────────────────────
# Each rule is a function that takes farmer dict and returns True/False
ELIGIBILITY_RULES = {
    "PM-KISAN": lambda f: True,  # All farmers eligible
    "PMFBY - Pradhan Mantri Fasal Bima Yojana": lambda f: True,
    "KCC - Kisan Credit Card": lambda f: True,
    "Soil Health Card Scheme": lambda f: True,
    "eNAM - National Agriculture Market": lambda f: True,
    "Mukhyamantri Kisan Sahay Yojana": lambda f: f.get("state") == "Gujarat",
}

# ── Scheme messages in Indian languages ───────────────────────────────────────
SCHEME_MESSAGES = {
    "PM-KISAN": {
        "hi": "पीएम किसान योजना: आपको सालाना ₹6000 मिलेंगे। अभी pmkisan.gov.in पर आवेदन करें।",
        "gu": "પીએમ કિસાન યોજના: તમને વાર્ષિક ₹6000 મળશે. અત્યારે pmkisan.gov.in પર અરજી કરો.",
        "mr": "पीएम किसान योजना: तुम्हाला वार्षिक ₹6000 मिळतील. आत्ता pmkisan.gov.in वर अर्ज करा.",
        "pa": "ਪੀਐਮ ਕਿਸਾਨ ਯੋਜਨਾ: ਤੁਹਾਨੂੰ ਸਾਲਾਨਾ ₹6000 ਮਿਲਣਗੇ। ਹੁਣੇ pmkisan.gov.in 'ਤੇ ਅਰਜ਼ੀ ਦਿਓ।",
        "ta": "பிஎம் கிசான் திட்டம்: உங்களுக்கு ஆண்டுக்கு ₹6000 கிடைக்கும். இப்போதே pmkisan.gov.in இல் விண்ணப்பிக்கவும்.",
        "te": "పిఎం కిసాన్ పథకం: మీకు సంవత్సరానికి ₹6000 వస్తాయి. ఇప్పుడే pmkisan.gov.in లో దరఖాస్తు చేయండి.",
        "bn": "পিএম কিসান যোজনা: আপনি বার্ষিক ₹6000 পাবেন। এখনই pmkisan.gov.in এ আবেদন করুন।",
    },
    "PMFBY - Pradhan Mantri Fasal Bima Yojana": {
        "hi": "फसल बीमा योजना: प्राकृतिक आपदा से फसल नुकसान पर बीमा मिलेगा। pmfby.gov.in पर आवेदन करें।",
        "gu": "પાક વીમા યોજना: કુदरती આपदाथी पाक नुकसान पर विमो मळशे. pmfby.gov.in पर अरजी करो.",
        "mr": "पीक विमा योजना: नैसर्गिक आपत्तीमुळे पीक नुकसानावर विमा मिळेल. pmfby.gov.in वर अर्ज करा.",
        "pa": "ਫਸਲ ਬੀਮਾ ਯੋਜਨਾ: ਕੁਦਰਤੀ ਆਫ਼ਤ ਨਾਲ ਫਸਲ ਨੁਕਸਾਨ 'ਤੇ ਬੀਮਾ ਮਿਲੇਗਾ। pmfby.gov.in 'ਤੇ ਅਰਜ਼ੀ ਦਿਓ।",
        "ta": "பயிர் காப்பீட்டு திட்டம்: இயற்கை பேரிடரால் பயிர் இழப்புக்கு காப்பீடு கிடைக்கும். pmfby.gov.in இல் விண்ணப்பிக்கவும்.",
        "te": "పంట బీమా పథకం: ప్రకృతి వైపరీత్యాల వల్ల పంట నష్టానికి బీమా వస్తుంది. pmfby.gov.in లో దరఖాస్తు చేయండి.",
        "bn": "ফসল বীমা যোজনা: প্রাকৃতিক দুর্যোগে ফসল ক্ষতিতে বীমা পাবেন। pmfby.gov.in এ আবেদন করুন।",
    },
    "Mukhyamantri Kisan Sahay Yojana": {
        "gu": "મુખ્યમંત્રી કિસાન સહાય યોજना: ૩૩% થી વધુ પাક નુकसान पर ₹20,000 प्रति हेक्टर मળशे.",
        "hi": "मुख्यमंत्री किसान सहाय योजना (गुजरात): 33% से अधिक फसल नुकसान पर ₹20,000 प्रति हेक्टर मिलेगा।",
    },
}


# ── Core matcher function ─────────────────────────────────────────────────────
def match_schemes(farmer: dict) -> list[dict]:
    """
    Match farmer profile to eligible government schemes.

    Args:
        farmer: dict with keys — state, land_acres, has_bank_account

    Returns:
        List of matched schemes with eligibility reason
    """
    db      = SessionLocal()
    state   = farmer.get("state", "")
    matched = []

    try:
        # Get all schemes for this state
        schemes = db.query(Scheme).filter(
            (Scheme.state == "ALL") | (Scheme.state == state)
        ).all()

        for scheme in schemes:
            # Check eligibility rule
            rule      = ELIGIBILITY_RULES.get(scheme.name, lambda f: True)
            eligible  = rule(farmer)

            if eligible:
                lang    = farmer.get("language", "hi")
                message = SCHEME_MESSAGES.get(scheme.name, {}).get(lang, "")

                matched.append({
                    "id":          scheme.id,
                    "name":        scheme.name,
                    "description": scheme.description,
                    "benefit":     scheme.benefit,
                    "eligibility": scheme.eligibility,
                    "apply_url":   scheme.apply_url,
                    "message":     message,
                    "state":       scheme.state,
                })

        print(f"[अन्नदाता AI] ✅ Found {len(matched)} schemes for {state} farmer")
        return matched

    except Exception as e:
        print(f"❌ Scheme matching error: {e}")
        return []
    finally:
        db.close()


def get_scheme_voice(scheme_name: str, lang: str = "hi") -> str:
    """Get voice message for a scheme in farmer's language."""
    messages = SCHEME_MESSAGES.get(scheme_name, {})
    return messages.get(lang, messages.get("hi", ""))


def print_schemes(schemes: list[dict], lang: str = "hi"):
    """Pretty print matched schemes."""
    if not schemes:
        print("  No schemes found.")
        return
    for i, s in enumerate(schemes, 1):
        print(f"\n  {i}. {s['name']}")
        print(f"     Benefit   : {s['benefit']}")
        print(f"     Apply at  : {s['apply_url']}")
        if s.get("message"):
            print(f"     Message   : {s['message']}")


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  अन्नदाता AI — Phase 4: Scheme Matcher")
    print("="*55 + "\n")

    # Test 1: Gujarat farmer (Gujarati)
    print("── Test 1: Gujarat farmer ──")
    farmer_gujarat = {
        "name":     "Rameshbhai Patel",
        "state":    "Gujarat",
        "district": "Anand",
        "language": "gu",
    }
    schemes = match_schemes(farmer_gujarat)
    print_schemes(schemes, lang="gu")

    # Test 2: Punjab farmer (Punjabi)
    print("\n── Test 2: Punjab farmer ──")
    farmer_punjab = {
        "name":     "Gurpreet Singh",
        "state":    "Punjab",
        "district": "Ludhiana",
        "language": "pa",
    }
    schemes = match_schemes(farmer_punjab)
    print_schemes(schemes, lang="pa")

    # Test 3: UP farmer (Hindi)
    print("\n── Test 3: Uttar Pradesh farmer ──")
    farmer_up = {
        "name":     "Ramesh Kumar",
        "state":    "Uttar Pradesh",
        "district": "Agra",
        "language": "hi",
    }
    schemes = match_schemes(farmer_up)
    print_schemes(schemes, lang="hi")

    print("\n\n🎉 Phase 4 complete!")