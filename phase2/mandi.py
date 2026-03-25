"""
अन्नदाता AI — Phase 2
Mandi Price Fetcher + XGBoost Price Predictor

Run:
    python phase2/mandi.py
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from database import SessionLocal, MandiPrice

load_dotenv()
API_KEY  = os.getenv("DATAGOV_API_KEY")
BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"


def fetch_mandi_prices(state=None, commodity=None, limit=100):
    if not API_KEY:
        raise ValueError("DATAGOV_API_KEY not found in .env!")

    params = {"api-key": API_KEY, "format": "json", "limit": limit}
    if state:     params["filters[state]"]     = state
    if commodity: params["filters[commodity]"] = commodity

    print(f"[अन्नदाता AI] Fetching prices... state={state} commodity={commodity}")
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        records = r.json().get("records", [])
        print(f"✅ Fetched {len(records)} records")
        return records
    except Exception as e:
        print(f"❌ API error: {e}")
        return []


def save_prices_to_db(records):
    if not records: return 0
    db = SessionLocal()
    saved = 0
    try:
        for r in records:
            try:
                try:
                    price_date = datetime.strptime(r.get("arrival_date",""), "%d/%m/%Y").date()
                except:
                    price_date = date.today()

                exists = db.query(MandiPrice).filter(
                    MandiPrice.market     == r.get("market",""),
                    MandiPrice.commodity  == r.get("commodity",""),
                    MandiPrice.price_date == price_date,
                ).first()
                if exists: continue

                db.add(MandiPrice(
                    state       = r.get("state",""),
                    district    = r.get("district",""),
                    market      = r.get("market",""),
                    commodity   = r.get("commodity",""),
                    min_price   = float(r.get("min_price",0) or 0),
                    max_price   = float(r.get("max_price",0) or 0),
                    modal_price = float(r.get("modal_price",0) or 0),
                    price_date  = price_date,
                ))
                saved += 1
            except Exception as e:
                continue
        db.commit()
        print(f"✅ Saved {saved} new records to database")
        return saved
    except Exception as e:
        db.rollback()
        print(f"❌ DB error: {e}")
        return 0
    finally:
        db.close()


def predict_prices(commodity, market, days=7):
    try:
        from xgboost import XGBRegressor
    except:
        print("❌ Run: pip install xgboost")
        return []

    db = SessionLocal()
    try:
        records = db.query(MandiPrice).filter(
            MandiPrice.commodity.ilike(f"%{commodity}%"),
            MandiPrice.market.ilike(f"%{market}%"),
        ).order_by(MandiPrice.price_date).all()

        if len(records) < 10:
            print(f"⚠️  Need at least 10 records. Found {len(records)}.")
            return []

        df = pd.DataFrame([{
            "date": r.price_date, "modal_price": r.modal_price,
        } for r in records]).sort_values("date").reset_index(drop=True)

        df["dow"]         = pd.to_datetime(df["date"]).dt.dayofweek
        df["dom"]         = pd.to_datetime(df["date"]).dt.day
        df["month"]       = pd.to_datetime(df["date"]).dt.month
        df["lag1"]        = df["modal_price"].shift(1)
        df["lag7"]        = df["modal_price"].shift(7)
        df["roll7"]       = df["modal_price"].rolling(7).mean()
        df = df.dropna()

        feats = ["dow","dom","month","lag1","lag7","roll7"]
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
        model.fit(df[feats], df["modal_price"])

        last_prices = list(df["modal_price"].values[-7:])
        last_date   = pd.to_datetime(df["date"].values[-1])
        predictions = []

        for i in range(1, days+1):
            d = last_date + timedelta(days=i)
            row = pd.DataFrame([{
                "dow": d.dayofweek, "dom": d.day, "month": d.month,
                "lag1": last_prices[-1],
                "lag7": last_prices[-7] if len(last_prices)>=7 else last_prices[0],
                "roll7": np.mean(last_prices[-7:]),
            }])
            price = round(float(model.predict(row)[0]), 2)
            last_prices.append(price)
            predictions.append({
                "date": d.strftime("%d %b %Y"),
                "predicted_price": price,
                "commodity": commodity,
                "market": market,
            })

        print(f"✅ {days}-day price forecast ready for {commodity} at {market}")
        return predictions
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return []
    finally:
        db.close()


def get_latest_prices(state=None, commodity=None, limit=20):
    db = SessionLocal()
    try:
        q = db.query(MandiPrice)
        if state:     q = q.filter(MandiPrice.state.ilike(f"%{state}%"))
        if commodity: q = q.filter(MandiPrice.commodity.ilike(f"%{commodity}%"))
        prices = q.order_by(MandiPrice.price_date.desc()).limit(limit).all()
        return [{"state":p.state,"district":p.district,"market":p.market,
                 "commodity":p.commodity,"min_price":p.min_price,
                 "max_price":p.max_price,"modal_price":p.modal_price,
                 "date":str(p.price_date)} for p in prices]
    finally:
        db.close()


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  अन्नदाता AI — Phase 2: Mandi Price Fetcher")
    print("="*55 + "\n")

    print("── Step 1: Fetching live prices (Gujarat) ──")
    records = fetch_mandi_prices(state="Gujarat", limit=50)

    print("\n── Step 2: Saving to PostgreSQL ──")
    save_prices_to_db(records)

    print("\n── Step 3: Latest prices from DB ──")
    prices = get_latest_prices(state="Gujarat", limit=5)
    if prices:
        for p in prices:
            print(f"  {p['commodity']:20} | {p['market']:20} | ₹{p['modal_price']}")
    else:
        print("  No prices in DB yet.")

    print("\n── Step 4: 7-day price prediction ──")
    if prices:
        preds = predict_prices(prices[0]["commodity"], prices[0]["market"], days=7)
        if preds:
            print(f"\n  📈 Forecast:")
            for p in preds:
                print(f"  {p['date']:15} → ₹{p['predicted_price']}")

    print("\n🎉 Phase 2 complete!")