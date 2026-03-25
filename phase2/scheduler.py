"""
अन्नदाता AI — Daily Price Scheduler
Fetches mandi prices every day automatically and stores in PostgreSQL.
After 10+ days, XGBoost prediction will work.

Run once to start:
    python phase2/scheduler.py

Or add to Windows Task Scheduler to run daily at 8 AM.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase2.mandi import fetch_mandi_prices, save_prices_to_db

# States to fetch daily
STATES = [
    "Gujarat",
    "Maharashtra",
    "Punjab",
    "Uttar Pradesh",
    "Rajasthan",
    "Madhya Pradesh",
    "Karnataka",
    "Andhra Pradesh",
    "West Bengal",
]

def fetch_all_states(limit_per_state: int = 100):
    """Fetch prices for all states and save to DB."""
    print(f"\n[{datetime.now().strftime('%d %b %Y %H:%M')}] Starting daily price fetch...")
    total_saved = 0
    for state in STATES:
        records = fetch_mandi_prices(state=state, limit=limit_per_state)
        saved   = save_prices_to_db(records)
        total_saved += saved
    print(f"✅ Daily fetch complete — {total_saved} new records saved\n")
    return total_saved


def run_once():
    """Run fetch once right now."""
    fetch_all_states()


def run_daily(hour: int = 8):
    """
    Run fetch every day at specified hour.
    Keeps running — use Ctrl+C to stop.
    """
    print(f"[अन्नदाता AI] Daily scheduler started — fetching every day at {hour}:00 AM")
    print("Press Ctrl+C to stop.\n")

    while True:
        now = datetime.now()
        # Run if it's the right hour and minute
        if now.hour == hour and now.minute == 0:
            fetch_all_states()
            time.sleep(60)  # Wait 60s to avoid double-fetch
        else:
            # Show countdown
            next_run = now.replace(hour=hour, minute=0, second=0)
            if next_run < now:
                from datetime import timedelta
                next_run += timedelta(days=1)
            remaining = next_run - now
            hours_left = remaining.seconds // 3600
            mins_left  = (remaining.seconds % 3600) // 60
            print(f"\r⏰ Next fetch in {hours_left}h {mins_left}m ... ", end="", flush=True)
            time.sleep(30)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  अन्नदाता AI — Mandi Price Scheduler")
    print("="*55)
    print("\nOptions:")
    print("  1 → Fetch now (once)")
    print("  2 → Run daily scheduler (keeps running)")
    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        run_once()
        print("\n✅ Done! Run again tomorrow to build historical data.")
        print("   After 10 days → XGBoost prediction will work!")
    elif choice == "2":
        run_daily(hour=8)
    else:
        print("Invalid choice.")