"""
अन्नदाता AI — Database Layer
SQLAlchemy models + connection + helper functions
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, Boolean, Text, Date, DateTime,
    ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Engine + Session ──────────────────────────────────────────────────────────
engine       = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base         = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class Farmer(Base):
    __tablename__ = "farmers"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(100))
    phone      = Column(String(15), unique=True)
    state      = Column(String(50))
    district   = Column(String(50))
    language   = Column(String(10), default="hi")
    created_at = Column(DateTime, default=datetime.now)

    predictions = relationship("Prediction", back_populates="farmer")
    alerts      = relationship("PriceAlert",  back_populates="farmer")

    def __repr__(self):
        return f"<Farmer {self.name} | {self.state} | {self.language}>"


class Prediction(Base):
    __tablename__ = "predictions"

    id            = Column(Integer, primary_key=True, index=True)
    farmer_id     = Column(Integer, ForeignKey("farmers.id"), nullable=True)
    disease_class = Column(String(100))
    disease_hindi = Column(String(200))
    confidence    = Column(Float)
    treatment     = Column(Text)
    image_path    = Column(String(255))
    created_at    = Column(DateTime, default=datetime.now)

    farmer = relationship("Farmer", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction {self.disease_class} | {self.confidence:.1f}%>"


class MandiPrice(Base):
    __tablename__ = "mandi_prices"

    id          = Column(Integer, primary_key=True, index=True)
    state       = Column(String(50))
    district    = Column(String(50))
    market      = Column(String(100))
    commodity   = Column(String(100))
    min_price   = Column(Float)
    max_price   = Column(Float)
    modal_price = Column(Float)
    price_date  = Column(Date)
    created_at  = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<MandiPrice {self.commodity} @ {self.market} = ₹{self.modal_price}>"


class Scheme(Base):
    __tablename__ = "schemes"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(200))
    description = Column(Text)
    eligibility = Column(Text)
    benefit     = Column(Text)
    apply_url   = Column(String(255))
    state       = Column(String(50), default="ALL")

    def __repr__(self):
        return f"<Scheme {self.name}>"


class PriceAlert(Base):
    __tablename__ = "price_alerts"

    id           = Column(Integer, primary_key=True, index=True)
    farmer_id    = Column(Integer, ForeignKey("farmers.id"))
    commodity    = Column(String(100))
    target_price = Column(Float)
    market       = Column(String(100))
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=datetime.now)

    farmer = relationship("Farmer", back_populates="alerts")

    def __repr__(self):
        return f"<PriceAlert {self.commodity} @ ₹{self.target_price}>"


# ── DB Helper functions ───────────────────────────────────────────────────────

def get_db():
    """Get database session — use as context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    """Test if database connection works."""
    try:
        with engine.connect() as conn:
            print("✅ PostgreSQL connected successfully!")
            print(f"   Host     : {DB_HOST}:{DB_PORT}")
            print(f"   Database : {DB_NAME}")
            print(f"   User     : {DB_USER}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def seed_schemes():
    """Insert default government schemes into database."""
    db = SessionLocal()
    try:
        # Check if already seeded
        if db.query(Scheme).count() > 0:
            print("Schemes already seeded.")
            return

        schemes = [
            Scheme(
                name        = "PM-KISAN",
                description = "Direct income support of ₹6000/year to farmer families",
                eligibility = "Small and marginal farmers with cultivable land",
                benefit     = "₹6000 per year in 3 installments of ₹2000",
                apply_url   = "https://pmkisan.gov.in",
                state       = "ALL"
            ),
            Scheme(
                name        = "PMFBY - Pradhan Mantri Fasal Bima Yojana",
                description = "Crop insurance scheme for farmers against natural calamities",
                eligibility = "All farmers growing notified crops",
                benefit     = "Insurance coverage for crop loss due to flood, drought, pest",
                apply_url   = "https://pmfby.gov.in",
                state       = "ALL"
            ),
            Scheme(
                name        = "KCC - Kisan Credit Card",
                description = "Short term credit for farmers at low interest rate",
                eligibility = "All farmers, sharecroppers, tenant farmers",
                benefit     = "Credit up to ₹3 lakh at 4% interest rate",
                apply_url   = "https://www.nabard.org",
                state       = "ALL"
            ),
            Scheme(
                name        = "Soil Health Card Scheme",
                description = "Free soil testing and nutrient recommendations",
                eligibility = "All farmers",
                benefit     = "Free soil health card with fertilizer recommendations",
                apply_url   = "https://soilhealth.dac.gov.in",
                state       = "ALL"
            ),
            Scheme(
                name        = "eNAM - National Agriculture Market",
                description = "Online trading platform for agricultural commodities",
                eligibility = "All farmers with produce to sell",
                benefit     = "Better price discovery, reduced middlemen",
                apply_url   = "https://enam.gov.in",
                state       = "ALL"
            ),
            Scheme(
                name        = "Mukhyamantri Kisan Sahay Yojana",
                description = "Gujarat state crop loss compensation scheme",
                eligibility = "Gujarat farmers who suffered crop loss",
                benefit     = "₹20,000 per hectare for 33-60% loss, ₹25,000 for 60%+ loss",
                apply_url   = "https://agri.gujarat.gov.in/MMKSY.htm",
                state       = "Gujarat"
            ),
        ]

        db.add_all(schemes)
        db.commit()
        print(f"✅ Seeded {len(schemes)} government schemes!")

    except Exception as e:
        db.rollback()
        print(f"❌ Seeding failed: {e}")
    finally:
        db.close()


def log_prediction(disease_class, disease_hindi, confidence, treatment,
                   image_path=None, farmer_id=None):
    """Log a disease prediction to database."""
    db = SessionLocal()
    try:
        pred = Prediction(
            farmer_id     = farmer_id,
            disease_class = disease_class,
            disease_hindi = disease_hindi,
            confidence    = confidence,
            treatment     = treatment,
            image_path    = image_path,
        )
        db.add(pred)
        db.commit()
        db.refresh(pred)
        print(f"✅ Prediction logged (id={pred.id})")
        return pred.id
    except Exception as e:
        db.rollback()
        print(f"❌ Failed to log prediction: {e}")
        return None
    finally:
        db.close()


def get_farmer_history(farmer_id: int):
    """Get all predictions for a farmer."""
    db = SessionLocal()
    try:
        preds = db.query(Prediction)\
                  .filter(Prediction.farmer_id == farmer_id)\
                  .order_by(Prediction.created_at.desc())\
                  .all()
        return preds
    finally:
        db.close()


def get_schemes_for_state(state: str):
    """Get all schemes available for a state."""
    db = SessionLocal()
    try:
        schemes = db.query(Scheme)\
                    .filter((Scheme.state == "ALL") | (Scheme.state == state))\
                    .all()
        return schemes
    finally:
        db.close()


# ── Run directly to test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  अन्नदाता AI — Database Setup")
    print("="*50 + "\n")

    # Test connection
    if test_connection():
        # Seed schemes
        seed_schemes()

        # Test log a dummy prediction
        pred_id = log_prediction(
            disease_class = "Tomato_Early_blight",
            disease_hindi = "टमाटर का अगेती झुलसा",
            confidence    = 94.3,
            treatment     = "Spray Copper Oxychloride 50% WP @ 3g/L",
        )
        print(f"\n✅ Test prediction logged with id={pred_id}")

        # Test fetch schemes for Gujarat
        schemes = get_schemes_for_state("Gujarat")
        print(f"\n✅ Schemes for Gujarat: {len(schemes)} found")
        for s in schemes:
            print(f"   → {s.name}")

        print("\n🎉 Database fully working!")   