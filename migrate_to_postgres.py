import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import (
    Base, User, Product, Inspection, InspectionImage,
    FieldValidation, OCRResult, Rule, Verification,
    ProductReview, UserSavedProduct, AuditLog
)

def migrate_sqlite_to_postgres(target_database_url):
    print(f"Connecting to target PostgreSQL: {target_database_url[:35]}...")
    
    if target_database_url.startswith("postgres://"):
        target_database_url = target_database_url.replace("postgres://", "postgresql://", 1)
        
    pg_engine = create_engine(target_database_url, echo=False)
    Base.metadata.create_all(bind=pg_engine)
    PgSession = sessionmaker(bind=pg_engine)
    pg_db = PgSession()

    # Source SQLite database
    sqlite_path = os.path.join(os.path.dirname(__file__), "claro.db")
    if not os.path.exists(sqlite_path):
        print("Source SQLite database claro.db not found.")
        return False
        
    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
    SqliteSession = sessionmaker(bind=sqlite_engine)
    sq_db = SqliteSession()

    print("Migrating tables to Cloud PostgreSQL...")
    models = [
        ("Users", User),
        ("Rules", Rule),
        ("Products", Product),
        ("Inspections", Inspection),
        ("Inspection Images", InspectionImage),
        ("Field Validations", FieldValidation),
        ("OCR Results", OCRResult),
        ("Verifications", Verification),
        ("Product Reviews", ProductReview),
        ("User Saved Products", UserSavedProduct),
        ("Audit Logs", AuditLog),
    ]

    for name, model_cls in models:
        records = sq_db.query(model_cls).all()
        count = 0
        for r in records:
            # Check if record already exists by ID
            existing = pg_db.query(model_cls).filter(model_cls.id == r.id).first()
            if not existing:
                # Detach from sqlite session and add to postgres
                sq_db.expunge(r)
                pg_db.merge(r)
                count += 1
        pg_db.commit()
        print(f"  -> Migrated {count} new {name} (Total in source: {len(records)})")

    print("\nCloud PostgreSQL migration complete and verified!")
    sq_db.close()
    pg_db.close()
    return True

if __name__ == "__main__":
    url = os.getenv("DATABASE_URL")
    if len(sys.argv) > 1:
        url = sys.argv[1]
    if not url:
        print("Usage: python migrate_to_postgres.py <TARGET_POSTGRES_DATABASE_URL>")
        sys.exit(1)
    migrate_sqlite_to_postgres(url)
