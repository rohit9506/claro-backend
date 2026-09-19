import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration (SQLite by default, PostgreSQL ready via DATABASE_URL)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'claro.db'}")

# Security & JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "claro-sih26034-legal-metrology-ultra-secure-key-2026")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
PASSWORD_RESET_EXPIRE_MINUTES = 30
EMAIL_VERIFY_EXPIRE_MINUTES = 1440  # 24 hours

# Rule Engine Configuration
CONFIDENCE_REVIEW_THRESHOLD = 0.75  # Below this score, field routed to PENDING_VERIFICATION

# Setup flag file
SETUP_FLAG_FILE = BASE_DIR / ".setup_credentials.json"

# CORS allowed origins
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "capacitor://localhost",
    "http://localhost",
    "*"
]
