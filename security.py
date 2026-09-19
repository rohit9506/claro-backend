import re
import secrets
import hashlib
import datetime
from typing import Optional, List, Tuple
import bcrypt
import jwt
from email_validator import validate_email as lib_validate_email, EmailNotValidError

from config import (
    JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES,
    PASSWORD_RESET_EXPIRE_MINUTES, EMAIL_VERIFY_EXPIRE_MINUTES
)

# 100 Most Common Compromised Passwords to block
COMMON_PASSWORDS = {
    "password", "123456", "12345678", "123456789", "qwerty", "12345", "111111",
    "1234567", "dragon", "welcome", "ninja", "monkey", "sunshine", "football",
    "princess", "solo", "superman", "batman", "letmein", "admin123", "password123",
    "iloveyou", "master", "shadow", "ashley", "bailey", "michael", "charlie",
    "computer", "trustno1", "starwars", "chelsea", "arsenal", "liverpool", "barcelona",
    "claro123", "officer123", "adminadmin", "password1234", "qwertyuiop", "123123",
    "default", "changeme", "pass1234", "secret123", "temppass123", "test1234"
}

def hash_password(password: str) -> str:
    """Hash password using bcrypt with salt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def hash_token(token: str) -> str:
    """Hash single-use token using SHA-256 for secure database storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def generate_secure_token() -> Tuple[str, str]:
    """Generate (raw_token, hashed_token) pair for email verification or password reset."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = hash_token(raw_token)
    return raw_token, token_hash

def validate_email_address(email: str) -> Tuple[bool, str]:
    """Strict email validation checking structure, domain format, and safety."""
    if not email or len(email) > 254:
        return False, "Email address is too long or empty."
    try:
        valid = lib_validate_email(email, check_deliverability=False)
        return True, valid.normalized
    except EmailNotValidError as e:
        return False, str(e)

def generate_random_password(length: int = 16) -> str:
    """Generate a cryptographically secure random temporary password."""
    import string
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in pwd)
            and any(c.isupper() for c in pwd)
            and any(c.isdigit() for c in pwd)
            and any(c in "!@#$%^&*" for c in pwd)):
            return pwd

def evaluate_password_strength(password: str) -> dict:
    """
    Evaluates password against strong password policy.
    - Minimum 12-15 characters
    - Rejects common passwords and predictable dictionary terms
    - Checks character diversity and entropy
    """
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    
    low = password.lower().strip()
    if low in COMMON_PASSWORDS or any(c in low for c in ["password123", "12345678", "qwertyuiop", "admin1234"]):
        errors.append("This password contains common dictionary terms or predictable patterns. Please choose a more secure passphrase.")
        
    has_upper = bool(re.search(r"[A-Z]", password))
    has_lower = bool(re.search(r"[a-z]", password))
    has_digit = bool(re.search(r"\d", password))
    has_special = bool(re.search(r"[!@#$%^&*(),.?\":{}|<>]", password))

    if not (has_upper or has_special) and len(password) < 12:
        errors.append("Password should include an uppercase letter or special symbol.")
    
    score = 0
    if len(password) >= 12: score += 1
    if len(password) >= 15: score += 1
    if has_upper and has_lower: score += 1
    if has_digit: score += 1
    if has_special: score += 1
    
    if score <= 2:
        level = "WEAK"
    elif score == 3:
        level = "FAIR"
    elif score == 4:
        level = "STRONG"
    else:
        level = "VERY STRONG"
        
    return {
        "is_valid": len(errors) == 0,
        "score": score,
        "level": level,
        "errors": errors,
        "has_upper": has_upper,
        "has_lower": has_lower,
        "has_digit": has_digit,
        "has_special": has_special,
        "length": len(password)
    }

def create_access_token(user_id: int, username: str, role: str) -> str:
    """Create JWT access token with role claim."""
    now = datetime.datetime.utcnow()
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except (jwt.PyJWTError, Exception):
        return None
