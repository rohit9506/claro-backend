import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(120), nullable=False)
    username = Column(String(60), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(30), nullable=False, default="ROLE_USER")  # ROLE_ADMIN, ROLE_OFFICER, ROLE_USER
    
    # Officer-specific fields
    officer_id = Column(String(60), unique=True, nullable=True)
    department = Column(String(120), nullable=True)
    mobile_number = Column(String(20), nullable=True)
    
    # Account status & Security
    is_active = Column(Boolean, default=True)
    is_temporary_password = Column(Boolean, default=False)
    
    # Email verification
    email_verified = Column(Boolean, default=False)
    email_verified_at = Column(DateTime, nullable=True)
    verification_token_hash = Column(String(128), nullable=True)
    verification_token_expires_at = Column(DateTime, nullable=True)
    
    # Password reset
    password_reset_token_hash = Column(String(128), nullable=True)
    password_reset_token_expires_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    # Relationships
    inspections = relationship("Inspection", back_populates="officer")
    reviews = relationship("ProductReview", back_populates="user")
    saved_products = relationship("UserSavedProduct", back_populates="user")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), index=True, nullable=False)
    brand = Column(String(120), nullable=True)
    category = Column(String(100), index=True, nullable=True)  # Food, FMCG, Personal Care, Electronics, etc.
    barcode = Column(String(50), index=True, nullable=True)
    
    # Mandatory declarations
    mrp = Column(String(50), nullable=True)
    net_quantity = Column(String(50), nullable=True)
    unit_sale_price = Column(String(50), nullable=True)
    mfg_date = Column(String(50), nullable=True)
    expiry_date = Column(String(50), nullable=True)
    country_of_origin = Column(String(80), nullable=True)
    
    # Entities
    manufacturer = Column(String(255), nullable=True)
    packer = Column(String(255), nullable=True)
    importer = Column(String(255), nullable=True)
    complete_address = Column(Text, nullable=True)
    
    # Consumer care
    consumer_care_email = Column(String(150), nullable=True)
    consumer_care_phone = Column(String(50), nullable=True)
    consumer_care_address = Column(Text, nullable=True)
    
    # Consumer product info (Ingredients & Nutrition separated from compliance)
    ingredients = Column(Text, nullable=True)
    nutrition_facts = Column(Text, nullable=True)  # JSON string: calories, protein_g, carbs_g, sugar_g, fat_g, sodium_mg
    allergens = Column(String(255), nullable=True)
    
    image_url = Column(String(255), nullable=True)
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    inspections = relationship("Inspection", back_populates="product")
    reviews = relationship("ProductReview", back_populates="product")


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    inspection_number = Column(String(50), unique=True, index=True, nullable=False)  # INSP-2026-0001
    officer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    product_name = Column(String(200), nullable=False)
    brand = Column(String(120), nullable=True)
    variant = Column(String(120), nullable=True)
    mrp = Column(String(50), nullable=True)
    net_quantity = Column(String(50), nullable=True)
    source = Column(String(20), default="CAMERA")  # CAMERA or UPLOAD
    
    status = Column(String(30), default="PENDING_VERIFICATION", index=True)  # COMPLIANT, NON_COMPLIANT, PENDING_VERIFICATION
    pass_count = Column(Integer, default=0)
    fail_count = Column(Integer, default=0)
    review_count = Column(Integer, default=0)
    
    # Uploaded/Captured Image paths
    front_image = Column(String(255), nullable=True)
    back_image = Column(String(255), nullable=True)
    side_image = Column(String(255), nullable=True)
    right_image = Column(String(255), nullable=True)
    left_image = Column(String(255), nullable=True)
    
    location = Column(String(255), nullable=True)
    officer_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    finalized_at = Column(DateTime, nullable=True)

    # Relationships
    officer = relationship("User", back_populates="inspections")
    product = relationship("Product", back_populates="inspections")
    validations = relationship("FieldValidation", back_populates="inspection", cascade="all, delete-orphan")
    verifications = relationship("Verification", back_populates="inspection", cascade="all, delete-orphan")
    ocr_results = relationship("OCRResult", back_populates="inspection", cascade="all, delete-orphan")


class InspectionImage(Base):
    __tablename__ = "inspection_images"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False)
    side = Column(String(20), nullable=False)  # front, back, right_side, left_side
    view_type = Column(String(20), nullable=True)
    image_path = Column(String(255), nullable=False)
    source = Column(String(20), default="CAMERA")  # CAMERA or UPLOAD
    mime_type = Column(String(50), default="image/jpeg")
    file_size = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    processing_status = Column(String(30), default="PROCESSED")
    quality_score = Column(Float, default=1.0)
    is_accepted = Column(Boolean, default=True)
    quality_notes = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class FieldValidation(Base):
    __tablename__ = "field_validations"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False)
    
    field_name = Column(String(80), nullable=False)  # mrp, net_quantity, manufacturer, dates, consumer_care, etc.
    detected_value = Column(Text, nullable=True)
    requirement_summary = Column(Text, nullable=False)
    rule_reference = Column(String(100), nullable=False)  # e.g., Rule 6(1)(a), Rule 6(1)(c)
    
    status = Column(String(20), nullable=False)  # PASS, FAIL, REVIEW
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    reason = Column(Text, nullable=False)
    
    # Visual Evidence
    image_side = Column(String(20), nullable=True)  # front, back, side
    bbox_json = Column(Text, nullable=True)  # JSON: [ymin, xmin, ymax, xmax] normalized (0.0 to 1.0)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    inspection = relationship("Inspection", back_populates="validations")


class OCRResult(Base):
    __tablename__ = "ocr_results"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False)
    image_side = Column(String(20), nullable=False)
    raw_text = Column(Text, nullable=False)
    boxes_json = Column(Text, nullable=True)  # Full raw bounding boxes & confidences
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    inspection = relationship("Inspection", back_populates="ocr_results")


class Rule(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, index=True)
    rule_code = Column(String(50), unique=True, index=True, nullable=False)  # LMR-2011-R6-1-A
    rule_name = Column(String(150), nullable=False)
    declaration_checked = Column(String(100), nullable=False)
    requirement = Column(Text, nullable=False)
    applicable_category = Column(String(100), default="All Packaged Commodities")
    validation_criteria = Column(Text, nullable=False)
    legal_source = Column(String(150), default="Legal Metrology (Packaged Commodities) Rules, 2011")
    version = Column(String(20), default="2011.Amended2021")
    effective_date = Column(String(30), default="2011-04-01")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class Verification(Base):
    __tablename__ = "verifications"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False)
    field_validation_id = Column(Integer, ForeignKey("field_validations.id"), nullable=True)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reviewer_name = Column(String(120), nullable=False)
    
    decision = Column(String(30), nullable=False)  # CONFIRM_COMPLIANT, MARK_NON_COMPLIANT, REQUEST_RESCAN
    previous_status = Column(String(20), nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    inspection = relationship("Inspection", back_populates="verifications")


class ProductReview(Base):
    __tablename__ = "product_reviews"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user_name = Column(String(120), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    
    overall_rating = Column(Float, nullable=False)  # 1.0 to 5.0
    taste_rating = Column(Float, nullable=True)
    vfm_rating = Column(Float, nullable=True)  # Value for money
    would_buy_again = Column(Boolean, default=True)
    review_text = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="reviews")
    product = relationship("Product", back_populates="reviews")


class UserSavedProduct(Base):
    __tablename__ = "user_saved_products"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="saved_products")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor_id = Column(Integer, nullable=True)
    actor_name = Column(String(120), nullable=False)
    actor_role = Column(String(30), nullable=False)
    action = Column(String(80), nullable=False)  # USER_REGISTER, ADMIN_LOGIN, RULE_UPDATE, VERIFICATION_DECISION, etc.
    target_type = Column(String(60), nullable=True)  # USER, RULE, INSPECTION, OFFICER
    target_id = Column(String(60), nullable=True)
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    recipient_role = Column(String(30), nullable=True)  # ROLE_ADMIN, ROLE_OFFICER, ROLE_USER
    user_id = Column(Integer, nullable=True)
    title = Column(String(150), nullable=False)
    message = Column(Text, nullable=False)
    notif_type = Column(String(40), default="INFO")  # INFO, WARNING, SUCCESS, ALERT
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
