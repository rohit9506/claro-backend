import json
import uuid
import datetime
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from config import UPLOAD_DIR
from database import get_db
from models import User, Product, ProductReview, UserSavedProduct
from auth_deps import get_current_user, require_role, log_audit
from ocr_service import ocr_service
from product_identifier import identify_product_multi_signal
from field_extractor import extract_declarations_from_multi_side

router = APIRouter(prefix="/consumer", tags=["Consumer Product Intelligence"])

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

# Schemas
class ReviewRequest(BaseModel):
    product_id: int
    overall_rating: float
    taste_rating: Optional[float] = 4.0
    vfm_rating: Optional[float] = 4.0
    would_buy_again: Optional[bool] = True
    review_text: Optional[str] = None

class CompareRequest(BaseModel):
    product_id_1: int
    product_id_2: int

# 1. Product Catalog & Search
@router.get("/products")
def list_products(
    search: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Product)
    if search:
        s = f"%{search.strip()}%"
        query = query.filter(
            (Product.name.ilike(s)) | (Product.brand.ilike(s)) | (Product.barcode == search.strip())
        )
    if category:
        query = query.filter(Product.category == category)

    products = query.all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "brand": p.brand,
            "category": p.category,
            "barcode": p.barcode,
            "mrp": p.mrp,
            "net_quantity": p.net_quantity,
            "unit_sale_price": p.unit_sale_price,
            "country_of_origin": p.country_of_origin,
            "image_url": p.image_url
        }
        for p in products
    ]

# 2. Product Details & Nutrition Breakdown (Know Your Product)
@router.get("/products/{product_id}")
def get_product_details(product_id: int, db: Session = Depends(get_db)):
    p = db.query(Product).filter(Product.id == product_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Product not found.")

    # Calculate average community ratings
    reviews = db.query(ProductReview).filter(ProductReview.product_id == product_id).all()
    avg_overall = sum(r.overall_rating for r in reviews) / len(reviews) if reviews else 0.0
    avg_taste = sum(r.taste_rating for r in reviews if r.taste_rating) / len(reviews) if reviews else 0.0
    avg_vfm = sum(r.vfm_rating for r in reviews if r.vfm_rating) / len(reviews) if reviews else 0.0
    would_buy_pct = (sum(1 for r in reviews if r.would_buy_again) / len(reviews) * 100) if reviews else 100.0

    nutrition = json.loads(p.nutrition_facts) if p.nutrition_facts else {
        "serving_size": "100 g",
        "calories": 0, "protein_g": 0.0, "carbs_g": 0.0, "sugar_g": 0.0, "fat_g": 0.0, "sodium_mg": 0
    }

    return {
        "id": p.id,
        "name": p.name,
        "brand": p.brand,
        "category": p.category,
        "barcode": p.barcode,
        "declarations": {
            "mrp": p.mrp,
            "net_quantity": p.net_quantity,
            "unit_sale_price": p.unit_sale_price,
            "mfg_date": p.mfg_date,
            "expiry_date": p.expiry_date,
            "country_of_origin": p.country_of_origin,
            "manufacturer": p.manufacturer,
            "complete_address": p.complete_address,
            "consumer_care": {
                "email": p.consumer_care_email,
                "phone": p.consumer_care_phone,
                "address": p.consumer_care_address
            }
        },
        "ingredients": p.ingredients,
        "nutrition_facts": nutrition,
        "allergens": p.allergens,
        "community_ratings": {
            "review_count": len(reviews),
            "avg_overall": round(avg_overall, 1),
            "avg_taste": round(avg_taste, 1),
            "avg_vfm": round(avg_vfm, 1),
            "would_buy_again_pct": int(would_buy_pct),
            "disclaimer": "Community ratings are user-submitted opinions and do not represent Legal Metrology compliance findings."
        },
        "reviews": [
            {
                "user_name": r.user_name,
                "overall_rating": r.overall_rating,
                "taste_rating": r.taste_rating,
                "review_text": r.review_text,
                "created_at": r.created_at.strftime("%Y-%m-%d")
            }
            for r in reviews[-5:]
        ]
    }

# 3. Consumer Camera / Multi-Side Image Scan
@router.post("/scan")
async def consumer_scan(
    front_image: Optional[UploadFile] = File(None),
    back_image: Optional[UploadFile] = File(None),
    side_image: Optional[UploadFile] = File(None),
    right_image: Optional[UploadFile] = File(None),
    left_image: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),  # Backward compatibility
    db: Session = Depends(get_db)
):
    """
    Consumer scanning - supports 4-sided photo capture (Front, Back, Right Side, Left Side) or single capture.
    Performs multi-signal product identification (Barcode, Catalog visual match, OCR text match)
    and field extraction.
    """
    # Normalize inputs: if only `image` was provided, treat it as front_image
    if not front_image and image:
        front_image = image

    if not front_image:
        raise HTTPException(status_code=400, detail="Please provide at least the front image of the product.")

    session_id = uuid.uuid4().hex[:8]
    images_saved = {}
    ocr_side_detections = {}

    async def save_and_ocr(side_name: str, upload_file: UploadFile):
        if not upload_file:
            return
        content = await upload_file.read()
        if not content:
            return
        filename = f"consumer_{session_id}_{side_name}.jpg"
        filepath = UPLOAD_DIR / filename
        with open(filepath, "wb") as f:
            f.write(content)
        images_saved[side_name] = f"/uploads/{filename}"
        dets = ocr_service.extract_text_with_boxes(str(filepath))
        ocr_side_detections[side_name] = dets

    await save_and_ocr("front", front_image)
    if back_image:
        await save_and_ocr("back", back_image)
    if right_image:
        await save_and_ocr("right_side", right_image)
    elif side_image:
        await save_and_ocr("side", side_image)
    if left_image:
        await save_and_ocr("left_side", left_image)

    # Multi-Signal Product Identification
    prod_ident = identify_product_multi_signal(
        image_paths={k: str(UPLOAD_DIR / f"consumer_{session_id}_{k}.jpg") for k in images_saved.keys()},
        ocr_side_detections=ocr_side_detections,
        db=db
    )

    # Extract Declarations
    extracted_declarations = extract_declarations_from_multi_side(ocr_side_detections)

    # Matched product details
    matched_prod_dict = prod_ident.get("matched_product")
    matched_product_id = matched_prod_dict.get("id") if matched_prod_dict else None

    # Reviews and community stats if product is in catalog
    community_stats = None
    if matched_product_id:
        reviews = db.query(ProductReview).filter(ProductReview.product_id == matched_product_id).all()
        avg_overall = sum(r.overall_rating for r in reviews) / len(reviews) if reviews else 0.0
        community_stats = {
            "review_count": len(reviews),
            "avg_overall": round(avg_overall, 1)
        }

    # Canonical product name resolution
    if matched_prod_dict and matched_prod_dict.get("name"):
        resolved_name = matched_prod_dict["name"]
        verification_status = "Verified"
    elif extracted_declarations.get("product_name", {}).get("detected") and extracted_declarations["product_name"]["value"] not in ["Not detected", ""]:
        resolved_name = extracted_declarations["product_name"]["value"]
        verification_status = "Identified via Physical Package"
    else:
        resolved_name = "Product could not be confidently identified."
        verification_status = "Needs Verification"

    return {
        "success": True,
        "status": prod_ident["status"],
        "matched": bool(matched_prod_dict),
        "product_id": matched_product_id,
        "product_name": resolved_name,
        "brand": (matched_prod_dict.get("brand") if matched_prod_dict else None) or extracted_declarations.get("brand", {}).get("value") or "Not confidently detected",
        "verification_status": verification_status,
        "matched_product": matched_prod_dict,
        "candidates": prod_ident.get("candidates", []),
        "barcode_detected": prod_ident.get("barcode_detected"),
        "extracted_declarations": extracted_declarations,
        "community_stats": community_stats,
        "images": images_saved
    }

# 4. Product Comparison (Normalized per 100g)
@router.post("/compare")
def compare_products(req: CompareRequest, db: Session = Depends(get_db)):
    p1 = db.query(Product).filter(Product.id == req.product_id_1).first()
    p2 = db.query(Product).filter(Product.id == req.product_id_2).first()

    if not p1 or not p2:
        raise HTTPException(status_code=404, detail="One or both products could not be found.")

    n1 = json.loads(p1.nutrition_facts) if p1.nutrition_facts else {}
    n2 = json.loads(p2.nutrition_facts) if p2.nutrition_facts else {}

    return {
        "product_1": {
            "id": p1.id,
            "name": p1.name,
            "brand": p1.brand,
            "price": p1.mrp,
            "net_quantity": p1.net_quantity,
            "unit_sale_price": p1.unit_sale_price,
            "calories": n1.get("calories", "N/A"),
            "protein_g": n1.get("protein_g", "N/A"),
            "sugar_g": n1.get("sugar_g", "N/A"),
            "fat_g": n1.get("fat_g", "N/A"),
            "sodium_mg": n1.get("sodium_mg", "N/A"),
            "basis": n1.get("serving_size", "per 100g")
        },
        "product_2": {
            "id": p2.id,
            "name": p2.name,
            "brand": p2.brand,
            "price": p2.mrp,
            "net_quantity": p2.net_quantity,
            "unit_sale_price": p2.unit_sale_price,
            "calories": n2.get("calories", "N/A"),
            "protein_g": n2.get("protein_g", "N/A"),
            "sugar_g": n2.get("sugar_g", "N/A"),
            "fat_g": n2.get("fat_g", "N/A"),
            "sodium_mg": n2.get("sodium_mg", "N/A"),
            "basis": n2.get("serving_size", "per 100g")
        },
        "disclaimer": "Nutritional comparisons are normalized per 100g for objective analysis. These values are distinct from statutory Legal Metrology compliance."
    }

# 5. Submit Community Review
@router.post("/reviews")
def submit_review(
    req: ReviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    review = ProductReview(
        user_id=current_user.id,
        user_name=current_user.full_name,
        product_id=req.product_id,
        overall_rating=max(1.0, min(5.0, req.overall_rating)),
        taste_rating=req.taste_rating,
        vfm_rating=req.vfm_rating,
        would_buy_again=req.would_buy_again,
        review_text=req.review_text.strip() if req.review_text else None,
        created_at=utc_now()
    )
    db.add(review)
    db.commit()
    return {"success": True, "message": "Thank you for submitting your community feedback!"}

# 6. Saved Products (Bookmarks)
@router.get("/saved")
def get_saved_products(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    saved = db.query(UserSavedProduct).filter(UserSavedProduct.user_id == current_user.id).all()
    results = []
    for s in saved:
        p = db.query(Product).filter(Product.id == s.product_id).first()
        if p:
            results.append({
                "saved_id": s.id,
                "product_id": p.id,
                "name": p.name,
                "brand": p.brand,
                "mrp": p.mrp,
                "net_quantity": p.net_quantity,
                "category": p.category,
                "saved_at": s.created_at.strftime("%Y-%m-%d")
            })
    return results

@router.post("/saved/{product_id}")
def save_product(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    exists = db.query(UserSavedProduct).filter(
        UserSavedProduct.user_id == current_user.id,
        UserSavedProduct.product_id == product_id
    ).first()
    if not exists:
        sp = UserSavedProduct(user_id=current_user.id, product_id=product_id)
        db.add(sp)
        db.commit()
    return {"success": True, "message": "Product saved to your favorites."}

@router.delete("/saved/{product_id}")
def remove_saved_product(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sp = db.query(UserSavedProduct).filter(
        UserSavedProduct.user_id == current_user.id,
        UserSavedProduct.product_id == product_id
    ).first()
    if sp:
        db.delete(sp)
        db.commit()
    return {"success": True, "message": "Product removed from saved items."}
