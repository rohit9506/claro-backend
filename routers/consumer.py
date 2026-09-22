import os
import re
import json
import uuid
import datetime
import asyncio
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from pathlib import Path
from fastapi.responses import FileResponse
from PIL import Image as PILImage

from config import UPLOAD_DIR, REPORT_DIR
from database import get_db
from models import User, Product, ProductReview, UserSavedProduct, Inspection, InspectionImage, FieldValidation, OCRResult, Rule
from auth_deps import get_current_user, require_role, log_audit
from cv_quality import check_and_enhance_image
from ocr_service import ocr_service
from product_identifier import identify_product_multi_signal
from field_extractor import extract_declarations_from_multi_side
from rule_engine import evaluate_legal_metrology_rules
from report_generator import generate_inspection_pdf, render_pdf_to_base64_pages

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
    right_image: Optional[UploadFile] = File(None),
    left_image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Consumer scanning - supports flexible 2, 3, or 4 package views.
    Executes identical visual product identification pipeline, multi-side OCR, 
    statutory Legal Metrology rule engine, database audit persistence, and PDF generation.
    """
    available_uploads = []
    if front_image: available_uploads.append(("front", front_image))
    if back_image: available_uploads.append(("back", back_image))
    if right_image: available_uploads.append(("right_side", right_image))
    if left_image: available_uploads.append(("left_side", left_image))

    if len(available_uploads) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two package views are required to run Legal Metrology audit."
        )

    scan_id = f"CLARO-SCAN-2026-{uuid.uuid4().hex[:8].upper()}"
    images_saved = {}
    images_metadata = {}
    ocr_side_detections = {}

    async def process_side(side_name: str, upload_file: UploadFile):
        if not upload_file:
            return None
        content = await upload_file.read()
        if not content:
            return None
        
        filename = f"pkg_{scan_id}_{side_name}.jpg"
        filepath = UPLOAD_DIR / filename
        with open(filepath, "wb") as f:
            f.write(content)
        
        # Quality check: auto-fix mild blur with unsharp mask/CLAHE, or raise error if severe blur
        q_info = check_and_enhance_image(str(filepath))
        if not q_info.get("can_proceed"):
            friendly_side = side_name.replace("_", " ").title()
            raise HTTPException(
                status_code=400,
                detail=f"The {friendly_side} package photo is too blurry to reliably detect text. Please hold the camera steady, ensure good lighting, and retake."
            )

        # Record metadata
        try:
            with PILImage.open(filepath) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0

        file_size = len(content)
        quality_score = float(q_info.get("laplacian_var", 100.0))

        images_saved[side_name] = f"/uploads/{filename}"
        images_metadata[side_name] = {
            "file_size": file_size,
            "width": w,
            "height": h,
            "quality_score": quality_score
        }
        return filepath

    # Process file uploads and quality checks
    save_tasks = [process_side(name, upload) for name, upload in available_uploads]
    await asyncio.gather(*save_tasks)

    # Run OCR across available views sequentially for memory safety (~180MB RAM) and zero OpenMP thread contention
    for s_name, _ in available_uploads:
        fp = UPLOAD_DIR / f"pkg_{scan_id}_{s_name}.jpg"
        if fp.exists():
            ocr_side_detections[s_name] = ocr_service.extract_text_with_boxes(str(fp))
        else:
            ocr_side_detections[s_name] = []

    # 1. Multi-Side Declaration Aggregation
    extracted_declarations = extract_declarations_from_multi_side(ocr_side_detections)

    # 2. Multi-Signal Convergence Product Identification (Barcode + Visual + Physical Package OCR)
    prod_ident = identify_product_multi_signal(
        image_paths={k: str(UPLOAD_DIR / f"pkg_{scan_id}_{k}.jpg") for k in images_saved.keys()},
        ocr_side_detections=ocr_side_detections,
        extracted_declarations=extracted_declarations,
        db=db
    )

    matched_p = prod_ident.get("matched_product")
    matched_product_id = matched_p.get("id") if matched_p else None

    # Canonical product name resolution
    def is_filename(val: Optional[str]) -> bool:
        if not val or not str(val).strip():
            return True
        v = str(val).strip().lower()
        if any(v.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".pdf", ".svg"]):
            return True
        if re.search(r"^(web|front|back|side|image|img|photo|pic|screenshot|scan|upload)(\.\w+)?$", v):
            return True
        return False

    if matched_p and matched_p.get("name") and not is_filename(matched_p.get("name")):
        final_product_name = matched_p["name"]
        final_brand = matched_p.get("brand") or "Brand on Package"
        final_variant = matched_p.get("variant")
        verification_status = "Verified"
    elif extracted_declarations.get("product_name", {}).get("detected") and extracted_declarations["product_name"]["value"] not in ["Not detected", ""] and not is_filename(extracted_declarations["product_name"]["value"]):
        final_product_name = extracted_declarations["product_name"]["value"]
        final_brand = extracted_declarations.get("brand", {}).get("value") or "Brand on Package"
        final_variant = extracted_declarations.get("variant", {}).get("value")
        verification_status = "Identified via Physical Package"
    else:
        final_product_name = "Product could not be confidently identified."
        final_brand = "Not confidently detected"
        final_variant = None
        verification_status = "Needs Verification"

    # 3. Query Active Rules & Evaluate Compliance via Deterministic Rule Engine
    active_rules = db.query(Rule).filter(Rule.is_active == True).all()
    validations_res, overall_status, pass_c, fail_c, review_c, correction_guidance = evaluate_legal_metrology_rules(
        extracted_declarations, active_rules
    )

    if final_product_name == "Product could not be confidently identified." and overall_status == "COMPLIANT":
        overall_status = "PENDING_VERIFICATION"
        review_c += 1

    # 4. Persist to Centralized Database
    inspection = Inspection(
        inspection_number=scan_id,
        officer_id=None,
        product_id=matched_product_id,
        product_name=final_product_name,
        brand=final_brand,
        variant=final_variant,
        mrp=extracted_declarations.get("mrp", {}).get("value"),
        net_quantity=extracted_declarations.get("net_quantity", {}).get("value"),
        source="CAMERA",
        status=overall_status,
        pass_count=pass_c,
        fail_count=fail_c,
        review_count=review_c,
        front_image=images_saved.get("front"),
        back_image=images_saved.get("back"),
        right_image=images_saved.get("right_side"),
        left_image=images_saved.get("left_side"),
        location="Consumer Scan",
        officer_notes="Consumer package scan and statutory verification.",
        created_at=utc_now(),
        finalized_at=utc_now() if overall_status != "PENDING_VERIFICATION" else None
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    # Persist all captured package images with complete metadata
    for side_k, img_rel_path in images_saved.items():
        meta = images_metadata.get(side_k, {})
        insp_img = InspectionImage(
            inspection_id=inspection.id,
            side=side_k,
            view_type=side_k,
            image_path=img_rel_path,
            source=inspection.source,
            mime_type="image/jpeg",
            file_size=meta.get("file_size"),
            width=meta.get("width"),
            height=meta.get("height"),
            quality_score=meta.get("quality_score", 100.0),
            is_accepted=True,
            quality_notes=f"Accepted during package inspection ({side_k})",
            processing_status="PROCESSED"
        )
        db.add(insp_img)

    # Save Field Validations
    field_records = []
    for v in validations_res:
        fv = FieldValidation(
            inspection_id=inspection.id,
            field_name=v["field_name"],
            detected_value=v["detected_value"],
            requirement_summary=v["requirement_summary"],
            rule_reference=v["rule_reference"],
            status=v.get("status", v.get("raw_status", "REVIEW")),
            confidence=v["confidence"],
            reason=v["reason"],
            image_side=v["image_side"],
            bbox_json=json.dumps(v["bbox_norm"])
        )
        field_records.append(fv)
    db.add_all(field_records)

    # Save raw OCR outputs
    for side_k, dets in ocr_side_detections.items():
        ocr_rec = OCRResult(
            inspection_id=inspection.id,
            image_side=side_k,
            raw_text=ocr_service.get_full_text(dets),
            boxes_json=json.dumps(dets)
        )
        db.add(ocr_rec)

    db.commit()

    # 5. Generate Official Digital Inspection Report PDF
    pdf_payload = {
        "inspection_number": scan_id,
        "officer_name": "Consumer Verification",
        "officer_badge": "CONSUMER",
        "location": "Consumer Scan",
        "product_name": final_product_name,
        "brand": final_brand,
        "variant": final_variant,
        "mrp": extracted_declarations.get("mrp", {}).get("value"),
        "net_quantity": extracted_declarations.get("net_quantity", {}).get("value"),
        "created_at": inspection.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        "status": overall_status,
        "pass_count": pass_c,
        "fail_count": fail_c,
        "review_count": review_c,
        "officer_notes": "Consumer package scan and statutory verification.",
        "validations": validations_res,
        "correction_guidance": correction_guidance,
        "front_image": images_saved.get("front"),
        "back_image": images_saved.get("back"),
        "right_image": images_saved.get("right_side"),
        "left_image": images_saved.get("left_side"),
        "images": images_saved
    }
    try:
        generate_inspection_pdf(pdf_payload)
    except Exception as pdf_err:
        print(f"[WARN] Consumer PDF generation deferred: {pdf_err}")

    # Community reviews/stats if product is in catalog
    community_stats = None
    if matched_product_id:
        reviews = db.query(ProductReview).filter(ProductReview.product_id == matched_product_id).all()
        avg_overall = sum(r.overall_rating for r in reviews) / len(reviews) if reviews else 0.0
        community_stats = {
            "review_count": len(reviews),
            "avg_overall": round(avg_overall, 1)
        }

    return {
        "success": True,
        "scan_id": scan_id,
        "inspection_id": inspection.id,
        "inspection_number": scan_id,
        "status": overall_status,
        "overall_status": overall_status,
        "product_id": matched_product_id,
        "product_name": final_product_name,
        "brand": final_brand,
        "variant": final_variant,
        "mrp": extracted_declarations.get("mrp", {}).get("value"),
        "net_quantity": extracted_declarations.get("net_quantity", {}).get("value"),
        "verification_status": verification_status,
        "matched": bool(matched_p),
        "matched_product": matched_p,
        "candidates": prod_ident.get("candidates", []),
        "barcode_detected": prod_ident.get("barcode_detected"),
        "extracted_declarations": extracted_declarations,
        "pass_count": pass_c,
        "fail_count": fail_c,
        "review_count": review_c,
        "validations": validations_res,
        "correction_guidance": correction_guidance,
        "report_url": f"/api/consumer/inspections/{inspection.id}/report.pdf",
        "community_stats": community_stats,
        "front_image": images_saved.get("front"),
        "back_image": images_saved.get("back"),
        "right_image": images_saved.get("right_side"),
        "left_image": images_saved.get("left_side"),
        "images": images_saved
    }

# 3b. Download or View Consumer Inspection Report PDF
@router.get("/inspections/{inspection_id}/report.pdf")
def get_consumer_inspection_report(
    inspection_id: int,
    inline: bool = Query(True),
    db: Session = Depends(get_db)
):
    insp = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not insp:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    pdf_path = REPORT_DIR / f"{insp.inspection_number}.pdf"
    if not pdf_path.exists():
        validations = [
            {
                "field_name": v.field_name,
                "detected_value": v.detected_value,
                "rule_reference": v.rule_reference,
                "status": v.status,
                "confidence": v.confidence,
                "reason": v.reason
            }
            for v in insp.validations
        ]
        generate_inspection_pdf({
            "inspection_number": insp.inspection_number,
            "officer_name": "Consumer Verification",
            "officer_badge": "CONSUMER",
            "location": insp.location or "Consumer Scan",
            "product_name": insp.product_name,
            "brand": insp.brand,
            "variant": insp.variant,
            "mrp": insp.mrp,
            "net_quantity": insp.net_quantity,
            "created_at": insp.created_at.strftime("%Y-%m-%d %H:%M UTC") if insp.created_at else datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "status": insp.status,
            "pass_count": insp.pass_count,
            "fail_count": insp.fail_count,
            "review_count": insp.review_count,
            "officer_notes": insp.officer_notes,
            "validations": validations,
            "front_image": insp.front_image,
            "back_image": insp.back_image,
            "right_image": insp.right_image,
            "left_image": insp.left_image
        })

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"Claro_Inspection_{insp.inspection_number}.pdf",
        content_disposition_type="inline" if inline else "attachment"
    )

# 3c. In-App High-Resolution PDF Report Preview for Consumer
@router.get("/inspections/{inspection_id}/report/preview")
def get_consumer_inspection_report_preview(
    inspection_id: int,
    db: Session = Depends(get_db)
):
    insp = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not insp:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    pdf_path = REPORT_DIR / f"{insp.inspection_number}.pdf"
    if not pdf_path.exists():
        validations = [
            {
                "field_name": v.field_name,
                "detected_value": v.detected_value,
                "rule_reference": v.rule_reference,
                "status": v.status,
                "confidence": v.confidence,
                "reason": v.reason
            }
            for v in insp.validations
        ]
        generate_inspection_pdf({
            "inspection_number": insp.inspection_number,
            "officer_name": "Consumer Verification",
            "officer_badge": "CONSUMER",
            "location": insp.location or "Consumer Scan",
            "product_name": insp.product_name,
            "brand": insp.brand,
            "variant": insp.variant,
            "mrp": insp.mrp,
            "net_quantity": insp.net_quantity,
            "created_at": insp.created_at.strftime("%Y-%m-%d %H:%M UTC") if insp.created_at else datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "status": insp.status,
            "pass_count": insp.pass_count,
            "fail_count": insp.fail_count,
            "review_count": insp.review_count,
            "officer_notes": insp.officer_notes,
            "validations": validations,
            "front_image": insp.front_image,
            "back_image": insp.back_image,
            "right_image": insp.right_image,
            "left_image": insp.left_image
        })

    pages = render_pdf_to_base64_pages(str(pdf_path), scale=2.0)
    return {
        "success": True,
        "inspection_id": insp.id,
        "inspection_number": insp.inspection_number,
        "product_name": insp.product_name,
        "status": insp.status,
        "total_pages": len(pages),
        "pages": pages,
        "pdf_url": f"/api/consumer/inspections/{insp.id}/report.pdf"
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
