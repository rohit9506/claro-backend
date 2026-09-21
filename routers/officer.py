import os
import re
import json
import uuid
import datetime
import asyncio
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from config import UPLOAD_DIR, REPORT_DIR
from database import get_db
from models import User, Product, Inspection, InspectionImage, FieldValidation, OCRResult, Rule, Verification
from auth_deps import get_current_user, require_role, log_audit
from cv_quality import evaluate_image_quality, check_and_enhance_image
from ocr_service import ocr_service
from field_extractor import extract_declarations_from_multi_side
from rule_engine import evaluate_legal_metrology_rules
from report_generator import generate_inspection_pdf, render_pdf_to_base64_pages
from product_identifier import identify_product_multi_signal

router = APIRouter(prefix="/officer", tags=["Officer Inspection"])

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

# 1. Image Quality Evaluation (Step-by-step scanner check)
@router.post("/quality-check")
async def check_quality(
    side: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Evaluates captured package image for blur, lighting, and presence.
    Gives immediate accept/retake feedback without exposing technical internals.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="No image file received.")

    result = evaluate_image_quality(content)
    result["side"] = side
    return result

# 2. Officer Dashboard KPIs
@router.get("/stats")
def get_officer_stats(
    current_user: User = Depends(require_role("ROLE_OFFICER", "ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    query = db.query(Inspection)
    if current_user.role == "ROLE_OFFICER":
        query = query.filter(Inspection.officer_id == current_user.id)

    total_inspections = query.count()
    compliant_count = query.filter(Inspection.status == "COMPLIANT").count()
    issues_count = query.filter(Inspection.status == "NON_COMPLIANT").count()
    pending_count = query.filter(Inspection.status == "PENDING_VERIFICATION").count()

    today_start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_inspections = query.filter(Inspection.created_at >= today_start).count()

    return {
        "total_inspections": total_inspections,
        "today_inspections": today_inspections,
        "compliant_count": compliant_count,
        "issues_count": issues_count,
        "pending_verification_count": pending_count
    }

# 3. Analyze Product (Core AI/OCR + Rule Engine Pipeline)
@router.post("/analyze")
async def analyze_product(
    front_image: Optional[UploadFile] = File(None),
    back_image: Optional[UploadFile] = File(None),
    right_image: Optional[UploadFile] = File(None),
    left_image: Optional[UploadFile] = File(None),
    product_name: Optional[str] = Form(None),
    location: Optional[str] = Form("Field Inspection Premise"),
    officer_notes: Optional[str] = Form("Standard first-level Legal Metrology package audit."),
    current_user: User = Depends(require_role("ROLE_OFFICER", "ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    """
    Full pipeline:
    Images -> Preprocessing -> OCR -> Multi-side Field Extraction -> Rule Engine -> Database -> PDF Report
    """
    # Flexible 2, 3, or 4 image verification: requires at least 2 views
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

    # Helper to save and process an uploaded side
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

        # Measure dimensions
        w, h = 1280, 720
        try:
            from PIL import Image as PILImg
            with PILImg.open(filepath) as pimg:
                w, h = pimg.size
        except Exception:
            pass

        images_saved[side_name] = f"/uploads/{filename}"
        images_metadata[side_name] = {
            "path": f"/uploads/{filename}",
            "file_size": len(content),
            "width": w,
            "height": h,
            "quality_score": round(float(q_info.get("enhanced_blur", q_info.get("blur_score", 100.0))), 1)
        }
        return str(filepath)

    # Process file uploads and quality checks
    save_tasks = [process_side(name, upload) for name, upload in available_uploads]
    await asyncio.gather(*save_tasks)

    # Run OCR per side in worker thread pool without memory spikes
    loop = asyncio.get_event_loop()
    for side_name, _ in available_uploads:
        filepath = UPLOAD_DIR / f"pkg_{scan_id}_{side_name}.jpg"
        if filepath.exists():
            detections = await loop.run_in_executor(None, ocr_service.extract_text_with_boxes, str(filepath))
            ocr_side_detections[side_name] = detections
        else:
            ocr_side_detections[side_name] = []

    # 1. Multi-side Field Extraction across available views
    extracted_declarations = extract_declarations_from_multi_side(ocr_side_detections)

    # 1b. Multi-Signal Product Identification (Barcode + Visual Features + Multimodal Assistance + OCR Text Match)
    prod_ident = identify_product_multi_signal(
        image_paths={k: str(UPLOAD_DIR / f"pkg_{scan_id}_{k}.jpg") for k in images_saved.keys()},
        ocr_side_detections=ocr_side_detections,
        extracted_declarations=extracted_declarations,
        db=db
    )

    # 1c. Product Identity Resolution: Physical package is Primary Source of Truth
    def is_filename(val: Optional[str]) -> bool:
        if not val or not str(val).strip():
            return True
        v = str(val).strip().lower()
        if any(v.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".pdf", ".svg"]):
            return True
        if re.search(r"^(web|front|back|side|image|img|photo|pic|screenshot|scan|upload)(\.\w+)?$", v):
            return True
        return False

    matched_p = prod_ident.get("matched_product")
    if matched_p and matched_p.get("name") and not is_filename(matched_p.get("name")):
        final_product_name = matched_p["name"]
        final_brand = matched_p.get("brand") or "Brand on Package"
        final_variant = matched_p.get("variant")
    elif extracted_declarations.get("product_name", {}).get("detected") and extracted_declarations["product_name"]["value"] not in ["Not detected", ""] and not is_filename(extracted_declarations["product_name"]["value"]):
        final_product_name = extracted_declarations["product_name"]["value"]
        final_brand = extracted_declarations.get("brand", {}).get("value") or "Brand on Package"
        final_variant = extracted_declarations.get("variant", {}).get("value")
    elif product_name and product_name not in ["Packaged Retail Commodity", "Packaged Commodity", ""] and not is_filename(product_name):
        final_product_name = product_name
        final_brand = extracted_declarations.get("brand", {}).get("value") or "Brand on Package"
        final_variant = extracted_declarations.get("variant", {}).get("value")
    else:
        final_product_name = "Product could not be confidently identified."
        final_brand = "Not confidently detected"
        final_variant = None

    # 2. Query Active Rules from Database
    active_rules = db.query(Rule).filter(Rule.is_active == True).all()

    # 3. Evaluate Compliance via Deterministic Rule Engine
    validations_res, overall_status, pass_c, fail_c, review_c, correction_guidance = evaluate_legal_metrology_rules(
        extracted_declarations, active_rules
    )

    if final_product_name == "Product could not be confidently identified." and overall_status == "COMPLIANT":
        overall_status = "PENDING_VERIFICATION"
        review_c += 1

    # 4. Persist to Centralized Database
    inspection = Inspection(
        inspection_number=scan_id,
        officer_id=current_user.id,
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
        location=location,
        officer_notes=officer_notes,
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
        "officer_name": current_user.full_name,
        "officer_badge": current_user.officer_id or "OFF-INSP",
        "location": location,
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
        "officer_notes": officer_notes,
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
        print(f"[WARN] PDF generation deferred: {pdf_err}")

    # Audit log
    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="INSPECTION_COMPLETED",
        target_type="INSPECTION", target_id=str(inspection.id),
        details={"inspection_number": scan_id, "status": overall_status}
    )

    return {
        "success": True,
        "scan_id": scan_id,
        "inspection_id": inspection.id,
        "inspection_number": scan_id,
        "status": overall_status,
        "product_name": final_product_name,
        "brand": final_brand,
        "variant": final_variant,
        "mrp": extracted_declarations.get("mrp", {}).get("value"),
        "net_quantity": extracted_declarations.get("net_quantity", {}).get("value"),
        "pass_count": pass_c,
        "fail_count": fail_c,
        "review_count": review_c,
        "validations": validations_res,
        "correction_guidance": correction_guidance,
        "extracted_declarations": extracted_declarations,
        "matched_product": matched_p,
        "evidence_source": prod_ident.get("evidence_source"),
        "barcode_detected": prod_ident.get("barcode_detected"),
        "front_image": images_saved.get("front"),
        "back_image": images_saved.get("back"),
        "right_image": images_saved.get("right_side"),
        "left_image": images_saved.get("left_side"),
        "images": images_saved,
        "created_at": inspection.created_at.isoformat(),
        "pdf_url": f"/officer/inspections/{inspection.id}/report/pdf"
    }

# 4. List Inspections with Search and Filters
@router.get("/inspections")
def list_inspections(
    status_filter: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(require_role("ROLE_OFFICER", "ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    query = db.query(Inspection)
    if current_user.role == "ROLE_OFFICER":
        query = query.filter(Inspection.officer_id == current_user.id)

    if status_filter:
        query = query.filter(Inspection.status == status_filter)

    if search:
        s = f"%{search.strip()}%"
        query = query.filter(
            (Inspection.product_name.ilike(s)) | (Inspection.inspection_number.ilike(s))
        )

    inspections = query.order_by(Inspection.created_at.desc()).all()

    return [
        {
            "id": insp.id,
            "inspection_number": insp.inspection_number,
            "product_name": insp.product_name,
            "status": insp.status,
            "pass_count": insp.pass_count,
            "fail_count": insp.fail_count,
            "review_count": insp.review_count,
            "location": insp.location,
            "created_at": insp.created_at.strftime("%Y-%m-%d %H:%M") if insp.created_at else None,
            "has_report": (REPORT_DIR / f"{insp.inspection_number}.pdf").exists()
        }
        for insp in inspections
    ]

# 5. Inspection Details with Visual Evidence coordinates
@router.get("/inspections/{inspection_id}")
def get_inspection_detail(
    inspection_id: int,
    current_user: User = Depends(require_role("ROLE_OFFICER", "ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    insp = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not insp:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    validations = []
    for v in insp.validations:
        bbox = json.loads(v.bbox_json) if v.bbox_json else [0.1, 0.1, 0.3, 0.9]
        validations.append({
            "id": v.id,
            "field_name": v.field_name,
            "detected_value": v.detected_value,
            "requirement_summary": v.requirement_summary,
            "rule_reference": v.rule_reference,
            "status": v.status,
            "confidence": v.confidence,
            "reason": v.reason,
            "image_side": v.image_side,
            "bbox_norm": bbox
        })

    return {
        "id": insp.id,
        "inspection_number": insp.inspection_number,
        "product_name": insp.product_name,
        "status": insp.status,
        "pass_count": insp.pass_count,
        "fail_count": insp.fail_count,
        "review_count": insp.review_count,
        "front_image": insp.front_image,
        "back_image": insp.back_image,
        "right_image": insp.right_image,
        "left_image": insp.left_image,
        "location": insp.location,
        "officer_notes": insp.officer_notes,
        "officer_name": insp.officer.full_name if insp.officer else "Enforcement Officer",
        "created_at": insp.created_at.strftime("%Y-%m-%d %H:%M") if insp.created_at else None,
        "validations": validations,
        "report_url": f"/officer/inspections/{insp.id}/report.pdf"
    }

def ensure_inspection_pdf(insp: Inspection, force_regenerate: bool = False) -> Path:
    pdf_path = REPORT_DIR / f"{insp.inspection_number}.pdf"
    needs_gen = force_regenerate or (not pdf_path.exists())
    if not needs_gen and insp.updated_at:
        try:
            if pdf_path.stat().st_mtime < insp.updated_at.timestamp():
                needs_gen = True
        except Exception:
            needs_gen = True

    if needs_gen:
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
            "officer_name": insp.officer.full_name if insp.officer else "Enforcement Officer",
            "officer_badge": insp.officer.officer_id if insp.officer else "LM-OFF",
            "location": insp.location,
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
    return pdf_path

# 6. Download or View Inspection Report PDF
@router.get("/inspections/{inspection_id}/report.pdf")
def get_inspection_report(
    inspection_id: int,
    inline: bool = Query(True),
    current_user: User = Depends(require_role("ROLE_OFFICER", "ROLE_ADMIN", "ROLE_CONSUMER")),
    db: Session = Depends(get_db)
):
    insp = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not insp:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    pdf_path = ensure_inspection_pdf(insp)

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="REPORT_DOWNLOADED",
        target_type="INSPECTION", target_id=str(insp.id)
    )

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"Claro_Inspection_{insp.inspection_number}.pdf",
        content_disposition_type="inline" if inline else "attachment"
    )

# 6b. In-App High-Resolution PDF Report Preview (Pages as Image URLs)
@router.get("/inspections/{inspection_id}/report/preview")
def get_inspection_report_preview(
    inspection_id: int,
    current_user: User = Depends(require_role("ROLE_OFFICER", "ROLE_ADMIN", "ROLE_CONSUMER")),
    db: Session = Depends(get_db)
):
    insp = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not insp:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    pdf_path = ensure_inspection_pdf(insp)
    pages = render_pdf_to_base64_pages(str(pdf_path), scale=2.0)

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="REPORT_PREVIEWED",
        target_type="INSPECTION", target_id=str(insp.id)
    )

    return {
        "success": True,
        "inspection_id": insp.id,
        "inspection_number": insp.inspection_number,
        "product_name": insp.product_name,
        "status": insp.status,
        "total_pages": len(pages),
        "pages": pages,
        "pdf_url": f"/officer/inspections/{insp.id}/report.pdf"
    }

# 7. Delete Inspection Record
@router.delete("/inspections/{inspection_id}")
def delete_inspection(
    inspection_id: int,
    current_user: User = Depends(require_role("ROLE_OFFICER", "ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    insp = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not insp:
        raise HTTPException(status_code=404, detail="Inspection record not found.")

    # Officer can only delete their own inspection; Admin can delete any
    if current_user.role == "ROLE_OFFICER" and insp.officer_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have permission to delete this inspection record.")

    insp_num = insp.inspection_number

    # Remove generated PDF report if present
    pdf_path = REPORT_DIR / f"{insp_num}.pdf"
    if pdf_path.exists():
        try:
            os.remove(pdf_path)
        except Exception:
            pass

    # Remove stored images from disk
    for img_attr in [insp.front_image, insp.back_image, insp.right_image, insp.left_image]:
        if img_attr and img_attr.startswith("/uploads/"):
            fpath = UPLOAD_DIR / img_attr.replace("/uploads/", "")
            if fpath.exists():
                try:
                    os.remove(fpath)
                except Exception:
                    pass

    # Cascade delete DB associations (including pending verifications and captured images)
    db.query(Verification).filter(Verification.inspection_id == inspection_id).delete()
    db.query(InspectionImage).filter(InspectionImage.inspection_id == inspection_id).delete()
    db.query(FieldValidation).filter(FieldValidation.inspection_id == inspection_id).delete()
    db.query(OCRResult).filter(OCRResult.inspection_id == inspection_id).delete()
    db.delete(insp)
    db.commit()

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="INSPECTION_DELETED",
        target_type="INSPECTION", target_id=str(inspection_id),
        details={"inspection_number": insp_num}
    )

    return {
        "success": True,
        "message": f"Inspection report {insp_num} and associated records deleted permanently."
    }

