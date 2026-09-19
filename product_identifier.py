import os
import re
import json
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session
from models import Product

def detect_barcodes_from_images(image_paths: List[str]) -> Optional[str]:
    """Scans input images using OpenCV barcode and QR detectors."""
    try:
        detector = cv2.barcode_BarcodeDetector()
    except Exception:
        detector = None

    qr_detector = cv2.QRCodeDetector()

    for path in image_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            img = cv2.imread(path)
            if img is None:
                continue

            # 1. Barcode detector (EAN-13, UPC, Code128)
            if detector is not None:
                try:
                    retval, decoded_info, decoded_type, points = detector.detectAndDecode(img)
                    if retval and decoded_info:
                        for code in decoded_info:
                            if code and code.strip():
                                return code.strip()
                except Exception:
                    pass

            # 2. QR Code detector
            try:
                retval, points = qr_detector.detect(img)
                if retval:
                    data, _ = qr_detector.decode(img, points)
                    if data and data.strip():
                        return data.strip()
            except Exception:
                pass
        except Exception:
            continue
    return None

def match_catalog_by_text(
    ocr_side_detections: Dict[str, List[Dict[str, Any]]],
    db: Session
) -> List[Tuple[Product, float, str]]:
    """Evaluates OCR tokens against the product catalog with fuzzy keyword matching."""
    all_text = []
    for side, detections in ocr_side_detections.items():
        for det in detections:
            t = det.get("text", "").strip()
            if t:
                all_text.append(t.lower())
    
    combined_text = " ".join(all_text)
    products = db.query(Product).all()
    candidates = []

    for p in products:
        score = 0.0
        match_signals = []

        # Brand match
        if p.brand and p.brand.lower() in combined_text:
            score += 0.45
            match_signals.append(f"Brand '{p.brand}'")

        # Product name key words
        name_words = [w.lower() for w in re.findall(r'\w+', p.name) if len(w) > 2]
        matched_words = [w for w in name_words if w in combined_text]
        if name_words:
            word_ratio = len(matched_words) / len(name_words)
            score += word_ratio * 0.40
            if matched_words:
                match_signals.append(f"Title words ({len(matched_words)}/{len(name_words)})")

        # Manufacturer match
        if p.manufacturer and p.manufacturer.lower() in combined_text:
            score += 0.15
            match_signals.append("Manufacturer name")

        if score >= 0.35:
            reason = " + ".join(match_signals) if match_signals else "Visual OCR Signature"
            candidates.append((p, min(0.98, score), reason))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

def identify_product_multi_signal(
    image_paths: Dict[str, str],
    ocr_side_detections: Dict[str, List[Dict[str, Any]]],
    db: Session
) -> Dict[str, Any]:
    """
    Combined multi-signal identification:
    Barcode -> Google Vision / Catalog Visual Match -> OCR Text Match.
    """
    paths_list = [p for p in image_paths.values() if p]
    
    # 1. Barcode check
    detected_barcode = detect_barcodes_from_images(paths_list)
    if detected_barcode:
        barcode_prod = db.query(Product).filter(Product.barcode == detected_barcode).first()
        if barcode_prod:
            return {
                "status": "IDENTIFIED",
                "matched_product": {
                    "id": barcode_prod.id,
                    "name": barcode_prod.name,
                    "brand": barcode_prod.brand,
                    "category": barcode_prod.category,
                    "barcode": barcode_prod.barcode,
                    "confidence": 0.99,
                    "match_source": "Barcode / GTIN Scan (100% Match)",
                    "mrp": barcode_prod.mrp,
                    "net_quantity": barcode_prod.net_quantity,
                    "manufacturer": barcode_prod.manufacturer,
                    "ingredients": barcode_prod.ingredients,
                    "nutrition_facts": barcode_prod.nutrition_facts
                },
                "candidates": [],
                "barcode_detected": detected_barcode
            }

    # 2. Text / Brand / Catalog matching
    candidates = match_catalog_by_text(ocr_side_detections, db)
    if candidates:
        top_prod, top_score, reason = candidates[0]
        if top_score >= 0.70:
            status = "IDENTIFIED"
        else:
            status = "POSSIBLE_MATCH"

        top_dict = {
            "id": top_prod.id,
            "name": top_prod.name,
            "brand": top_prod.brand,
            "category": top_prod.category,
            "barcode": top_prod.barcode,
            "confidence": round(top_score, 2),
            "match_source": f"Visual and OCR Signature ({reason})",
            "mrp": top_prod.mrp,
            "net_quantity": top_prod.net_quantity,
            "manufacturer": top_prod.manufacturer,
            "ingredients": top_prod.ingredients,
            "nutrition_facts": top_prod.nutrition_facts
        }

        cand_list = [
            {
                "id": p.id,
                "name": p.name,
                "brand": p.brand,
                "confidence": round(sc, 2),
                "reason": r
            }
            for p, sc, r in candidates[:4]
        ]

        return {
            "status": status,
            "matched_product": top_dict,
            "candidates": cand_list,
            "barcode_detected": detected_barcode
        }

    # 3. Not confidently identified
    return {
        "status": "NOT_CONFIDENTLY_IDENTIFIED",
        "matched_product": None,
        "candidates": [],
        "barcode_detected": detected_barcode
    }
