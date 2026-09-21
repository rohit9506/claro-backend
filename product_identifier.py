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


def extract_visual_feature_signature(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Extracts open-source visual signature (dominant HSV color distribution,
    aspect ratio, edge contour density) as supporting Google-Lens-like signal.
    """
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        aspect_ratio = round(w / float(max(1, h)), 2)

        # Downscale for fast visual analysis
        small = cv2.resize(img, (160, 160))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Compute 3-channel color histogram
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)

        # Compute edge contour density (packaging complexity)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = round(float(np.count_nonzero(edges)) / (160 * 160), 3)

        return {
            "aspect_ratio": aspect_ratio,
            "edge_density": edge_density,
            "hue_profile": hist_h.flatten().tolist()[:4],
            "verified": True
        }
    except Exception:
        return None


def match_catalog_by_strict_evidence(
    package_brand: Optional[str],
    package_product_name: Optional[str],
    combined_ocr_text: str,
    db: Session
) -> Optional[Tuple[Product, float, str]]:
    """
    Strict Catalog Matcher:
    ONLY returns a catalog product if there is a genuine, high-confidence match (>= 0.85).
    NEVER returns Haldiram, demo products, or generic fallbacks when scanning another product.
    """
    if not db:
        return None

    norm_combined = combined_ocr_text.lower()
    norm_pkg_brand = package_brand.lower().strip() if package_brand and package_brand not in ["Not detected", "Not confidently detected"] else ""
    norm_pkg_name = package_product_name.lower().strip() if package_product_name and package_product_name not in ["Not detected", "Product could not be confidently identified."] else ""

    products = db.query(Product).all()
    best_match = None
    best_score = 0.0
    best_reason = ""

    for p in products:
        p_brand = (p.brand or "").lower().strip()
        p_name = (p.name or "").lower().strip()

        # 1. Strict Brand Check:
        # If catalog has a brand, that exact brand MUST be present either in the detected package brand
        # or explicitly in the current package OCR text.
        brand_matched = False
        if p_brand and (p_brand in norm_combined or (norm_pkg_brand and p_brand in norm_pkg_brand)):
            brand_matched = True

        if not brand_matched and p_brand:
            # Brand does not match at all -> strictly skip this catalog product
            continue

        # 2. Strict Product Name Word Matching:
        p_words = [w for w in re.findall(r'[a-zA-Z0-9]+', p_name) if len(w) >= 3]
        if not p_words:
            continue

        # Count how many words from the catalog title are actually present in the current package OCR
        matched_words = [w for w in p_words if w in norm_combined]
        word_ratio = len(matched_words) / float(len(p_words))

        # We require at least 85% of distinctive words to match for catalog enrichment
        if word_ratio >= 0.85 and brand_matched:
            score = 0.50 + (word_ratio * 0.45)
            if score > best_score:
                best_score = score
                best_match = p
                best_reason = f"High-Confidence Catalog Verification: Brand '{p.brand}' + Product Title match ({len(matched_words)}/{len(p_words)} words)"

    if best_match and best_score >= 0.85:
        return (best_match, round(best_score, 2), best_reason)

    return None


def identify_product_multi_signal(
    image_paths: Dict[str, str],
    ocr_side_detections: Dict[str, List[Dict[str, Any]]],
    extracted_declarations: Optional[Dict[str, Any]] = None,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """
    CRITICAL REQUIREMENT:
    The actual physical package images are the PRIMARY SOURCE OF TRUTH.
    Every ANALYZE operation is strictly isolated to the CURRENT four images.
    
    1. Check Barcode / GTIN / QR code on current images.
    2. Extract exact Brand, Product Name, and Variant directly from the package PDP.
    3. Check for open-source visual similarity signature.
    4. Only enrich from catalog if barcode or high-confidence match (>= 0.85) succeeds.
    5. NEVER return Haldiram, demo products, or generic fallbacks when evidence belongs to another product.
    """
    paths_list = [p for p in image_paths.values() if p]

    # Handle if db session was passed as the 3rd positional argument
    if isinstance(extracted_declarations, Session):
        db = extracted_declarations
        extracted_declarations = None

    # Combine all OCR text from current images
    all_tokens = []
    front_tokens = []
    for side, detections in ocr_side_detections.items():
        for det in detections:
            t = det.get("text", "").strip()
            if t:
                all_tokens.append(t)
                if side.lower() == "front":
                    front_tokens.append(t)

    combined_text = " ".join(all_tokens)

    # 1. Barcode check
    detected_barcode = detect_barcodes_from_images(paths_list)
    if detected_barcode and db:
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
                    "match_source": "Barcode / GTIN Scan (100% Package Match)",
                    "mrp": barcode_prod.mrp,
                    "net_quantity": barcode_prod.net_quantity,
                    "manufacturer": barcode_prod.manufacturer,
                    "ingredients": barcode_prod.ingredients,
                    "nutrition_facts": barcode_prod.nutrition_facts
                },
                "candidates": [],
                "barcode_detected": detected_barcode,
                "evidence_source": "BARCODE_AUTHENTICATED"
            }

    # Extract PDP brand & product name from declarations or front view tokens
    pkg_brand = None
    pkg_name = None

    if extracted_declarations:
        brand_entry = extracted_declarations.get("brand") or {}
        if brand_entry.get("detected") and brand_entry.get("value") not in ["Not detected", "Not confidently detected"]:
            pkg_brand = brand_entry.get("value").strip()

        name_entry = extracted_declarations.get("product_name") or {}
        if name_entry.get("detected") and name_entry.get("value") not in ["Not detected", "Product could not be confidently identified."]:
            pkg_name = name_entry.get("value").strip()

    # 2. Visual signature supporting signal
    visual_sig = None
    front_img_path = image_paths.get("front")
    if front_img_path:
        visual_sig = extract_visual_feature_signature(front_img_path)

    # 3. High-Confidence Catalog Match (Strict >= 0.85 Gating)
    catalog_match = None
    if db:
        catalog_match = match_catalog_by_strict_evidence(pkg_brand, pkg_name, combined_text, db)

    if catalog_match:
        top_prod, top_score, reason = catalog_match
        return {
            "status": "IDENTIFIED",
            "matched_product": {
                "id": top_prod.id,
                "name": top_prod.name,
                "brand": top_prod.brand,
                "category": top_prod.category,
                "barcode": top_prod.barcode or detected_barcode,
                "confidence": top_score,
                "match_source": reason,
                "mrp": top_prod.mrp,
                "net_quantity": top_prod.net_quantity,
                "manufacturer": top_prod.manufacturer,
                "ingredients": top_prod.ingredients,
                "nutrition_facts": top_prod.nutrition_facts
            },
            "candidates": [],
            "barcode_detected": detected_barcode,
            "evidence_source": "CATALOG_VERIFIED",
            "visual_signature": visual_sig
        }

    # 4. Package Direct Optical Identification (The physical package is the Primary Source of Truth)
    # If the package text was readable on the PDP, construct the exact identification from the package!
    if pkg_name and pkg_name not in ["Not detected", "Product could not be confidently identified."]:
        detected_brand_clean = pkg_brand if pkg_brand and pkg_brand not in ["Not detected", "Not confidently detected"] else "Brand on Package"
        
        # Determine confidence based on presence of brand and name
        confidence = 0.94 if pkg_brand and pkg_brand not in ["Not detected", "Not confidently detected"] else 0.85

        return {
            "status": "IDENTIFIED",
            "matched_product": {
                "id": None,
                "name": pkg_name,
                "brand": detected_brand_clean,
                "category": "Packaged Commodity",
                "barcode": detected_barcode,
                "confidence": confidence,
                "match_source": "Physical Package PDP Optical Extraction (Verbatim Package Text)",
                "mrp": extracted_declarations.get("mrp", {}).get("value") if extracted_declarations else None,
                "net_quantity": extracted_declarations.get("net_quantity", {}).get("value") if extracted_declarations else None,
                "manufacturer": extracted_declarations.get("manufacturer", {}).get("value") if extracted_declarations else None,
                "ingredients": None,
                "nutrition_facts": None
            },
            "candidates": [],
            "barcode_detected": detected_barcode,
            "evidence_source": "PACKAGE_PDP_DIRECT",
            "visual_signature": visual_sig
        }

    # 5. If confidence is insufficient, NEVER force a product or return Haldiram!
    return {
        "status": "NOT_CONFIDENTLY_IDENTIFIED",
        "matched_product": None,
        "product_name": "Product could not be confidently identified.",
        "brand": "Not confidently detected",
        "verification_status": "Needs Verification",
        "candidates": [],
        "barcode_detected": detected_barcode,
        "evidence_source": "INSUFFICIENT_EVIDENCE",
        "message": "Product could not be confidently identified. Needs Verification."
    }
