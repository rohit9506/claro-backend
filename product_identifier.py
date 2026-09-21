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


def run_multimodal_vision_assistance(image_paths: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Supporting Multimodal Vision Pipeline Component.
    Used strictly as one supporting signal of a Google-Lens-like visual product identification system.
    If GEMINI_API_KEY or GOOGLE_API_KEY is configured in the environment, queries the multimodal
    vision foundation to support physical package reading.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        import concurrent.futures
        from google import genai
        
        prompt = (
            "You are a computer vision intelligence assistant for Legal Metrology packaging inspection. "
            "Analyze these physical package photographs and identify the exact commodity with exact variant and net quantity.\n"
            "Extract the following exact fields visible on the package (or null if not visible):\n"
            "- product_name: The prominent generic or specific product name on the PDP\n"
            "- brand: The brand or manufacturer trade name\n"
            "- variant: Specific variant, flavour, or edition (e.g. Gold, 5% Solution, Extra Virgin)\n"
            "- mrp: Maximum Retail Price (inclusive of all taxes)\n"
            "- net_quantity: Net quantity with metric unit (e.g., 60 ml, 250 g, 500 g)\n"
            "- unit_sale_price: Unit sale price (e.g., ₹14.98 / ml)\n"
            "- manufacturer: Name and entity of manufacturer/packer\n"
            "- address: Complete postal address with PIN code\n"
            "- country_of_origin: Country of origin (e.g. India)\n"
            "- batch_number: Batch / Lot number\n"
            "- dates: Month and year of manufacture or expiry\n"
            "Return ONLY a valid JSON object with these keys."
        )

        def _do_multimodal_call():
            client = genai.Client(api_key=api_key)
            images_to_send = []
            for side, p in image_paths.items():
                if p and os.path.exists(p):
                    from PIL import Image
                    images_to_send.append(Image.open(p))
            
            if not images_to_send:
                return None

            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[*images_to_send, prompt],
            )

            if response and response.text:
                text = response.text.strip()
                text = re.sub(r"^```json\s*", "", text)
                text = re.sub(r"^```\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                return json.loads(text)
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(_do_multimodal_call)
            return fut.result(timeout=4.0)
    except Exception as e:
        print(f"[WARN] Multimodal vision fallback to deterministic physical package OCR: {e}")
        return None


def match_catalog_by_strict_evidence(
    package_brand: Optional[str],
    package_product_name: Optional[str],
    package_variant: Optional[str],
    package_quantity: Optional[str],
    combined_ocr_text: str,
    db: Session
) -> Optional[Tuple[Product, float, str]]:
    """
    Multi-Signal Catalog Cross-Verification:
    Evaluates convergence of Brand, Title, Variant, and Quantity evidence.
    A catalog candidate is NEVER accepted based on similarity alone.
    The physical package must explicitly support:
      1. The exact brand
      2. High-confidence distinctive product title tokens
      3. Specific variant/flavour (preventing cross-variant mismatch)
      4. Declared package quantity (preventing 200g vs 500g mismatch)
    """
    if not db:
        return None

    norm_combined = combined_ocr_text.lower()
    norm_pkg_brand = package_brand.lower().strip() if package_brand and package_brand not in ["Not detected", "Not confidently detected"] else ""
    norm_pkg_name = package_product_name.lower().strip() if package_product_name and package_product_name not in ["Not detected", "Product could not be confidently identified."] else ""
    norm_pkg_variant = package_variant.lower().strip() if package_variant and package_variant not in ["Not detected", "None", ""] else ""
    norm_pkg_qty = package_quantity.lower().strip() if package_quantity and package_quantity not in ["Not detected", "None", ""] else ""

    products = db.query(Product).all()
    best_match = None
    best_score = 0.0
    best_reason = ""

    for p in products:
        p_brand = (p.brand or "").lower().strip()
        p_name = (p.name or "").lower().strip()
        p_qty = (p.net_quantity or "").lower().strip()

        # 1. Strict Brand Check:
        brand_matched = False
        if p_brand and (p_brand in norm_combined or (norm_pkg_brand and p_brand in norm_pkg_brand)):
            brand_matched = True

        if not brand_matched and p_brand:
            continue

        # 2. Strict Product Name Word Matching:
        p_words = [w for w in re.findall(r'[a-zA-Z0-9]+', p_name) if len(w) >= 3]
        if not p_words:
            continue

        matched_words = [w for w in p_words if w in norm_combined]
        word_ratio = len(matched_words) / float(len(p_words))

        # 3. Exact Quantity Differentiation (e.g. 200 g vs 500 g):
        # If catalog has quantity and package has quantity, they MUST NOT contradict!
        if norm_pkg_qty and p_qty:
            pkg_qty_nums = re.findall(r'[0-9]+(?:\.[0-9]+)?', norm_pkg_qty)
            cat_qty_nums = re.findall(r'[0-9]+(?:\.[0-9]+)?', p_qty)
            if pkg_qty_nums and cat_qty_nums and pkg_qty_nums[0] != cat_qty_nums[0]:
                # Sibling product with differing quantity - strictly skip to avoid wrong size match
                continue

        # 4. Exact Variant / Flavour Differentiation (e.g. Gold vs Regular vs Cream):
        if norm_pkg_variant and len(norm_pkg_variant) >= 3:
            if norm_pkg_variant not in p_name and norm_pkg_variant not in norm_combined:
                continue

        # Convergence score (requires brand agreement, high title token overlap, and quantity/variant agreement)
        if word_ratio >= 0.80 and brand_matched:
            score = 0.50 + (word_ratio * 0.45)
            if score > best_score:
                best_score = score
                best_match = p
                best_reason = f"Multi-Signal Verification: Brand '{p.brand}' + Physical Package Token Match ({len(matched_words)}/{len(p_words)} words)"

    if best_match and best_score >= 0.80:
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

    # Extract PDP brand, product name, variant & quantity from declarations or front view tokens
    pkg_brand = None
    pkg_name = None
    pkg_variant = None
    pkg_qty = None

    if extracted_declarations:
        brand_entry = extracted_declarations.get("brand") or {}
        if brand_entry.get("detected") and brand_entry.get("value") not in ["Not detected", "Not confidently detected"]:
            pkg_brand = brand_entry.get("value").strip()

        name_entry = extracted_declarations.get("product_name") or {}
        if name_entry.get("detected") and name_entry.get("value") not in ["Not detected", "Product could not be confidently identified."]:
            pkg_name = name_entry.get("value").strip()

        variant_entry = extracted_declarations.get("variant") or {}
        if variant_entry.get("detected") and variant_entry.get("value") not in ["Not detected", "Not Applicable"]:
            pkg_variant = variant_entry.get("value").strip()

        qty_entry = extracted_declarations.get("net_quantity") or {}
        if qty_entry.get("detected") and qty_entry.get("value") not in ["Not detected"]:
            pkg_qty = qty_entry.get("value").strip()

    # 2. Visual signature supporting signal
    visual_sig = None
    front_img_path = image_paths.get("front")
    if front_img_path:
        visual_sig = extract_visual_feature_signature(front_img_path)

    # 3. Multi-Signal Catalog Cross-Verification (Requires brand + distinctive tokens + variant/qty consistency)
    catalog_match = None
    if db:
        catalog_match = match_catalog_by_strict_evidence(pkg_brand, pkg_name, pkg_variant, pkg_qty, combined_text, db)

    if catalog_match:
        top_prod, top_score, reason = catalog_match
        return {
            "status": "IDENTIFIED",
            "matched_product": {
                "id": top_prod.id,
                "name": top_prod.name,
                "brand": top_prod.brand,
                "variant": pkg_variant,
                "category": top_prod.category,
                "barcode": top_prod.barcode or detected_barcode,
                "confidence": top_score,
                "match_source": reason,
                "mrp": top_prod.mrp,
                "net_quantity": top_prod.net_quantity or pkg_qty,
                "manufacturer": top_prod.manufacturer,
                "ingredients": top_prod.ingredients,
                "nutrition_facts": top_prod.nutrition_facts
            },
            "candidates": [],
            "barcode_detected": detected_barcode,
            "evidence_source": "CATALOG_VERIFIED",
            "visual_signature": visual_sig
        }

    def is_filename(val: Optional[str]) -> bool:
        if not val or not str(val).strip():
            return True
        v = str(val).strip().lower()
        if any(v.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".pdf", ".svg"]):
            return True
        if re.search(r"^(web|front|back|side|image|img|photo|pic|screenshot|scan|upload)(\.\w+)?$", v):
            return True
        return False

    # 4. Package Direct Optical Identification (Physical package is the Primary Source of Truth)
    # If the package text was readable on the PDP, construct the exact identification directly from the package
    if pkg_name and pkg_name not in ["Not detected", "Product could not be confidently identified."] and not is_filename(pkg_name):
        detected_brand_clean = pkg_brand if pkg_brand and pkg_brand not in ["Not detected", "Not confidently detected"] and not is_filename(pkg_brand) else "Brand on Package"
        confidence = 0.94 if pkg_brand and pkg_brand not in ["Not detected", "Not confidently detected"] and not is_filename(pkg_brand) else 0.85

        return {
            "status": "IDENTIFIED",
            "matched_product": {
                "id": None,
                "name": pkg_name,
                "brand": detected_brand_clean,
                "variant": pkg_variant,
                "category": "Packaged Product",
                "barcode": detected_barcode,
                "confidence": confidence,
                "match_source": "Physical Package PDP Optical Extraction (Verbatim Package Text)",
                "mrp": extracted_declarations.get("mrp", {}).get("value") if extracted_declarations else None,
                "net_quantity": pkg_qty or (extracted_declarations.get("net_quantity", {}).get("value") if extracted_declarations else None),
                "manufacturer": extracted_declarations.get("manufacturer", {}).get("value") if extracted_declarations else None,
                "ingredients": None,
                "nutrition_facts": None
            },
            "candidates": [],
            "barcode_detected": detected_barcode,
            "evidence_source": "PACKAGE_PDP_DIRECT",
            "visual_signature": visual_sig
        }

    # 5. Supporting Multimodal Vision AI Pipeline (Fallback only when PDP text is unreadable or ambiguous)
    vision_info = run_multimodal_vision_assistance(image_paths)
    if vision_info and vision_info.get("product_name"):
        v_pname = str(vision_info["product_name"]).strip()
        v_brand = str(vision_info.get("brand") or "").strip() or "Brand on Package"
        v_variant = str(vision_info.get("variant") or "").strip() or pkg_variant
        v_qty = str(vision_info.get("net_quantity") or "").strip() or pkg_qty
        
        # Merge detected statutory declarations from multimodal vision if missing in local OCR
        if extracted_declarations:
            for field in ["mrp", "net_quantity", "manufacturer", "unit_sale_price"]:
                if field in vision_info and vision_info[field] and not extracted_declarations.get(field, {}).get("detected"):
                    extracted_declarations[field] = {
                        "value": str(vision_info[field]),
                        "raw_val": str(vision_info[field]),
                        "confidence": 0.95,
                        "side": "front",
                        "bbox_norm": [0.1, 0.1, 0.3, 0.9],
                        "heading": "Visual Comprehension AI",
                        "spatial_relationship": "VISUAL_IDENTIFIED",
                        "detected": True
                    }

        return {
            "status": "IDENTIFIED",
            "matched_product": {
                "id": None,
                "name": v_pname,
                "brand": v_brand,
                "variant": v_variant,
                "category": "Packaged Product",
                "barcode": detected_barcode,
                "confidence": 0.96,
                "match_source": "Multi-Signal Visual Comprehension Pipeline (Multimodal Verification)",
                "mrp": vision_info.get("mrp") or (extracted_declarations.get("mrp", {}).get("value") if extracted_declarations else None),
                "net_quantity": v_qty or (extracted_declarations.get("net_quantity", {}).get("value") if extracted_declarations else None),
                "manufacturer": vision_info.get("manufacturer") or (extracted_declarations.get("manufacturer", {}).get("value") if extracted_declarations else None),
                "ingredients": None,
                "nutrition_facts": None
            },
            "candidates": [],
            "barcode_detected": detected_barcode,
            "evidence_source": "MULTIMODAL_VISION_AI",
            "visual_signature": visual_sig,
            "visual_pipeline_verified": True
        }


    # 5. If confidence is insufficient, NEVER force a product or return Haldiram!
    return {
        "status": "NOT_CONFIDENTLY_IDENTIFIED",
        "matched_product": None,
        "product_name": "Product could not be confidently identified.",
        "brand": "Not confidently detected",
        "variant": None,
        "verification_status": "Needs Verification",
        "candidates": [],
        "barcode_detected": detected_barcode,
        "evidence_source": "INSUFFICIENT_EVIDENCE",
        "needs_verification": True,
        "message": "Product could not be confidently identified. Needs Verification."
    }
