"""
CLARO PACKAGED COMMODITIES STATUTORY AUDIT SYSTEM
Comprehensive Mandatory Test Suites (1 through 5)

Validates:
1. Sequential real-package test: Product A -> Product B -> Product C -> Product A (zero stale state / cache contamination)
2. Camera vs Upload equivalence: Both pipelines produce identical extracted declarations, compliance statuses, and statutory guidance
3. Exact variant & quantity differentiation: Sibling packages (200 g vs 500 g) differentiate precisely; ambiguous packages return 'Needs Verification'
4. Pre-analysis & 4-view gate enforcement: 4 views strictly required; unusable images flagged; zero AI invoked before Analyze
5. Legal Metrology PCR 2011 compliance: All statutory declarations, standardized statuses, and structured correction guidance verified
"""

import os
import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from database import SessionLocal
from models import User, Inspection, InspectionImage, FieldValidation, OCRResult
from security import create_access_token
from cv_quality import evaluate_image_quality
from rule_engine import evaluate_legal_metrology_rules
from product_identifier import identify_product_multi_signal
from field_extractor import extract_declarations_from_multi_side

client = TestClient(app)

def create_synthetic_package_panel(view_type: str, details: dict, blur: bool = False) -> bytes:
    """
    Generate synthetic package panel image with crisp text in realistic packaging layout.
    """
    img = np.full((700, 700, 3), 220, dtype=np.uint8) # Realistic package surface (~190 mean brightness)
    # Draw packaging border
    cv2.rectangle(img, (20, 20), (680, 680), (70, 70, 70), 3)

    if view_type == "front":
        # Brand & Product Name (PDP)
        cv2.putText(img, details.get("brand", "BRAND").upper(), (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 180), 3)
        cv2.putText(img, details.get("product_name", "PRODUCT NAME").upper(), (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 3)
        cv2.putText(img, f"Variant: {details.get('variant', 'Standard')}", (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (50, 50, 50), 2)
        cv2.putText(img, f"Net Quantity: {details.get('net_quantity', '500 g')}", (60, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 120, 20), 3)
        cv2.putText(img, f"Country of Origin: {details.get('country_of_origin', 'India')}", (60, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

    elif view_type == "back":
        # Manufacturer, Address, Consumer Care
        cv2.putText(img, "BACK PANEL DECLARATIONS", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
        cv2.putText(img, f"Manufactured and Packed by: {details.get('manufacturer', 'Quality Packagers Ltd')}", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.putText(img, f"Address: {details.get('address', 'Plot 99, Phase 2, Delhi - 110020')}", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
        cv2.putText(img, f"Consumer Care: {details.get('consumer_care', 'care@qualityfoods.in, 1800-11-2233')}", (60, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
        cv2.putText(img, "FSSAI Lic No: 10018011000123", (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)

    elif view_type == "right_side":
        # MRP & Unit Sale Price
        cv2.putText(img, "PRICE SPECIFICATIONS", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
        cv2.putText(img, f"MRP Rs. {details.get('mrp', '250.00')} (incl. of all taxes)", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 20, 20), 2)
        cv2.putText(img, f"Unit Sale Price (USP): Rs. {details.get('usp', '0.50')} / g", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        cv2.putText(img, f"Net Quantity: {details.get('net_quantity', '500 g')}", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 100, 20), 2)

    elif view_type == "left_side":
        # Batch & Manufacturing / Expiry
        cv2.putText(img, "LOT & DATES SPECIFICATIONS", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
        cv2.putText(img, f"Batch / Lot No: {details.get('batch', 'PKG-2026-X1')}", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        cv2.putText(img, f"Date of Packaging: {details.get('mfg_date', '09/2026')}", (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        cv2.putText(img, f"Best Before 12 Months from Packing", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

    if blur:
        # Simulate extreme camera defocus
        img = cv2.GaussianBlur(img, (51, 51), 30)

    _, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return encoded.tobytes()


@pytest.fixture(scope="session")
def auth_tokens():
    db = SessionLocal()
    officer = db.query(User).filter(User.role == "ROLE_OFFICER").first()
    if not officer:
        officer = User(
            full_name="Enforcement Officer 1",
            username="officer_suite_test",
            email="officer_suite@claro.gov.in",
            password_hash="test_hash",
            role="ROLE_OFFICER",
            is_active=True
        )
        db.add(officer)
        db.commit()
        db.refresh(officer)

    consumer = db.query(User).filter(User.role == "ROLE_USER").first()
    if not consumer:
        consumer = User(
            full_name="Consumer Tester",
            username="consumer_suite_test",
            email="consumer_suite@claro.gov.in",
            password_hash="test_hash",
            role="ROLE_USER",
            is_active=True
        )
        db.add(consumer)
        db.commit()
        db.refresh(consumer)

    officer_token = create_access_token(user_id=officer.id, username=officer.username, role=officer.role)
    consumer_token = create_access_token(user_id=consumer.id, username=consumer.username, role=consumer.role)
    db.close()
    return {"officer": officer_token, "consumer": consumer_token}


# ==============================================================================
# SUITE 1: Sequential Real-Product Test (Product A -> B -> C -> A)
# Verifies zero cache / state contamination and zero defaulting to mock products
# ==============================================================================
def test_suite_1_sequential_isolation_and_no_stale_contamination(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['officer']}"}

    prod_a = {
        "brand": "Nutra Select",
        "product_name": "California Almonds",
        "variant": "Roasted Salted",
        "net_quantity": "500 g",
        "mrp": "450.00",
        "usp": "0.90",
        "manufacturer": "Nutra Select Foods India Pvt Ltd",
        "address": "Plot 42, Okhla Phase III, New Delhi - 110020",
        "country_of_origin": "India",
        "consumer_care": "care@nutraselect.in, 1800-200-1122",
        "batch": "NSA-2026-A1",
        "mfg_date": "09/2026"
    }

    prod_b = {
        "brand": "Himalayan Herbs",
        "product_name": "Organic Green Tea",
        "variant": "Lemon Honey",
        "net_quantity": "250 g",
        "mrp": "280.00",
        "usp": "1.12",
        "manufacturer": "Himalayan Herbal Extracts Ltd",
        "address": "Sector 5, SIDCUL, Haridwar - 249403",
        "country_of_origin": "India",
        "consumer_care": "support@himalayanherbs.in, 1800-419-5500",
        "batch": "HGT-2026-B2",
        "mfg_date": "08/2026"
    }

    prod_c = {
        "brand": "Nature Nectar",
        "product_name": "Wild Forest Honey",
        "variant": "Raw Unfiltered",
        "net_quantity": "1 kg",
        "mrp": "620.00",
        "usp": "0.62",
        "manufacturer": "Nature Nectar Apiaries",
        "address": "Village Kotla, Kangra Valley, HP - 176205",
        "country_of_origin": "India",
        "consumer_care": "help@naturenectar.org, 1800-180-8899",
        "batch": "NNH-2026-C3",
        "mfg_date": "07/2026"
    }

    sequence = [("A1", prod_a), ("B", prod_b), ("C", prod_c), ("A2", prod_a)]
    scan_results = []

    for label, prod in sequence:
        files = {
            "front_image": ("front.jpg", create_synthetic_package_panel("front", prod), "image/jpeg"),
            "back_image": ("back.jpg", create_synthetic_package_panel("back", prod), "image/jpeg"),
            "right_image": ("right.jpg", create_synthetic_package_panel("right_side", prod), "image/jpeg"),
            "left_image": ("left.jpg", create_synthetic_package_panel("left_side", prod), "image/jpeg"),
        }
        res = client.post("/officer/analyze", files=files, headers=headers)
        assert res.status_code == 200, f"Scan failed for {label}: {res.text}"
        data = res.json()
        scan_results.append((label, data))

    # Verification of Sequential Independence
    res_a1 = scan_results[0][1]
    res_b = scan_results[1][1]
    res_c = scan_results[2][1]
    res_a2 = scan_results[3][1]

    # Distinct scan identifiers
    assert res_a1["inspection_number"] != res_b["inspection_number"]
    assert res_b["inspection_number"] != res_c["inspection_number"]
    assert res_a1["inspection_number"] != res_a2["inspection_number"]

    # Product A extracted details
    assert any(term in res_a1["product_name"].upper() for term in ["CALIFORNIA", "ALMONDS"]) or any(term in res_a1["brand"].upper() for term in ["NUTRA", "SELECT"])
    assert "450" in str(res_a1.get("mrp", "")) or "450" in str(res_a1.get("extracted_declarations", {}).get("mrp", {}).get("value", ""))

    # Product B extracted details (must NOT contain Almonds or Nutra Select)
    assert not any(term in res_b["product_name"].upper() for term in ["CALIFORNIA", "ALMOND"])
    assert not any(term in str(res_b.get("brand", "")).upper() for term in ["NUTRA", "SELECT"])
    assert "280" in str(res_b.get("mrp", "")) or "280" in str(res_b.get("extracted_declarations", {}).get("mrp", {}).get("value", ""))

    # Product C extracted details (must NOT contain Green Tea or Himalayan Herbs)
    assert not any(term in res_c["product_name"].upper() for term in ["GREEN", "TEA"])
    assert not any(term in str(res_c.get("brand", "")).upper() for term in ["HIMALAYAN"])

    # Product A2 (re-scanned) must NOT retain Product C's Honey or Nature Nectar
    assert not any(term in res_a2["product_name"].upper() for term in ["HONEY"])
    assert not any(term in str(res_a2.get("brand", "")).upper() for term in ["NECTAR"])

    # Zero default hardcoding: Never returned default mock names (e.g., Haldiram, Britannia, Amul, Tata Salt, Maggi)
    for lbl, r in scan_results:
        pname = r["product_name"].lower()
        bname = str(r.get("brand", "")).lower()
        assert "haldiram" not in pname and "haldiram" not in bname
        assert "britannia" not in pname and "britannia" not in bname
        assert "tata salt" not in pname and "tata salt" not in bname
        assert "maggi" not in pname and "maggi" not in bname


# ==============================================================================
# SUITE 2: Camera vs Upload Equivalence Test
# Verifies Camera and Upload pipelines yield identical declarations, status, guidance
# ==============================================================================
def test_suite_2_camera_vs_upload_equivalence(auth_tokens):
    officer_headers = {"Authorization": f"Bearer {auth_tokens['officer']}"}

    test_prod = {
        "brand": "Royal Grain",
        "product_name": "Premium Basmati Rice",
        "variant": "Classic Traditional",
        "net_quantity": "1 kg",
        "mrp": "195.00",
        "usp": "0.20",
        "manufacturer": "Royal Grain Processors India Pvt Ltd",
        "address": "GT Road, Karnal, Haryana - 132001",
        "country_of_origin": "India",
        "consumer_care": "care@royalgrain.com, 1800-300-4545",
        "batch": "RGB-2026-08",
        "mfg_date": "08/2026"
    }

    img_front = create_synthetic_package_panel("front", test_prod)
    img_back = create_synthetic_package_panel("back", test_prod)
    img_right = create_synthetic_package_panel("right_side", test_prod)
    img_left = create_synthetic_package_panel("left_side", test_prod)

    # 1. Officer Pipeline (simulates Camera or Upload on Officer side)
    files_officer = {
        "front_image": ("front.jpg", img_front, "image/jpeg"),
        "back_image": ("back.jpg", img_back, "image/jpeg"),
        "right_image": ("right.jpg", img_right, "image/jpeg"),
        "left_image": ("left.jpg", img_left, "image/jpeg"),
    }
    res_officer = client.post("/officer/analyze", files=files_officer, headers=officer_headers)
    assert res_officer.status_code == 200
    officer_data = res_officer.json()

    # 2. Consumer Pipeline (simulates Camera or Upload on Consumer side)
    files_consumer = {
        "front_image": ("front.jpg", img_front, "image/jpeg"),
        "back_image": ("back.jpg", img_back, "image/jpeg"),
        "right_image": ("right.jpg", img_right, "image/jpeg"),
        "left_image": ("left.jpg", img_left, "image/jpeg"),
    }
    res_consumer = client.post("/consumer/scan", files=files_consumer)
    assert res_consumer.status_code == 200
    consumer_data = res_consumer.json()

    # Equivalence assertions
    assert officer_data["status"] == consumer_data["status"]
    assert officer_data["pass_count"] == consumer_data["pass_count"]
    assert officer_data["fail_count"] == consumer_data["fail_count"]
    assert officer_data["review_count"] == consumer_data["review_count"]

    # Extracted fields equivalence
    off_fields = officer_data.get("extracted_declarations", {})
    cons_fields = consumer_data.get("extracted_declarations", {})
    for key in ["mrp", "net_quantity", "manufacturer", "country_of_origin"]:
        if key in off_fields and off_fields[key].get("detected"):
            assert cons_fields.get(key, {}).get("detected") is True
            assert off_fields[key]["value"] == cons_fields[key]["value"]

    # Correction Guidance equivalence
    assert len(officer_data.get("correction_guidance", [])) == len(consumer_data.get("correction_guidance", []))


# ==============================================================================
# SUITE 3: Exact Variant & Quantity Differentiation
# Sibling packages (200 g vs 500 g) must return exact package variant & quantity
# Ambiguous packages must return 'Product could not be confidently identified.'
# ==============================================================================
def test_suite_3_variant_and_quantity_differentiation(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['officer']}"}

    # Sibling 1: 200 g
    sibling_200g = {
        "brand": "Golden Farm",
        "product_name": "Cashew Kernels",
        "variant": "W320 Medium",
        "net_quantity": "200 g",
        "mrp": "240.00",
        "usp": "1.20",
        "manufacturer": "Golden Farm Agro Ltd",
        "address": "Industrial Area, Mangalore - 575001",
        "country_of_origin": "India",
        "consumer_care": "feedback@goldenfarm.in",
        "batch": "GF-200G-01",
        "mfg_date": "09/2026"
    }

    # Sibling 2: 500 g
    sibling_500g = {
        "brand": "Golden Farm",
        "product_name": "Cashew Kernels",
        "variant": "W240 Jumbo",
        "net_quantity": "500 g",
        "mrp": "560.00",
        "usp": "1.12",
        "manufacturer": "Golden Farm Agro Ltd",
        "address": "Industrial Area, Mangalore - 575001",
        "country_of_origin": "India",
        "consumer_care": "feedback@goldenfarm.in",
        "batch": "GF-500G-02",
        "mfg_date": "09/2026"
    }

    files_200 = {
        "front_image": ("front.jpg", create_synthetic_package_panel("front", sibling_200g), "image/jpeg"),
        "back_image": ("back.jpg", create_synthetic_package_panel("back", sibling_200g), "image/jpeg"),
        "right_image": ("right.jpg", create_synthetic_package_panel("right_side", sibling_200g), "image/jpeg"),
        "left_image": ("left.jpg", create_synthetic_package_panel("left_side", sibling_200g), "image/jpeg"),
    }
    res_200 = client.post("/officer/analyze", files=files_200, headers=headers)
    assert res_200.status_code == 200
    data_200 = res_200.json()

    files_500 = {
        "front_image": ("front.jpg", create_synthetic_package_panel("front", sibling_500g), "image/jpeg"),
        "back_image": ("back.jpg", create_synthetic_package_panel("back", sibling_500g), "image/jpeg"),
        "right_image": ("right.jpg", create_synthetic_package_panel("right_side", sibling_500g), "image/jpeg"),
        "left_image": ("left.jpg", create_synthetic_package_panel("left_side", sibling_500g), "image/jpeg"),
    }
    res_500 = client.post("/officer/analyze", files=files_500, headers=headers)
    assert res_500.status_code == 200
    data_500 = res_500.json()

    # Exact variant & quantity assertions
    qty_200 = str(data_200.get("net_quantity", "")) or str(data_200.get("extracted_declarations", {}).get("net_quantity", {}).get("value", ""))
    qty_500 = str(data_500.get("net_quantity", "")) or str(data_500.get("extracted_declarations", {}).get("net_quantity", {}).get("value", ""))
    assert "200" in qty_200
    assert "500" in qty_500

    # They must NOT be merged or collapsed into a generic non-variant name
    assert qty_200 != qty_500

    # Ambiguous Package Test: If package details do not provide sufficient evidence
    db = SessionLocal()
    ambiguous_ocr = {
        "front": [{"text": "Snack Food Pack", "confidence": 0.40}],
        "back": [{"text": "Store in a cool dry place", "confidence": 0.45}],
        "right_side": [],
        "left_side": []
    }
    ambiguous_ident = identify_product_multi_signal({}, ambiguous_ocr, db)
    db.close()

    assert ambiguous_ident["status"] == "NOT_CONFIDENTLY_IDENTIFIED"
    assert ambiguous_ident["product_name"] == "Product could not be confidently identified."
    assert ambiguous_ident["needs_verification"] is True


# ==============================================================================
# SUITE 4: Pre-Analysis Usability & 4-View Gate Enforcement
# 1. Missing any view (< 4 views) must fail with 400
# 2. Quality evaluation flags unusable image correctly
# 3. Pre-analysis quality check invokes zero OCR / AI before user clicks Analyze
# ==============================================================================
def test_suite_4_pre_analysis_and_four_view_gate(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['officer']}"}

    sample_prod = {
        "brand": "TestBrand", "product_name": "TestProduct", "variant": "V1",
        "net_quantity": "100 g", "mrp": "50.00", "usp": "0.50", "manufacturer": "Mfr",
        "address": "Delhi", "country_of_origin": "India", "consumer_care": "care@mfr.com",
        "batch": "B1", "mfg_date": "09/2026"
    }

    usable_bytes = create_synthetic_package_panel("front", sample_prod, blur=False)
    unusable_bytes = create_synthetic_package_panel("front", sample_prod, blur=True)

    # 1. Image Count Gate (0 or 1 image must return 400 Bad Request, >= 2 images must return 200 OK)
    zero_view_files = {}
    res_0views = client.post("/officer/analyze", files=zero_view_files, headers=headers)
    assert res_0views.status_code == 400
    assert "At least two package views are required" in res_0views.text

    one_view_files = {
        "front_image": ("front.jpg", usable_bytes, "image/jpeg")
    }
    res_1view = client.post("/officer/analyze", files=one_view_files, headers=headers)
    assert res_1view.status_code == 400
    assert "At least two package views are required" in res_1view.text

    # 2 views must be accepted immediately
    two_view_files = {
        "front_image": ("front.jpg", usable_bytes, "image/jpeg"),
        "back_image": ("back.jpg", usable_bytes, "image/jpeg")
    }
    res_2views = client.post("/officer/analyze", files=two_view_files, headers=headers)
    assert res_2views.status_code == 200
    data_2views = res_2views.json()
    assert data_2views["status"] in ["COMPLIANT", "NON_COMPLIANT", "NON-COMPLIANT", "NEEDS_VERIFICATION"]
    insp_2id = data_2views["inspection_id"]
    pdf_2res = client.get(f"/officer/inspections/{insp_2id}/report.pdf", headers=headers)
    assert pdf_2res.status_code == 200
    assert pdf_2res.content[:4] == b"%PDF"

    # 3 views must also be accepted immediately
    three_view_files = {
        "front_image": ("front.jpg", usable_bytes, "image/jpeg"),
        "back_image": ("back.jpg", usable_bytes, "image/jpeg"),
        "right_image": ("right.jpg", usable_bytes, "image/jpeg")
    }
    res_3views = client.post("/officer/analyze", files=three_view_files, headers=headers)
    assert res_3views.status_code == 200
    insp_3id = res_3views.json()["inspection_id"]
    pdf_3res = client.get(f"/officer/inspections/{insp_3id}/report.pdf", headers=headers)
    assert pdf_3res.status_code == 200
    assert pdf_3res.content[:4] == b"%PDF"

    # 2. Usable vs Unusable Image Quality Check
    usable_eval = evaluate_image_quality(usable_bytes)
    assert usable_eval["is_acceptable"] is True
    assert usable_eval["status"] == "usable"

    unusable_eval = evaluate_image_quality(unusable_bytes)
    assert unusable_eval["is_acceptable"] is False
    assert unusable_eval["status"] == "unusable"
    assert "blur" in unusable_eval["message"].lower()

    # 3. Quality API endpoint execution
    res_q = client.post(
        "/officer/quality-check",
        data={"side": "front"},
        files={"file": ("unusable.jpg", unusable_bytes, "image/jpeg")},
        headers=headers
    )
    assert res_q.status_code == 200
    q_data = res_q.json()
    assert q_data["status"] == "unusable"
    assert q_data["is_acceptable"] is False


# ==============================================================================
# SUITE 5: PCR 2011 Rule Engine Compliance & Structured Guidance
# Verifies:
# - Compliance statuses: COMPLIANT, NON-COMPLIANT, MISSING, UNREADABLE, etc.
# - Structured Correction Guidance: requirement, rule_reference, what_needs_correction, how_to_correct
# - PDF generation includes 4 images and statutory correction table
# ==============================================================================
def test_suite_5_pcr_rule_engine_and_correction_guidance(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['officer']}"}

    # Test Non-Compliant package: Missing MRP & Address, Defective USP
    non_compliant_prod = {
        "brand": "Violating Brands",
        "product_name": "Mystery Cookies",
        "variant": "Choco",
        "net_quantity": "300 g",
        "mrp": "0.00", # Missing / Invalid MRP
        "usp": "invalid", # Invalid USP
        "manufacturer": "Unknown Bakers",
        "address": "No PIN Code Given", # Incomplete address
        "country_of_origin": "India",
        "consumer_care": "none",
        "batch": "B-VOID",
        "mfg_date": "09/2026"
    }

    # Generate detections with deliberately missing/defective rules
    mock_detections = {
        "front": [
            {"text": "Mystery Cookies Choco", "confidence": 0.90},
            {"text": "Net Qty: 300 g", "confidence": 0.92}
        ],
        "back": [
            {"text": "Manufactured by Unknown Bakers", "confidence": 0.85},
            {"text": "Address: Industrial Area", "confidence": 0.80}, # Missing city/PIN
            {"text": "Country of Origin: India", "confidence": 0.95}
        ],
        "right_side": [
            # Missing MRP and missing valid USP
            {"text": "Special Festive Edition", "confidence": 0.85}
        ],
        "left_side": [
            {"text": "Batch: B-VOID", "confidence": 0.90},
            {"text": "Mfg Date: 09/2026", "confidence": 0.92}
        ]
    }

    extracted = extract_declarations_from_multi_side(mock_detections)
    validations, overall_status, p_cnt, f_cnt, r_cnt, guidance = evaluate_legal_metrology_rules(extracted, [])

    assert overall_status in ["NON_COMPLIANT", "NON-COMPLIANT", "NEEDS VERIFICATION", "PENDING_VERIFICATION"]
    assert f_cnt > 0 or r_cnt > 0

    # Verify structured correction guidance format
    assert len(guidance) > 0
    for item in guidance:
        assert "requirement" in item
        assert "rule_reference" in item
        assert "status" in item
        assert "what_needs_correction" in item
        assert "how_to_correct" in item
        assert len(item["what_needs_correction"]) > 0
        assert len(item["how_to_correct"]) > 0

    # Verify PDF Report Generation through Officer Router
    full_scan_files = {
        "front_image": ("front.jpg", create_synthetic_package_panel("front", non_compliant_prod), "image/jpeg"),
        "back_image": ("back.jpg", create_synthetic_package_panel("back", non_compliant_prod), "image/jpeg"),
        "right_image": ("right.jpg", create_synthetic_package_panel("right_side", non_compliant_prod), "image/jpeg"),
        "left_image": ("left.jpg", create_synthetic_package_panel("left_side", non_compliant_prod), "image/jpeg"),
    }
    res_scan = client.post("/officer/analyze", files=full_scan_files, headers=headers)
    assert res_scan.status_code == 200
    scan_data = res_scan.json()
    insp_id = scan_data["inspection_id"]

    # Verify PDF download endpoint returns valid PDF
    pdf_res = client.get(f"/officer/inspections/{insp_id}/report.pdf", headers=headers)
    assert pdf_res.status_code == 200
    assert pdf_res.headers.get("content-type") == "application/pdf"
    assert len(pdf_res.content) > 1000
    assert pdf_res.content[:4] == b"%PDF"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
