import pytest
import io
import json
import numpy as np
import cv2
from fastapi.testclient import TestClient

import datetime
from main import app
from database import SessionLocal
from models import User, Rule, Inspection
from cv_quality import evaluate_image_quality
from rule_engine import evaluate_legal_metrology_rules
from report_generator import generate_inspection_pdf
from security import evaluate_password_strength, validate_email_address, create_access_token

client = TestClient(app)

def test_health_and_root():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    root_res = client.get("/")
    assert root_res.status_code == 200
    assert root_res.json()["platform"] == "CLARO"

def test_password_strength_policy():
    # Weak password
    weak = evaluate_password_strength("short123")
    assert not weak["is_valid"]
    assert weak["level"] in ["WEAK", "FAIR"]

    # Common password
    common = evaluate_password_strength("password123456")
    assert not common["is_valid"]

    # Strong password
    strong = evaluate_password_strength("Claro@2026!MetrologyEnforce")
    assert strong["is_valid"]
    assert strong["level"] in ["STRONG", "VERY STRONG"]

def test_email_validation():
    valid, res = validate_email_address("officer.delhi@claro.gov.in")
    assert valid

    invalid, err = validate_email_address("not-an-email")
    assert not invalid

def test_user_registration_and_verification_flow():
    reg_payload = {
        "full_name": "Pooja Sharma",
        "username": "pooja_sharma",
        "email": "pooja.sharma@example.com",
        "password": "Consumer@2026!SecurePass",
        "confirm_password": "Consumer@2026!SecurePass",
        "mobile_number": "9876543210"
    }
    # Register directly without email verification
    res = client.post("/auth/register", json=reg_payload)
    if res.status_code == 400 and "already taken" in res.text:
        pass # Already created
    else:
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "Account created successfully" in data["message"]

    # Now login directly succeeds
    login_res = client.post("/auth/login", json={
        "username_or_email": "pooja_sharma",
        "password": "Consumer@2026!SecurePass"
    })
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["user"]["role"] == "ROLE_USER"

def test_rbac_enforcement():
    # Login as User
    login_res = client.post("/auth/login", json={
        "username_or_email": "pooja_sharma",
        "password": "Consumer@2026!SecurePass"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # User trying to access Admin dashboard should get 403 FORBIDDEN
    admin_res = client.get("/admin/dashboard-stats", headers=headers)
    assert admin_res.status_code == 403

    # User trying to access Officer stats should get 403 FORBIDDEN
    officer_res = client.get("/officer/stats", headers=headers)
    assert officer_res.status_code == 403

def test_image_quality_evaluator():
    # Create a dummy image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.putText(img, "CLARO TEST PACKAGE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode(".jpg", img)
    
    quality_res = evaluate_image_quality(buffer.tobytes())
    assert "is_acceptable" in quality_res
    assert "message" in quality_res

def test_rule_engine():
    mock_extracted = {
        "mrp": {"value": "₹120.00 (incl. of all taxes)", "confidence": 0.95, "detected": True, "side": "front"},
        "net_quantity": {"value": "500 g", "confidence": 0.94, "detected": True, "side": "front"},
        "unit_sale_price": {"value": "₹0.24 / g", "confidence": 0.88, "detected": True, "side": "front"},
        "manufacturer": {"value": "Haldiram Snacks Pvt. Ltd.", "confidence": 0.92, "detected": True, "side": "back"},
        "complete_address": {"value": "Sector 68, Noida - 201301", "confidence": 0.91, "detected": True, "side": "back"},
        "dates": {"value": "08/2026", "confidence": 0.90, "detected": True, "side": "back"},
        "consumer_care": {"value": "care@haldirams.com", "confidence": 0.93, "detected": True, "side": "back"},
        "country_of_origin": {"value": "India", "confidence": 0.95, "detected": True, "side": "front"}
    }
    validations, overall_status, pass_c, fail_c, review_c = evaluate_legal_metrology_rules(mock_extracted, [])
    assert overall_status == "COMPLIANT"
    assert pass_c >= 7
    assert fail_c == 0

def test_pdf_report_generation():
    pdf_path = generate_inspection_pdf({
        "inspection_number": "INSP-TEST-9999",
        "officer_name": "Test Officer",
        "officer_badge": "LM-TEST",
        "location": "Central Lab",
        "product_name": "Test Product",
        "created_at": "2026-09-18 12:00 UTC",
        "status": "COMPLIANT",
        "pass_count": 8,
        "fail_count": 0,
        "review_count": 0,
        "officer_notes": "Unit test verified.",
        "validations": []
    })
    import os
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000

def test_admin_credential_management():
    # 1. Create admin token
    admin_token = create_access_token(user_id=1, username="claro_admin", role="ROLE_ADMIN")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 2. Provision new officer with Auto-Generate
    officer_payload = {
        "full_name": "Test Officer Auto",
        "email": f"officer_auto_{datetime.datetime.now().timestamp()}@claro.gov.in",
        "officer_id": f"LM-AUTO-{int(datetime.datetime.now().timestamp())}",
        "credential_method": "GENERATE"
    }
    resp = client.post("/admin/officers", json=officer_payload, headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "credentials" in data
    assert data["credentials"]["temporary_password"] is not None
    assert len(data["credentials"]["temporary_password"]) >= 12
    officer_user_id = data["officer_id"]

    # 3. Check credentials list
    list_resp = client.get("/admin/credentials", headers=admin_headers)
    assert list_resp.status_code == 200
    creds_list = list_resp.json()
    assert any(acc["id"] == officer_user_id for acc in creds_list)

    # 4. Reset password using GENERATE method
    reset_resp = client.post(
        f"/admin/accounts/{officer_user_id}/reset-password",
        json={"method": "GENERATE"},
        headers=admin_headers
    )
    assert reset_resp.status_code == 200
    reset_data = reset_resp.json()
    assert reset_data["success"] is True
    assert reset_data["credentials"]["temporary_password"] is not None

    # 5. Reset password using MANUAL method
    manual_pwd = "Str0ng#ClaroLegal2026!"
    reset_man_resp = client.post(
        f"/admin/accounts/{officer_user_id}/reset-password",
        json={"method": "MANUAL", "new_password": manual_pwd, "confirm_password": manual_pwd},
        headers=admin_headers
    )
    assert reset_man_resp.status_code == 200
    assert reset_man_resp.json()["success"] is True

def test_product_identification_and_deletion():
    from product_identifier import identify_product_multi_signal
    from database import SessionLocal
    from models import Inspection, User

    db = SessionLocal()
    # Test multi-signal identification
    mock_ocr = {
        "front": [{"text": "Amul Pure Ghee 1L Tin", "confidence": 0.95}],
        "back": [{"text": "Manufactured by Gujarat Cooperative Milk Marketing Federation", "confidence": 0.92}]
    }
    ident_res = identify_product_multi_signal({}, mock_ocr, db)
    assert "status" in ident_res
    assert ident_res["status"] in ["IDENTIFIED", "POSSIBLE_MATCH", "NOT_CONFIDENTLY_IDENTIFIED"]

    # Test inspection creation and deletion
    officer = db.query(User).filter(User.role == "ROLE_OFFICER").first()
    if not officer:
        officer = db.query(User).first()

    test_insp = Inspection(
        inspection_number="INSP-DELETE-TEST-001",
        officer_id=officer.id,
        product_name="Temporary Inspection For Deletion Test",
        status="COMPLIANT",
        pass_count=5,
        fail_count=0,
        review_count=0
    )
    db.add(test_insp)
    db.commit()
    db.refresh(test_insp)
    test_id = test_insp.id

    # Create auth token for officer
    token = create_access_token(user_id=officer.id, username=officer.username, role=officer.role)
    headers = {"Authorization": f"Bearer {token}"}

    # Delete inspection
    del_res = client.delete(f"/officer/inspections/{test_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Verify deleted
    verify = db.query(Inspection).filter(Inspection.id == test_id).first()
def test_admin_delete_officer():
    import uuid
    uid = uuid.uuid4().hex[:6]
    db = SessionLocal()
    admin = db.query(User).filter(User.role == "ROLE_ADMIN").first()
    assert admin is not None
    admin_token = create_access_token(user_id=admin.id, username=admin.username, role=admin.role)
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Create a temporary officer
    temp_officer = User(
        full_name=f"Temp Officer {uid}",
        username=f"officer_del_{uid}",
        email=f"officer_{uid}@claro.gov.in",
        password_hash="testhash123456",
        role="ROLE_OFFICER",
        officer_id=f"DEL-{uid}",
        department="Test Enforcement"
    )
    db.add(temp_officer)
    db.commit()
    db.refresh(temp_officer)
    off_id = temp_officer.id

    # Create an inspection linked to this officer
    temp_insp = Inspection(
        inspection_number=f"INSP-OFF-DEL-{uid}",
        officer_id=off_id,
        product_name="Officer Cascade Inspection",
        status="COMPLIANT"
    )
    db.add(temp_insp)
    db.commit()
    db.refresh(temp_insp)
    insp_id = temp_insp.id

    # Call DELETE /admin/officers/{officer_id}
    del_res = client.delete(f"/admin/officers/{off_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Check officer is deleted
    db.expire_all()
    assert db.query(User).filter(User.id == off_id).first() is None
    # Check inspection was NOT deleted, but officer_id was safely nullified
    insp_after = db.query(Inspection).filter(Inspection.id == insp_id).first()
    assert insp_after is not None
    assert insp_after.officer_id is None

    # Cleanup inspection
    db.delete(insp_after)
    db.commit()
    db.close()

def test_admin_adjudicate_manual_text():
    import uuid
    uid = uuid.uuid4().hex[:6]
    db = SessionLocal()
    admin = db.query(User).filter(User.role == "ROLE_ADMIN").first()
    admin_token = create_access_token(user_id=admin.id, username=admin.username, role=admin.role)
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Create a pending inspection with a REVIEW field validation having empty / undetected text
    from models import FieldValidation
    insp = Inspection(
        inspection_number=f"INSP-PENDING-{uid}",
        product_name="Manual Entry Test Pack",
        status="PENDING_VERIFICATION",
        pass_count=0,
        fail_count=0,
        review_count=1
    )
    db.add(insp)
    db.commit()
    db.refresh(insp)

    fv = FieldValidation(
        inspection_id=insp.id,
        field_name="net_quantity",
        requirement_summary="Net quantity declaration required",
        rule_reference="Rule 12",
        status="REVIEW",
        detected_value="None",
        confidence=0.45,
        reason="Optical confidence low"
    )
    db.add(fv)
    db.commit()
    db.refresh(fv)

    # Admin manually types the detected text as "Net Wt: 500g" and confirms compliant
    adj_payload = {
        "field_validation_id": fv.id,
        "decision": "CONFIRM_COMPLIANT",
        "comment": "Admin manually verified text from packaging photo",
        "corrected_value": "Net Wt: 500g"
    }
    res = client.post(f"/admin/verifications/{insp.id}/adjudicate", json=adj_payload, headers=headers)
    assert res.status_code == 200
    assert res.json()["success"] is True

    # Verify field validation updated with manual value
    db.expire_all()
    fv_updated = db.query(FieldValidation).filter(FieldValidation.id == fv.id).first()
    assert fv_updated.status == "PASS"
    assert fv_updated.detected_value == "Net Wt: 500g"

    # Verify overall inspection moved to COMPLIANT
    insp_updated = db.query(Inspection).filter(Inspection.id == insp.id).first()
    assert insp_updated.status == "COMPLIANT"

    # Cleanup
    db.delete(fv_updated)
    db.delete(insp_updated)
    db.commit()
    db.close()


def test_inspection_report_preview_endpoint():
    # 1. Login as officer
    token = create_access_token(user_id=2, username="officer_verma", role="ROLE_OFFICER")
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Call preview on inspection 1
    res = client.get("/officer/inspections/1/report/preview", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["total_pages"] >= 1
    assert len(data["pages"]) >= 1
    assert data["pages"][0].startswith("data:image/jpeg;base64,")
    assert "pdf_url" in data


def test_strict_wrong_password_rejection():
    # Test that entering wrong password returns exact required message
    res = client.post("/auth/login", json={
        "username_or_email": "admin",
        "password": "WrongPassword123!"
    })
    assert res.status_code == 401
    assert res.json()["detail"] == "Invalid password. Please re-enter your password."


def test_field_extractor_2d_spatial_and_disambiguation():
    from field_extractor import extract_declarations_from_multi_side

    # Mock 4 package views detections with 2D spatial arrangement
    mock_detections = {
        "front": [
            {"text": "PREMIUM ROASTED ALMONDS", "confidence": 0.95, "bbox_norm": [0.20, 0.15, 0.30, 0.85]},
            {"text": "NUTRA SELECT", "confidence": 0.98, "bbox_norm": [0.08, 0.25, 0.16, 0.75]},
            # Spatial 2D: Net Quantity heading on line 1, numeric value on line 2 (BELOW)
            {"text": "Net Weight:", "confidence": 0.92, "bbox_norm": [0.70, 0.20, 0.74, 0.40]},
            {"text": "500 g", "confidence": 0.94, "bbox_norm": [0.75, 0.20, 0.80, 0.35]},
            # Disambiguation: Nutritional serving size should NOT be picked as net quantity
            {"text": "Serving Size: 30g (approx 28 pieces)", "confidence": 0.90, "bbox_norm": [0.85, 0.10, 0.89, 0.60]},
        ],
        "back": [
            # Disambiguation: Special offer should NOT be picked as MRP
            {"text": "Special Festive Offer: Save Rs. 50 on Combo!", "confidence": 0.93, "bbox_norm": [0.10, 0.10, 0.14, 0.90]},
            # Spatial 2D: MRP Heading on left, Price on right
            {"text": "MRP (incl. of all taxes):", "confidence": 0.96, "bbox_norm": [0.20, 0.10, 0.25, 0.45]},
            {"text": "Rs. 450.00", "confidence": 0.97, "bbox_norm": [0.20, 0.48, 0.25, 0.68]},
            # Unit Sale Price
            {"text": "Unit Sale Price: ₹0.90 / g", "confidence": 0.91, "bbox_norm": [0.26, 0.10, 0.30, 0.50]},
            # Manufacturer & Address
            {"text": "Manufactured and Packed by: Royal Agro Foods Pvt Ltd", "confidence": 0.94, "bbox_norm": [0.35, 0.10, 0.40, 0.90]},
            {"text": "Plot 42, Industrial Area, Phase II, New Delhi - 110020", "confidence": 0.93, "bbox_norm": [0.41, 0.10, 0.46, 0.90]},
            {"text": "Consumer Care: 1800-11-4040, email: care@royalagro.com", "confidence": 0.95, "bbox_norm": [0.50, 0.10, 0.55, 0.85]},
            {"text": "Country of Origin: India", "confidence": 0.96, "bbox_norm": [0.57, 0.10, 0.61, 0.45]},
        ],
        "right_side": [
            # Batch and Dates on right side panel
            {"text": "Batch No: RAF-2026-B8", "confidence": 0.92, "bbox_norm": [0.30, 0.10, 0.35, 0.60]},
            {"text": "Mfg Date: 09/2026", "confidence": 0.95, "bbox_norm": [0.40, 0.10, 0.45, 0.50]},
        ],
        "left_side": [
            {"text": "Best Before 12 Months from Packaging", "confidence": 0.89, "bbox_norm": [0.30, 0.10, 0.36, 0.80]}
        ]
    }

    results = extract_declarations_from_multi_side(mock_detections)

    # Assertions
    assert results["product_name"]["detected"] is True
    assert "ALMONDS" in results["product_name"]["value"]

    assert results["brand"]["detected"] is True
    assert results["brand"]["value"] == "NUTRA SELECT"

    # MRP was disambiguated from offer (Rs. 450, not Rs. 50)
    assert results["mrp"]["detected"] is True
    assert "450" in results["mrp"]["value"]
    assert results["mrp"]["spatial_relationship"] in ["RIGHT", "INLINE", "STICKER_BOX"]

    # Net quantity was disambiguated from serving size (500 g, not 30g)
    assert results["net_quantity"]["detected"] is True
    assert "500" in results["net_quantity"]["value"]
    assert results["net_quantity"]["spatial_relationship"] in ["BELOW", "INLINE"]

    assert results["unit_sale_price"]["detected"] is True
    assert "0.90" in results["unit_sale_price"]["value"]

    assert results["manufacturer"]["detected"] is True
    assert "Royal Agro Foods" in results["manufacturer"]["value"]

    assert results["complete_address"]["detected"] is True
    assert "110020" in results["complete_address"]["value"]

    assert results["country_of_origin"]["detected"] is True
    assert "India" in results["country_of_origin"]["value"]

    assert results["batch_lot_number"]["detected"] is True
    assert "RAF-2026-B8" in results["batch_lot_number"]["value"]

    assert results["dates"]["detected"] is True
    assert "09/2026" in results["dates"]["value"]





