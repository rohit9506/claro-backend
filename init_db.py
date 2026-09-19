import os
import json
import secrets
import string
import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import engine, SessionLocal, Base
from models import User, Product, Rule, Inspection, FieldValidation, AuditLog
from security import hash_password
from config import SETUP_FLAG_FILE

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

def generate_random_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in pwd)
            and any(c.isupper() for c in pwd)
            and any(c.isdigit() for c in pwd)
            and any(c in "!@#$%^&*" for c in pwd)):
            return pwd

def init_database():
    """Initializes tables, auto-generates first-time Admin, and seeds standard rules."""
    Base.metadata.create_all(bind=engine)
    
    # Auto-migrate any newly added columns for SQLite
    with engine.connect() as conn:
        for col in ["right_image", "left_image"]:
            try:
                conn.execute(text(f"ALTER TABLE inspections ADD COLUMN {col} VARCHAR(255)"))
                conn.commit()
            except Exception:
                pass

    db: Session = SessionLocal()

    try:
        # 1. Check for Admin Account
        admin_user = db.query(User).filter(User.role == "ROLE_ADMIN").first()
        if not admin_user:
            admin_username = "claro_admin"
            admin_temp_password = generate_random_password(16)
            admin_hash = hash_password(admin_temp_password)

            admin_user = User(
                full_name="Chief Legal Metrology Administrator",
                username=admin_username,
                email="admin@claro.gov.in",
                password_hash=admin_hash,
                role="ROLE_ADMIN",
                is_active=True,
                is_temporary_password=True,
                email_verified=True,
                email_verified_at=utc_now()
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)

            # Store credentials securely for first-time setup UI notification
            setup_info = {
                "first_time_setup": True,
                "username": admin_username,
                "temporary_password": admin_temp_password,
                "message": "First-time Administrator account created. Save these credentials before continuing."
            }
            with open(SETUP_FLAG_FILE, "w") as f:
                json.dump(setup_info, f)

            print("=" * 60)
            print("CLARO FIRST-TIME SETUP COMPLETED")
            print(f"Generated Admin Username: {admin_username}")
            print(f"Generated Temporary Password: {admin_temp_password}")
            print("=" * 60)

        # 2. Seed Default Officer Account for Demo/Testing
        officer = db.query(User).filter(User.username == "officer_verma").first()
        if not officer:
            officer = User(
                full_name="Inspector Rajesh Verma",
                username="officer_verma",
                email="r.verma@legalmetrology.gov.in",
                password_hash=hash_password("Inspector@2026!Verma"),
                role="ROLE_OFFICER",
                officer_id="LM-DL-8492",
                department="Legal Metrology Enforcement Wing",
                mobile_number="+91 98765 43210",
                is_active=True,
                is_temporary_password=False,
                email_verified=True,
                email_verified_at=utc_now()
            )
            db.add(officer)
            db.commit()

        # 3. Seed Standard Legal Metrology Rules (Packaged Commodities Rules 2011)
        if db.query(Rule).count() == 0:
            default_rules = [
                Rule(
                    rule_code="LMR-2011-R6-1-A",
                    rule_name="Manufacturer / Packer Identity & Address",
                    declaration_checked="Manufacturer / Packer / Importer",
                    requirement="Name and complete postal address of the manufacturer, packer, or importer with postal PIN code.",
                    applicable_category="All Packaged Commodities",
                    validation_criteria="Must state entity name and valid 6-digit postal PIN code.",
                    legal_source="Legal Metrology (Packaged Commodities) Rules, 2011 Rule 6(1)(a)",
                    version="2011.Amended2021",
                    is_active=True
                ),
                Rule(
                    rule_code="LMR-2011-R6-1-B",
                    rule_name="Common or Generic Commodity Name",
                    declaration_checked="Commodity Name",
                    requirement="The generic or common name of the commodity must be prominently declared.",
                    applicable_category="All Packaged Commodities",
                    validation_criteria="Clear commodity designation legible to consumers.",
                    legal_source="Legal Metrology (Packaged Commodities) Rules, 2011 Rule 6(1)(b)",
                    version="2011.Amended2021",
                    is_active=True
                ),
                Rule(
                    rule_code="LMR-2011-R6-1-C",
                    rule_name="Standard Net Quantity Declaration",
                    declaration_checked="Net Quantity",
                    requirement="Net quantity in terms of standard units of weight or measure (g, kg, ml, l) or number.",
                    applicable_category="All Packaged Commodities",
                    validation_criteria="Must use SI metric units; no non-standard symbols allowed.",
                    legal_source="Legal Metrology (Packaged Commodities) Rules, 2011 Rule 6(1)(c)",
                    version="2011.Amended2021",
                    is_active=True
                ),
                Rule(
                    rule_code="LMR-2011-R6-1-D",
                    rule_name="Month & Year of Manufacture / Packing",
                    declaration_checked="Manufacturing Date",
                    requirement="Month and year of manufacture or pre-packing must be clearly visible.",
                    applicable_category="All Packaged Commodities",
                    validation_criteria="Standard date formatting (MM/YYYY or DD/MM/YYYY).",
                    legal_source="Legal Metrology (Packaged Commodities) Rules, 2011 Rule 6(1)(d)",
                    version="2011.Amended2021",
                    is_active=True
                ),
                Rule(
                    rule_code="LMR-2011-R6-1-DA",
                    rule_name="Maximum Retail Price (MRP)",
                    declaration_checked="Maximum Retail Price",
                    requirement="Maximum Retail Price inclusive of all taxes in standard currency symbols (₹ or Rs.).",
                    applicable_category="All Packaged Commodities",
                    validation_criteria="Numeric value accompanied by 'incl. of all taxes' or equivalent.",
                    legal_source="Legal Metrology (Packaged Commodities) Rules, 2011 Rule 6(1)(da)",
                    version="2011.Amended2021",
                    is_active=True
                ),
                Rule(
                    rule_code="LMR-2011-R6-1-E",
                    rule_name="Unit Sale Price (USP)",
                    declaration_checked="Unit Sale Price",
                    requirement="Mandatory Unit Sale Price (per g or per ml) for commodities sold in non-standard units.",
                    applicable_category="Packaged Food & Retail Goods",
                    validation_criteria="Unit price indicated rounded to two decimal places.",
                    legal_source="Legal Metrology (Packaged Commodities) Rules, 2011 Rule 6(1)(e)",
                    version="2011.Amended2021",
                    is_active=True
                ),
                Rule(
                    rule_code="LMR-2011-R6-2",
                    rule_name="Consumer Grievance Redressal Mechanism",
                    declaration_checked="Consumer Care Details",
                    requirement="Name, address, telephone number, and email address of person/office to contact for consumer complaints.",
                    applicable_category="All Packaged Commodities",
                    validation_criteria="Toll-free or phone number plus valid email or postal contact.",
                    legal_source="Legal Metrology (Packaged Commodities) Rules, 2011 Rule 6(2)",
                    version="2011.Amended2021",
                    is_active=True
                )
            ]
            db.add_all(default_rules)
            db.commit()

        # 4. Seed Standard Products (FMCG and Packaged Commodities)
        if db.query(Product).count() == 0:
            demo_products = [
                Product(
                    name="Haldiram's Aloo Bhujia",
                    brand="Haldiram's",
                    category="Snacks & Savouries",
                    barcode="8904004400123",
                    mrp="₹50.00",
                    net_quantity="200 g",
                    unit_sale_price="₹0.25 / g",
                    mfg_date="08/2026",
                    expiry_date="02/2027",
                    country_of_origin="India",
                    manufacturer="Haldiram Snacks Pvt. Ltd.",
                    packer="Haldiram Snacks Pvt. Ltd.",
                    complete_address="Plot No. 14, Commercial Complex, Sector 68, Noida, Uttar Pradesh - 201301",
                    consumer_care_email="care@haldirams.com",
                    consumer_care_phone="1800-102-4040",
                    consumer_care_address="Customer Grievance Cell, Sector 68, Noida - 201301",
                    ingredients="Potato Flakes (45%), Edible Vegetable Oil (Palmolein), Gram Pulse Flour, Spices & Condiments, Iodised Salt.",
                    nutrition_facts=json.dumps({
                        "serving_size": "100 g",
                        "calories": 576,
                        "protein_g": 8.5,
                        "carbs_g": 42.0,
                        "sugar_g": 2.1,
                        "fat_g": 41.5,
                        "sodium_mg": 780
                    }),
                    allergens="May contain traces of gluten and peanut.",
                    is_demo=True
                ),
                Product(
                    name="Tata Salt Vacuum Evaporated",
                    brand="Tata Salt",
                    category="Pantry Essentials",
                    barcode="8901030384102",
                    mrp="₹28.00",
                    net_quantity="1 kg",
                    unit_sale_price="₹0.028 / g",
                    mfg_date="07/2026",
                    expiry_date="07/2028",
                    country_of_origin="India",
                    manufacturer="Tata Consumer Products Ltd.",
                    packer="Tata Consumer Products Ltd.",
                    complete_address="1, Bishop Lefroy Road, Kolkata, West Bengal - 700020",
                    consumer_care_email="care@tataconsumer.com",
                    consumer_care_phone="1800-108-4488",
                    consumer_care_address="Tata Consumer Care Cell, Kolkata - 700020",
                    ingredients="Edible Common Salt, Potassium Iodate (30 ppm), Anti-caking Agent (INS 551).",
                    nutrition_facts=json.dumps({
                        "serving_size": "100 g",
                        "calories": 0,
                        "protein_g": 0.0,
                        "carbs_g": 0.0,
                        "sugar_g": 0.0,
                        "fat_g": 0.0,
                        "sodium_mg": 38700
                    }),
                    allergens="None",
                    is_demo=True
                ),
                Product(
                    name="Amul Pasteurized Butter",
                    brand="Amul",
                    category="Dairy Products",
                    barcode="8901262010058",
                    mrp="₹60.00",
                    net_quantity="100 g",
                    unit_sale_price="₹0.60 / g",
                    mfg_date="09/2026",
                    expiry_date="06/2027",
                    country_of_origin="India",
                    manufacturer="Gujarat Cooperative Milk Marketing Federation Ltd.",
                    packer="GCMMF Ltd.",
                    complete_address="Amul Dairy Road, Anand, Gujarat - 388001",
                    consumer_care_email="customercare@amul.coop",
                    consumer_care_phone="1800-258-3333",
                    consumer_care_address="Amul Customer Service, Anand - 388001",
                    ingredients="Butter (Milk Fat 80%), Common Salt, Annatto (INS 160b).",
                    nutrition_facts=json.dumps({
                        "serving_size": "100 g",
                        "calories": 722,
                        "protein_g": 0.6,
                        "carbs_g": 0.0,
                        "sugar_g": 0.0,
                        "fat_g": 80.0,
                        "sodium_mg": 830
                    }),
                    allergens="Contains Milk.",
                    is_demo=True
                )
            ]
            db.add_all(demo_products)
            db.commit()

        # 5. Seed initial Compliant and Non-Compliant Inspections
        if db.query(Inspection).count() == 0:
            haldiram = db.query(Product).filter(Product.brand == "Haldiram's").first()
            officer_user = db.query(User).filter(User.username == "officer_verma").first()

            insp1 = Inspection(
                inspection_number="INSP-2026-0001",
                officer_id=officer_user.id if officer_user else None,
                product_id=haldiram.id if haldiram else None,
                product_name="Haldiram's Aloo Bhujia 200g",
                status="COMPLIANT",
                pass_count=8,
                fail_count=0,
                review_count=0,
                location="Big Bazaar Superstore, Connaught Place, New Delhi",
                officer_notes="All PCR 2011 declarations verified. Clear contrast and standard metric units.",
                created_at=utc_now() - datetime.timedelta(days=2),
                finalized_at=utc_now() - datetime.timedelta(days=2)
            )
            db.add(insp1)
            db.commit()

            validations_seed = [
                FieldValidation(
                    inspection_id=insp1.id,
                    field_name="mrp",
                    detected_value="₹50.00 (incl. of all taxes)",
                    requirement_summary="Declaration of MRP inclusive of all taxes.",
                    rule_reference="PCR-2011 Rule 6(1)(da)",
                    status="PASS",
                    confidence=0.98,
                    reason="MRP declaration present with ₹ symbol and tax declaration.",
                    image_side="front",
                    bbox_json="[0.72, 0.60, 0.82, 0.92]"
                ),
                FieldValidation(
                    inspection_id=insp1.id,
                    field_name="net_quantity",
                    detected_value="200 g",
                    requirement_summary="Net quantity in standard metric units.",
                    rule_reference="PCR-2011 Rule 6(1)(c)",
                    status="PASS",
                    confidence=0.96,
                    reason="Standard metric unit 'g' clearly indicated.",
                    image_side="front",
                    bbox_json="[0.83, 0.65, 0.91, 0.88]"
                ),
                FieldValidation(
                    inspection_id=insp1.id,
                    field_name="manufacturer",
                    detected_value="Haldiram Snacks Pvt. Ltd.",
                    requirement_summary="Manufacturer identity and address.",
                    rule_reference="PCR-2011 Rule 6(1)(a)",
                    status="PASS",
                    confidence=0.94,
                    reason="Manufacturer identity declared.",
                    image_side="back",
                    bbox_json="[0.15, 0.10, 0.28, 0.85]"
                ),
                FieldValidation(
                    inspection_id=insp1.id,
                    field_name="consumer_care",
                    detected_value="1800-102-4040, care@haldirams.com",
                    requirement_summary="Consumer grievance redressal details.",
                    rule_reference="PCR-2011 Rule 6(2)",
                    status="PASS",
                    confidence=0.95,
                    reason="Toll-free phone and official email verified.",
                    image_side="back",
                    bbox_json="[0.45, 0.12, 0.58, 0.90]"
                )
            ]
            db.add_all(validations_seed)
            db.commit()

            # Seed Inspection 2 (Non-compliant example)
            insp2 = Inspection(
                inspection_number="INSP-2026-0002",
                officer_id=officer_user.id if officer_user else None,
                product_id=None,
                product_name="Imported Wafer Rolls 150g",
                status="NON_COMPLIANT",
                pass_count=6,
                fail_count=2,
                review_count=0,
                location="Metro Cash & Carry, Gurugram",
                officer_notes="Violations detected under Rule 6(2) and Rule 6(1)(e). Missing Consumer Care helpline.",
                created_at=utc_now() - datetime.timedelta(days=1),
                finalized_at=utc_now() - datetime.timedelta(days=1)
            )
            db.add(insp2)
            db.commit()

            validations_seed2 = [
                FieldValidation(
                    inspection_id=insp2.id,
                    field_name="mrp",
                    detected_value="₹140.00",
                    requirement_summary="Declaration of MRP inclusive of all taxes.",
                    rule_reference="PCR-2011 Rule 6(1)(da)",
                    status="PASS",
                    confidence=0.95,
                    reason="MRP detected.",
                    image_side="front",
                    bbox_json="[0.65, 0.50, 0.75, 0.85]"
                ),
                FieldValidation(
                    inspection_id=insp2.id,
                    field_name="consumer_care",
                    detected_value="Not detected on submitted package sides",
                    requirement_summary="Consumer grievance redressal mechanism.",
                    rule_reference="PCR-2011 Rule 6(2)",
                    status="FAIL",
                    confidence=0.94,
                    reason="Mandatory consumer grievance redressal phone or email missing.",
                    image_side="back",
                    bbox_json="[0.10, 0.10, 0.30, 0.90]"
                ),
                FieldValidation(
                    inspection_id=insp2.id,
                    field_name="unit_sale_price",
                    detected_value="Not detected on submitted package sides",
                    requirement_summary="Mandatory Unit Sale Price (USP).",
                    rule_reference="PCR-2011 Rule 6(1)(e)",
                    status="FAIL",
                    confidence=0.91,
                    reason="Unit Sale Price missing for non-standard package volume.",
                    image_side="front",
                    bbox_json="[0.10, 0.10, 0.30, 0.90]"
                )
            ]
            db.add_all(validations_seed2)
            db.commit()

            print("Sample seed inspections successfully created!")

    finally:
        db.close()

if __name__ == "__main__":
    init_database()
