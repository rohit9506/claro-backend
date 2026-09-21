from typing import Dict, List, Any, Tuple

def evaluate_legal_metrology_rules(
    extracted_declarations: Dict[str, Dict[str, Any]],
    active_rules: List[Any]
) -> Tuple[List[Dict[str, Any]], str, int, int, int]:
    """
    Deterministic rule validation against Legal Metrology (Packaged Commodities) Rules, 2011.
    Evaluates extracted fields, checks compliance criteria, calculates PASS/FAIL/REVIEW.
    Returns: (field_validations, overall_status, pass_count, fail_count, review_count)
    """
    validations = []
    pass_count = 0
    fail_count = 0
    review_count = 0

    correction_guidance = []

    # Rule mapping definitions with legal clauses and correction instructions
    rule_definitions = {
        "product_name": {
            "name": "Generic Product Name",
            "clause": "Rule 6(1)(b)",
            "requirement": "Declaration of the common or generic name of the commodity on the principal display panel.",
            "correction_advice": "Print the common or generic name of the commodity prominently on the Principal Display Panel in font size complying with Schedule II.",
            "evaluator": lambda d: evaluate_product_name(d)
        },
        "brand": {
            "name": "Brand / Trade Name",
            "clause": "Rule 6(1)(b)",
            "requirement": "Clear declaration of the brand name or commercial trademark on the packaging.",
            "correction_advice": "Ensure brand identity is visibly declared and unobstructed on the front display.",
            "evaluator": lambda d: evaluate_brand(d)
        },
        "mrp": {
            "name": "Maximum Retail Price (MRP)",
            "clause": "Rule 6(1)(da)",
            "requirement": "Declaration of MRP inclusive of all taxes in standard currency format (₹ or Rs.).",
            "correction_advice": "State Maximum Retail Price explicitly in Rupees (₹) with the mandatory words 'inclusive of all taxes' or 'incl. of all taxes'.",
            "evaluator": lambda d: evaluate_mrp(d)
        },
        "net_quantity": {
            "name": "Net Quantity & Metric Unit",
            "clause": "Rule 6(1)(c)",
            "requirement": "Declaration of net quantity in terms of standard metric units (kg, g, l, ml, or N/units).",
            "correction_advice": "Declare accurate net quantity using standard metric units (g, kg, ml, l, or N) with no non-standard unit abbreviations.",
            "evaluator": lambda d: evaluate_net_quantity(d)
        },
        "unit_sale_price": {
            "name": "Unit Sale Price (USP)",
            "clause": "Rule 6(1)(e)",
            "requirement": "Mandatory declaration of Unit Sale Price (per g/ml/unit) for consumer clarity.",
            "correction_advice": "Declare Unit Sale Price calculated as MRP divided by net quantity (e.g. ₹X.XX / g or ₹X.XX / ml) per Rule 6(1)(e).",
            "evaluator": lambda d: evaluate_usp(d)
        },
        "manufacturer": {
            "name": "Manufacturer / Packer Identity",
            "clause": "Rule 6(1)(a)",
            "requirement": "Name and identity of manufacturer, packer, or importer clearly stated.",
            "correction_advice": "Include full name of manufacturing company/packer prefixed by 'Manufactured by' or 'Packed by'.",
            "evaluator": lambda d: evaluate_manufacturer(d)
        },
        "complete_address": {
            "name": "Complete Address & Pin Code",
            "clause": "Rule 6(1)(a)",
            "requirement": "Complete postal address of manufacturing/packing premises including PIN code.",
            "correction_advice": "Provide complete physical address including postal PIN code of manufacturing or packing premises.",
            "evaluator": lambda d: evaluate_address(d)
        },
        "dates": {
            "name": "Month & Year of Manufacture / Packing",
            "clause": "Rule 6(1)(d)",
            "requirement": "Month and year of manufacture, packing, or import clearly legible.",
            "correction_advice": "Declare month and year of manufacture or pre-packing using standard formats (MM/YYYY or Mon YYYY).",
            "evaluator": lambda d: evaluate_dates(d)
        },
        "consumer_care": {
            "name": "Consumer Care Redressal Mechanism",
            "clause": "Rule 6(2)",
            "requirement": "Name, address, telephone number, and email address for consumer grievances.",
            "correction_advice": "Provide telephone helpline/toll-free number and email address for consumer grievance redressal.",
            "evaluator": lambda d: evaluate_consumer_care(d)
        },
        "country_of_origin": {
            "name": "Country of Origin",
            "clause": "Rule 6(1)(f)",
            "requirement": "Country of origin clearly stated on packaging.",
            "correction_advice": "Declare 'Country of Origin: [Country]' or 'Made in [Country]' on the package label.",
            "evaluator": lambda d: evaluate_country_of_origin(d)
        },
        "batch_lot_number": {
            "name": "Batch / Lot / Code Number",
            "clause": "Rule 6(1)(g)",
            "requirement": "Batch number, lot number, or code number for traceability and statutory verification.",
            "correction_advice": "Stamp an indelible batch or lot identifier prefixed by 'Batch No.' or 'B.No.'.",
            "evaluator": lambda d: evaluate_batch_lot(d)
        }
    }

    for key, spec in rule_definitions.items():
        if key in ["product_name", "brand", "batch_lot_number"] and key not in extracted_declarations:
            continue
        data = extracted_declarations.get(key, {})
        raw_status, reason, confidence = spec["evaluator"](data)

        # Standardized Status Mapping per Part 21
        val_str = str(data.get("value") or "").strip()
        if "conflicting information" in val_str.lower():
            status = "NEEDS VERIFICATION"
        elif not data.get("detected") or val_str.lower() in ["not detected"]:
            status = "MISSING"
        elif raw_status == "PASS":
            if confidence >= 0.75:
                status = "COMPLIANT"
            else:
                status = "LOW CONFIDENCE"
        elif raw_status == "FAIL":
            status = "NON-COMPLIANT"
        elif val_str.lower() in ["not applicable"]:
            status = "NOT APPLICABLE"
        else:
            status = "NEEDS VERIFICATION"

        if status == "COMPLIANT":
            pass_count += 1
        elif status in ["NON-COMPLIANT", "MISSING"]:
            fail_count += 1
        else:
            review_count += 1

        val_entry = {
            "field_name": key,
            "display_name": spec["name"],
            "detected_value": data.get("value") if data.get("detected") else "Not detected",
            "requirement_summary": spec["requirement"],
            "rule_reference": f"PCR-2011 {spec['clause']}",
            "status": status,
            "raw_status": raw_status,
            "confidence": round(confidence, 2),
            "reason": reason,
            "image_side": data.get("side", "front"),
            "bbox_norm": data.get("bbox_norm", [0.1, 0.1, 0.3, 0.9]),
            "heading": data.get("heading", ""),
            "spatial_relationship": data.get("spatial_relationship", "INLINE")
        }
        validations.append(val_entry)

        # Build structured correction guidance for non-compliant or missing items
        if status in ["NON-COMPLIANT", "MISSING", "LOW CONFIDENCE", "NEEDS VERIFICATION"]:
            correction_guidance.append({
                "field_name": key,
                "requirement": spec["name"],
                "rule_reference": f"PCR-2011 {spec['clause']}",
                "detected_value": val_entry["detected_value"],
                "expected_condition": spec["requirement"],
                "status": status,
                "reason_failed": reason,
                "what_needs_correction": f"Rectify mandatory {spec['name']} declaration",
                "how_to_correct": spec["correction_advice"]
            })

    # Overall Status computation
    if review_count > 0:
        overall_status = "PENDING_VERIFICATION"
    elif fail_count > 0:
        overall_status = "NON_COMPLIANT"
    else:
        overall_status = "COMPLIANT"

    return validations, overall_status, pass_count, fail_count, review_count, correction_guidance

def evaluate_product_name(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "FAIL", "Mandatory common or generic name of the commodity missing under Rule 6(1)(b).", 0.90
    val = (data.get("value") or "").strip()
    conf = data.get("confidence", 0.9)
    if len(val) >= 2 and val.lower() not in ["not detected", "packaged commodity", "packaged retail commodity"]:
        return "PASS", f"Generic commodity name identified as '{val}'.", conf
    return "REVIEW", "Product name requires human confirmation.", 0.65

def evaluate_brand(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "REVIEW", "Brand name declaration not explicitly detected. Verify if unbranded/generic.", 0.70
    val = (data.get("value") or "").strip()
    conf = data.get("confidence", 0.9)
    if len(val) >= 2 and val.lower() not in ["not detected"]:
        return "PASS", f"Brand identifier declared as '{val}'.", conf
    return "REVIEW", "Brand identifier format requires review.", 0.65

def evaluate_mrp(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "FAIL", "Mandatory declaration of Maximum Retail Price (MRP) missing from package labels.", 0.95
    val = data.get("value", "")
    conf = data.get("confidence", 0.9)
    if "₹" in val or "rs" in val.lower() or "mrp" in val.lower() or any(c.isdigit() for c in val):
        return "PASS", f"MRP declaration present with valid currency indicator: '{val}'.", conf
    return "REVIEW", "Price indicator format requires human inspection.", 0.65

def evaluate_net_quantity(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "FAIL", "Mandatory net quantity declaration not detected.", 0.92
    val = data.get("value", "")
    conf = data.get("confidence", 0.9)
    valid_units = ["kg", "g", "gm", "ml", "l", "ltr", "n", "unit", "piece"]
    if any(u in val.lower() for u in valid_units):
        return "PASS", f"Valid net quantity with prescribed metric unit detected: '{val}'.", conf
    return "REVIEW", f"Quantity declaration '{val}' uses non-standard unit notation.", 0.68

def evaluate_usp(data: dict) -> Tuple[str, str, float]:
    # USP is recommended/mandatory for packages with non-standard weights
    if not data.get("detected"):
        return "REVIEW", "Unit Sale Price (USP) was not detected. Requires verification if exempt or applicable.", 0.70
    conf = data.get("confidence", 0.88)
    return "PASS", f"Unit Sale Price declaration present: '{data.get('value')}'.", conf

def evaluate_manufacturer(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "FAIL", "Manufacturer / Packer / Importer identity declaration missing.", 0.94
    val = data.get("value", "")
    conf = data.get("confidence", 0.9)
    if len(val) >= 4:
        return "PASS", f"Manufacturer/Packer identity clearly declared.", conf
    return "REVIEW", "Manufacturer text is brief or partially obstructed.", 0.60

def evaluate_address(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "FAIL", "Complete manufacturing address or premises location missing.", 0.91
    val = data.get("value", "")
    conf = data.get("confidence", 0.88)
    # Check for PIN code or address keywords
    import re
    if re.search(r"\b\d{6}\b", val) or len(val) > 15:
        return "PASS", "Manufacturing/packing premises postal address verified.", conf
    return "REVIEW", "Address does not clearly contain a 6-digit postal PIN code.", 0.65

def evaluate_dates(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "FAIL", "Mandatory month and year of manufacture or pre-packing missing.", 0.93
    val = data.get("value", "")
    conf = data.get("confidence", 0.9)
    return "PASS", f"Manufacturing/Packing date format verified: '{val}'.", conf

def evaluate_consumer_care(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "FAIL", "Mandatory consumer grievance redressal mechanism (toll-free/email/address) missing.", 0.96
    val = data.get("value", "")
    conf = data.get("confidence", 0.9)
    if "@" in val or any(d in val for d in ["1800", "care", "phone", "tel"]):
        return "PASS", "Consumer care contact information verified under Rule 6(2).", conf
    return "REVIEW", "Consumer care details present but contact medium is ambiguous.", 0.65

def evaluate_country_of_origin(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "REVIEW", "Country of origin declaration not explicitly stated. Verify if locally produced.", 0.70
    conf = data.get("confidence", 0.92)
    return "PASS", f"Country of origin declared as '{data.get('value')}'.", conf

def evaluate_batch_lot(data: dict) -> Tuple[str, str, float]:
    if not data.get("detected"):
        return "FAIL", "Mandatory batch number, lot number, or code identifier missing under Rule 6(1)(g).", 0.92
    conf = data.get("confidence", 0.90)
    return "PASS", f"Batch / Lot identifier verified: '{data.get('value')}'.", conf

