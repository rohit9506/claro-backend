import re
from typing import Dict, List, Any, Optional, Tuple

def get_box_props(bbox: Optional[List[float]]) -> Optional[Dict[str, float]]:
    """Calculates geometric properties of a normalized [ymin, xmin, ymax, xmax] bounding box."""
    if not bbox or len(bbox) < 4:
        return None
    ymin, xmin, ymax, xmax = [float(c) for c in bbox]
    w = max(0.001, xmax - xmin)
    h = max(0.001, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    area = w * h
    return {
        "ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax,
        "w": w, "h": h, "cx": cx, "cy": cy, "area": area
    }

def calc_spatial_relationship(head_box: Dict[str, float], val_box: Dict[str, float]) -> Tuple[str, float]:
    """
    Computes 2D spatial arrangement and proximity affinity between a heading label and a candidate value.
    Returns: (relationship_name, spatial_score)
    Where relationship_name is one of:
      - 'BELOW'
      - 'RIGHT'
      - 'STICKER_BOX'
      - 'DIAGONAL_OFFSET'
      - 'ABOVE'
      - 'LEFT'
    """
    dx = val_box["cx"] - head_box["cx"]
    dy = val_box["cy"] - head_box["cy"]
    
    # Overlap computations
    overlap_x = max(0.0, min(head_box["xmax"], val_box["xmax"]) - max(head_box["xmin"], val_box["xmin"]))
    overlap_y = max(0.0, min(head_box["ymax"], val_box["ymax"]) - max(head_box["ymin"], val_box["ymin"]))
    
    min_w = min(head_box["w"], val_box["w"])
    min_h = min(head_box["h"], val_box["h"])
    
    x_aligned = (overlap_x / min_w) > 0.25 or abs(dx) < 0.20
    y_aligned = (overlap_y / min_h) > 0.25 or abs(dy) < 0.08
    
    # 1. Directly RIGHT (same horizontal band, value is to the right)
    if dx > 0 and 0.01 <= dx <= 0.55 and y_aligned:
        score = 1.0 - (dx / 0.55) * 0.3
        return "RIGHT", round(score, 3)
    
    # 2. Directly BELOW (same column, value is underneath heading)
    if dy > 0 and 0.01 <= dy <= 0.35 and x_aligned:
        score = 1.0 - (dy / 0.35) * 0.3
        return "BELOW", round(score, 3)
    
    # 3. Clustered STICKER_BOX / Sub-panel (tight spatial proximity within designated bounding region)
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= 0.35:
        score = max(0.4, 0.9 - dist * 1.5)
        return "STICKER_BOX", round(score, 3)
    
    # 4. DIAGONAL_OFFSET (moderate diagonal shift within label cluster)
    if 0.35 < dist <= 0.55:
        score = max(0.3, 0.7 - dist * 0.8)
        return "DIAGONAL_OFFSET", round(score, 3)
    
    # 5. Above or Left fallback
    if dy < 0 and x_aligned and abs(dy) <= 0.25:
        return "ABOVE", 0.6
    if dx < 0 and y_aligned and abs(dx) <= 0.40:
        return "LEFT", 0.6
        
    return "DISTANT", 0.1

def extract_declarations_from_multi_side(
    side_detections: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Universal 2D Spatial + Contextual Extraction Engine across all package views:
    Front, Back, Right Side, Left Side, and Side.
    
    Analyzes normalized bounding box geometry ([ymin, xmin, ymax, xmax]),
    identifies headings vs candidate values, evaluates spatial pairings (INLINE, RIGHT, BELOW, STICKER_BOX),
    and strictly disambiguates conflicting fields (e.g. MRP vs Offer Price vs USP; Net Quantity vs Nutritional Serving).
    
    Returns structured, evidence-attributed declarations:
    {
        "field_name": {
            "value": str,
            "raw_val": Optional[str],
            "confidence": float,
            "side": str,
            "bbox_norm": [ymin, xmin, ymax, xmax],
            "heading": Optional[str],
            "spatial_relationship": str,
            "detected": bool
        }
    }
    """
    # Canonical declarations container
    extracted = {
        "product_name": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "front", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "Primary Display Panel", "spatial_relationship": "PDP_CENTER", "detected": False},
        "brand": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "front", "bbox_norm": [0.05, 0.1, 0.25, 0.9], "heading": "Brand Identity", "spatial_relationship": "PDP_TOP", "detected": False},
        "mrp": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "front", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "MRP", "spatial_relationship": "INLINE", "detected": False},
        "net_quantity": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "front", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "Net Quantity", "spatial_relationship": "INLINE", "detected": False},
        "unit_sale_price": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "front", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "Unit Sale Price", "spatial_relationship": "INLINE", "detected": False},
        "manufacturer": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "back", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "Manufacturer", "spatial_relationship": "INLINE", "detected": False},
        "complete_address": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "back", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "Premises / Address", "spatial_relationship": "BELOW", "detected": False},
        "dates": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "back", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "Mfg / Pkd Date", "spatial_relationship": "INLINE", "detected": False},
        "consumer_care": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "back", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "Consumer Redressal", "spatial_relationship": "INLINE", "detected": False},
        "country_of_origin": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "back", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "Country of Origin", "spatial_relationship": "INLINE", "detected": False},
        "batch_lot_number": {"value": "Not detected", "raw_val": None, "confidence": 0.0, "side": "back", "bbox_norm": [0.1, 0.1, 0.3, 0.9], "heading": "Batch / Lot No", "spatial_relationship": "INLINE", "detected": False}
    }

    # Normalize sides order for optimal statutory search priority:
    # Front (PDP, Brand, Qty, MRP), Back (Declarations, Mfg, Address, Dates), Right/Left Side (Batch, MRP sticker)
    side_priority = ["front", "back", "right_side", "left_side", "side", "right", "left"]
    sorted_sides = sorted(
        side_detections.keys(),
        key=lambda s: side_priority.index(s.lower()) if s.lower() in side_priority else 99
    )

    for side in sorted_sides:
        detections = side_detections.get(side) or []
        if not detections:
            continue

        # Prepare geometric bounding box representations for each detected text item
        enriched_dets = []
        for det in detections:
            t = det.get("text", "").strip()
            if not t:
                continue
            c = float(det.get("confidence", 0.8))
            b = det.get("bbox_norm")
            props = get_box_props(b)
            enriched_dets.append({
                "text": t,
                "confidence": c,
                "bbox_norm": b,
                "props": props
            })

        # =========================================================================
        # 1. MAXIMUM RETAIL PRICE (MRP) — 2D SPATIAL + CONTEXTUAL DISAMBIGUATION
        # =========================================================================
        if not extracted["mrp"]["detected"]:
            # Contextual exclusions for MRP: Discounts, offers, per-unit rates, nutritional numbers
            def is_discount_or_usp(text_lower: str) -> bool:
                return any(w in text_lower for w in [
                    "save", "discount", "off", "offer", "cashback", "deal",
                    "per g", "per gm", "per kg", "per ml", "per l", "per unit",
                    "/g", "/gm", "/kg", "/ml", "/unit", "usp", "unit sale price"
                ])

            # A. Check for single-line Inline MRP: e.g. "MRP Rs. 120.00 (incl. of all taxes)"
            for item in enriched_dets:
                t = item["text"]
                tl = t.lower()
                if is_discount_or_usp(tl):
                    continue
                
                # Regex for inline declaration
                mrp_inline = re.search(
                    r"(?:m\.?r\.?p\.?|max(?:imum)?\s*retail\s*price|₹|rs\.?)\s*[:.]?\s*(?:₹|rs\.?)?\s*([0-9]+(?:\.[0-9]{1,2})?)",
                    t, re.IGNORECASE
                )
                if mrp_inline and not any(nut in tl for nut in ["kcal", "protein", "sodium", "fat", "sugar", "100g"]):
                    amount = mrp_inline.group(1)
                    # Verify it's a realistic price, not a date year or weight
                    if float(amount) > 0 and amount not in ["2024", "2025", "2026", "2027"]:
                        extracted["mrp"] = {
                            "value": f"₹{amount} (incl. of all taxes)",
                            "raw_val": amount,
                            "confidence": item["confidence"],
                            "side": side,
                            "bbox_norm": item["bbox_norm"],
                            "heading": "MRP (incl. of all taxes)",
                            "spatial_relationship": "INLINE",
                            "detected": True
                        }
                        break

            # B. 2D Spatial Pairing: Headings on one line ("MRP (Incl. of all taxes)"), Value on adjacent line
            if not extracted["mrp"]["detected"]:
                for h_item in enriched_dets:
                    ht = h_item["text"]
                    htl = ht.lower()
                    if re.search(r"\b(m\.?r\.?p\.?|max(?:imum)?\s*retail\s*price)\b", htl) and not is_discount_or_usp(htl):
                        # Search for proximate numeric currency values across other bounding boxes
                        best_pair = None
                        best_score = 0.0

                        for v_item in enriched_dets:
                            if v_item == h_item or not h_item["props"] or not v_item["props"]:
                                continue
                            vt = v_item["text"]
                            vtl = vt.lower()
                            if is_discount_or_usp(vtl) or any(nut in vtl for nut in ["kcal", "g", "ml", "mg"]):
                                continue
                            
                            val_match = re.search(r"(?:₹|rs\.?)?\s*([0-9]+(?:\.[0-9]{1,2})?)\b", vt, re.IGNORECASE)
                            if val_match:
                                rel, score = calc_spatial_relationship(h_item["props"], v_item["props"])
                                if rel in ["RIGHT", "BELOW", "STICKER_BOX"] and score > best_score:
                                    best_score = score
                                    best_pair = (val_match.group(1), v_item, rel, h_item["text"])

                        if best_pair:
                            amount, matched_item, rel_name, head_text = best_pair
                            extracted["mrp"] = {
                                "value": f"₹{amount} (incl. of all taxes)",
                                "raw_val": amount,
                                "confidence": round(matched_item["confidence"] * best_score, 2),
                                "side": side,
                                "bbox_norm": matched_item["bbox_norm"],
                                "heading": head_text,
                                "spatial_relationship": rel_name,
                                "detected": True
                            }
                            break

        # =========================================================================
        # 2. NET QUANTITY — 2D SPATIAL + NUTRITIONAL DISAMBIGUATION
        # =========================================================================
        if not extracted["net_quantity"]["detected"]:
            def is_nutritional(text_lower: str) -> bool:
                return any(k in text_lower for k in [
                    "nutrition", "energy", "calorie", "kcal", "protein",
                    "carbohydrate", "carb", "sugar", "total fat", "trans fat",
                    "saturated", "sodium", "cholesterol", "per 100g", "per serving",
                    "serving size", "servings"
                ])

            # A. Inline Net Quantity Detection
            for item in enriched_dets:
                t = item["text"]
                tl = t.lower()
                if is_nutritional(tl):
                    continue

                qty_match = re.search(
                    r"(?:net\s*(?:wt\.?|weight|qty\.?|quantity|content|volume|mass)|net)\s*[:.]?\s*([0-9]+(?:\.[0-9]+)?\s*(?:kg|g|gm|gms|grams?|ml|l|ltr|litres?|mg|units?|pieces?|n|nos))\b",
                    t, re.IGNORECASE
                )
                if not qty_match:
                    # Look for standalone quantity if explicitly prefixed with standard volume/mass indicators
                    qty_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?\s*(?:kg|gm|gms|ml|ltr|litres?|mg))\b", t, re.IGNORECASE)

                if qty_match:
                    raw_qty = qty_match.group(1).strip()
                    extracted["net_quantity"] = {
                        "value": raw_qty,
                        "raw_val": raw_qty,
                        "confidence": item["confidence"],
                        "side": side,
                        "bbox_norm": item["bbox_norm"],
                        "heading": "Net Quantity",
                        "spatial_relationship": "INLINE",
                        "detected": True
                    }
                    break

            # B. 2D Spatial Pairing for Net Quantity (Heading on line A, Value on line B)
            if not extracted["net_quantity"]["detected"]:
                for h_item in enriched_dets:
                    ht = h_item["text"]
                    htl = ht.lower()
                    if re.search(r"\b(net\s*(?:wt\.?|weight|qty\.?|quantity|content|volume|mass)|net)\b", htl) and not is_nutritional(htl):
                        best_pair = None
                        best_score = 0.0

                        for v_item in enriched_dets:
                            if v_item == h_item or not h_item["props"] or not v_item["props"]:
                                continue
                            vt = v_item["text"]
                            vtl = vt.lower()
                            if is_nutritional(vtl):
                                continue

                            val_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?\s*(?:kg|g|gm|gms|grams?|ml|l|ltr|litres?|mg|units?|pieces?|n|nos))\b", vt, re.IGNORECASE)
                            if val_match:
                                rel, score = calc_spatial_relationship(h_item["props"], v_item["props"])
                                if rel in ["RIGHT", "BELOW", "STICKER_BOX"] and score > best_score:
                                    best_score = score
                                    best_pair = (val_match.group(1).strip(), v_item, rel, h_item["text"])

                        if best_pair:
                            qty_val, matched_item, rel_name, head_text = best_pair
                            extracted["net_quantity"] = {
                                "value": qty_val,
                                "raw_val": qty_val,
                                "confidence": round(matched_item["confidence"] * best_score, 2),
                                "side": side,
                                "bbox_norm": matched_item["bbox_norm"],
                                "heading": head_text,
                                "spatial_relationship": rel_name,
                                "detected": True
                            }
                            break

        # =========================================================================
        # 3. UNIT SALE PRICE (USP) — MANDATORY DECLARATION
        # =========================================================================
        if not extracted["unit_sale_price"]["detected"]:
            for item in enriched_dets:
                t = item["text"]
                usp_match = re.search(
                    r"(?:u\.?s\.?p\.?|unit\s*sale\s*price|unit\s*price)\s*[:.]?\s*(?:₹|rs\.?)?\s*([0-9]+(?:\.[0-9]{1,2})?\s*(?:per|\/)\s*(?:g|gm|kg|ml|l|ltr|unit|piece|n))\b",
                    t, re.IGNORECASE
                )
                if not usp_match:
                    usp_match = re.search(r"(?:₹|rs\.?)\s*([0-9]+(?:\.[0-9]{1,2})?\s*(?:per|\/)\s*(?:g|gm|kg|ml|l|ltr|unit|piece|n))\b", t, re.IGNORECASE)

                if usp_match:
                    usp_val = usp_match.group(1).strip()
                    extracted["unit_sale_price"] = {
                        "value": f"₹{usp_val}",
                        "raw_val": usp_val,
                        "confidence": item["confidence"],
                        "side": side,
                        "bbox_norm": item["bbox_norm"],
                        "heading": "Unit Sale Price (USP)",
                        "spatial_relationship": "INLINE",
                        "detected": True
                    }
                    break

        # =========================================================================
        # 4. MANUFACTURER / PACKER / IMPORTER & COMPLETE ADDRESS
        # =========================================================================
        if not extracted["manufacturer"]["detected"]:
            mfg_patterns = [
                r"(?:mfd\.?\s*by|manufactured\s*(?:and\s*packed)?\s*by|mfr\.?\s*by)\s*[:.]?\s*(.+)",
                r"(?:packed\s*by|packer)\s*[:.]?\s*(.+)",
                r"(?:marketed\s*by)\s*[:.]?\s*(.+)",
                r"(?:imported\s*by|importer)\s*[:.]?\s*(.+)"
            ]
            for item in enriched_dets:
                t = item["text"]
                for pat in mfg_patterns:
                    m = re.search(pat, t, re.IGNORECASE)
                    if m:
                        entity_name = m.group(1).strip()
                        # If entity name is substantial on same line
                        if len(entity_name) > 3 and not re.search(r"^(?:plot|survey|village|phase|road|p\.?o\.?)", entity_name, re.IGNORECASE):
                            extracted["manufacturer"] = {
                                "value": entity_name,
                                "raw_val": entity_name,
                                "confidence": item["confidence"],
                                "side": side,
                                "bbox_norm": item["bbox_norm"],
                                "heading": t[:m.start(1)].strip(" :.-"),
                                "spatial_relationship": "INLINE",
                                "detected": True
                            }
                            break
                        elif item["props"]:
                            # 2D Spatial search: look for corporate entity directly BELOW heading
                            best_below = None
                            for v_item in enriched_dets:
                                if v_item == item or not v_item["props"]:
                                    continue
                                rel, score = calc_spatial_relationship(item["props"], v_item["props"])
                                if rel in ["BELOW", "RIGHT"] and len(v_item["text"]) > 4:
                                    best_below = (v_item["text"], v_item, rel)
                                    break
                            if best_below:
                                v_text, v_item, rel = best_below
                                extracted["manufacturer"] = {
                                    "value": v_text,
                                    "raw_val": v_text,
                                    "confidence": round(v_item["confidence"] * 0.9, 2),
                                    "side": side,
                                    "bbox_norm": v_item["bbox_norm"],
                                    "heading": t,
                                    "spatial_relationship": rel,
                                    "detected": True
                                }
                                break
                if extracted["manufacturer"]["detected"]:
                    break

        # Complete Address & PIN Code
        if not extracted["complete_address"]["detected"]:
            for item in enriched_dets:
                t = item["text"]
                pin_match = re.search(r"\b([1-9][0-9]{2}\s?[0-9]{3})\b", t)
                addr_ind = re.search(r"(?:plot\s*no|ind\.?\s*area|industrial\s*area|phase|sector|road|dist|taluka|state|pin|p\.?o\.?|post\s*office|lane|street)", t, re.IGNORECASE)
                if pin_match or (addr_ind and len(t) > 20):
                    extracted["complete_address"] = {
                        "value": t.strip(),
                        "raw_val": pin_match.group(1) if pin_match else t.strip(),
                        "confidence": item["confidence"],
                        "side": side,
                        "bbox_norm": item["bbox_norm"],
                        "heading": "Manufacturing Premises Address",
                        "spatial_relationship": "BELOW" if extracted["manufacturer"]["detected"] else "INLINE",
                        "detected": True
                    }
                    break

        # =========================================================================
        # 5. DATES (MFG / PKD / EXPIRY / BEST BEFORE)
        # =========================================================================
        if not extracted["dates"]["detected"]:
            for item in enriched_dets:
                t = item["text"]
                date_match = re.search(
                    r"(?:mfg|pkd|packed|mfd|exp|expiry|best\s*before|use\s*by)\s*[:.]?\s*([0-9]{1,2}[\/\-\.][0-9]{2,4}|[a-z]{3}[\s\/\-\.][0-9]{2,4}|[0-9]{2}[\/\-\.][0-9]{2}[\/\-\.][0-9]{2,4})",
                    t, re.IGNORECASE
                )
                if date_match:
                    extracted["dates"] = {
                        "value": t.strip(),
                        "raw_val": date_match.group(1),
                        "confidence": item["confidence"],
                        "side": side,
                        "bbox_norm": item["bbox_norm"],
                        "heading": "Mfg / Expiry Date",
                        "spatial_relationship": "INLINE",
                        "detected": True
                    }
                    break
                elif item["props"] and re.search(r"\b(mfg|pkd|mfd|exp|use\s*by|best\s*before)\b", t, re.IGNORECASE):
                    # 2D Spatial search: Date value on adjacent coordinate
                    for v_item in enriched_dets:
                        if v_item == item or not v_item["props"]:
                            continue
                        val_m = re.search(r"\b([0-9]{1,2}[\/\-\.][0-9]{2,4}|[a-z]{3}[\s\/\-\.][0-9]{2,4}|[0-9]{2}[\/\-\.][0-9]{2}[\/\-\.][0-9]{2,4})\b", v_item["text"], re.IGNORECASE)
                        if val_m:
                            rel, score = calc_spatial_relationship(item["props"], v_item["props"])
                            if rel in ["RIGHT", "BELOW", "STICKER_BOX"]:
                                extracted["dates"] = {
                                    "value": f"{t}: {val_m.group(1)}",
                                    "raw_val": val_m.group(1),
                                    "confidence": round(v_item["confidence"] * score, 2),
                                    "side": side,
                                    "bbox_norm": v_item["bbox_norm"],
                                    "heading": t,
                                    "spatial_relationship": rel,
                                    "detected": True
                                }
                                break
                    if extracted["dates"]["detected"]:
                        break

        # =========================================================================
        # 6. CONSUMER CARE & GRIEVANCE REDRESSAL MECHANISM
        # =========================================================================
        if not extracted["consumer_care"]["detected"]:
            for item in enriched_dets:
                t = item["text"]
                phone_match = re.search(r"(?:toll\s*free|care|helpline|phone|call)\s*[:.]?\s*([0-9]{3,4}[-\s]?[0-9]{3,4}[-\s]?[0-9]{3,4}|1800[-\s]?[0-9]{3}[-\s]?[0-9]{3,4})", t, re.IGNORECASE)
                email_match = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", t)
                if phone_match or email_match or any(k in t.lower() for k in ["consumer care", "feedback", "grievance cell", "contact us"]):
                    extracted["consumer_care"] = {
                        "value": t.strip(),
                        "raw_val": phone_match.group(1) if phone_match else (email_match.group(1) if email_match else t.strip()),
                        "confidence": item["confidence"],
                        "side": side,
                        "bbox_norm": item["bbox_norm"],
                        "heading": "Consumer Redressal Mechanism",
                        "spatial_relationship": "INLINE",
                        "detected": True
                    }
                    break

        # =========================================================================
        # 7. COUNTRY OF ORIGIN
        # =========================================================================
        if not extracted["country_of_origin"]["detected"]:
            for item in enriched_dets:
                t = item["text"]
                origin_match = re.search(r"(?:country\s*of\s*origin|made\s*in|product\s*of)\s*[:.]?\s*([a-z\s]+)", t, re.IGNORECASE)
                if origin_match or "made in india" in t.lower() or "product of india" in t.lower():
                    val = origin_match.group(1).strip().title() if origin_match else "India"
                    if len(val) > 2:
                        extracted["country_of_origin"] = {
                            "value": val,
                            "raw_val": val,
                            "confidence": item["confidence"],
                            "side": side,
                            "bbox_norm": item["bbox_norm"],
                            "heading": "Country of Origin",
                            "spatial_relationship": "INLINE",
                            "detected": True
                        }
                        break

        # =========================================================================
        # 8. BATCH / LOT NUMBER
        # =========================================================================
        if not extracted["batch_lot_number"]["detected"]:
            for item in enriched_dets:
                t = item["text"]
                batch_m = re.search(r"\b(?:batch\s*(?:no\.?|number)?|lot\s*(?:no\.?|number)?|b\.?\s*no\.?)\s*[:.]?\s*([A-Za-z0-9\-\/]+)", t, re.IGNORECASE)
                if batch_m and len(batch_m.group(1)) >= 2:
                    extracted["batch_lot_number"] = {
                        "value": batch_m.group(1).strip(),
                        "raw_val": batch_m.group(1).strip(),
                        "confidence": item["confidence"],
                        "side": side,
                        "bbox_norm": item["bbox_norm"],
                        "heading": "Batch / Lot No",
                        "spatial_relationship": "INLINE",
                        "detected": True
                    }
                    break

        # =========================================================================
        # 9. PRODUCT NAME & BRAND (Primary Display Panel - Front View Analysis)
        # =========================================================================
        if side == "front":
            # Exclude legal and nutritional boilerplate
            def is_boilerplate(text: str) -> bool:
                tl = text.lower()
                return any(k in tl for k in [
                    "mrp", "net wt", "quantity", "batch", "fssai", "100g", "ingredients",
                    "nutrition", "save", "offer", "discount", "servings", "license",
                    "regd", "trademark", "patent", "expiry", "best before", "serving size"
                ])

            front_candidates = [it for it in enriched_dets if not is_boilerplate(it["text"]) and len(it["text"]) >= 3]

            # 1. Product Name: Prominent central display text (largest bounding box area on PDP)
            if not extracted["product_name"]["detected"] and front_candidates:
                pdp_candidates = [
                    it for it in front_candidates
                    if it["props"] and 0.10 <= it["props"]["cy"] <= 0.80
                ]
                if pdp_candidates:
                    pdp_candidates.sort(key=lambda it: it["props"]["area"] if it["props"] else 0, reverse=True)
                    best_pdp = pdp_candidates[0]
                    extracted["product_name"] = {
                        "value": best_pdp["text"],
                        "raw_val": best_pdp["text"],
                        "confidence": best_pdp["confidence"],
                        "side": side,
                        "bbox_norm": best_pdp["bbox_norm"],
                        "heading": "Product Name (PDP)",
                        "spatial_relationship": "PDP_CENTER",
                        "detected": True
                    }

            # 2. Brand: Upper text (typically above product name or at top of package ymin <= 0.30)
            if not extracted["brand"]["detected"] and front_candidates:
                pdp_val = extracted["product_name"].get("value")
                brand_candidates = [
                    it for it in front_candidates
                    if it["text"] != pdp_val and it["props"] and it["props"]["ymin"] <= 0.35
                ]
                if brand_candidates:
                    # Sort primarily by vertical position (uppermost on front) then by area
                    brand_candidates.sort(key=lambda it: (it["props"]["ymin"], -it["props"]["area"]))
                    best_brand = brand_candidates[0]
                    extracted["brand"] = {
                        "value": best_brand["text"],
                        "raw_val": best_brand["text"],
                        "confidence": best_brand["confidence"],
                        "side": side,
                        "bbox_norm": best_brand["bbox_norm"],
                        "heading": "Brand Identity",
                        "spatial_relationship": "PDP_TOP",
                        "detected": True
                    }

    # Final fallback normalization: If any declaration not detected, mark clearly without hallucinating
    for k, v in extracted.items():
        if not v["detected"]:
            v["value"] = "Not detected"
            v["raw_val"] = None
            v["confidence"] = 0.0

    return extracted
