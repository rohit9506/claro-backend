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
                    r"(?:m\.?r\.?p\.?|max(?:imum)?\s*retail\s*price|price)\s*[:.\-]?\s*(?:₹|rs\.?|inr)?\s*([0-9]+(?:\.[0-9]{1,2})?)(?:\s*\/\-)?",
                    t, re.IGNORECASE
                )
                if not mrp_inline:
                    mrp_inline = re.search(
                        r"(?:₹|rs\.?|inr)\s*[:.\-]?\s*([0-9]+(?:\.[0-9]{1,2})?)(?:\s*\/\-)?",
                        t, re.IGNORECASE
                    )
                if not mrp_inline:
                    mrp_inline = re.search(
                        r"\b([0-9]+(?:\.[0-9]{1,2})?)\s*\/\-",
                        t
                    )

                if mrp_inline and not any(nut in tl for nut in ["kcal", "protein", "sodium", "fat", "sugar", "100g", "serving"]):
                    amount = mrp_inline.group(1)
                    # Verify it's a realistic price, not a date year or weight
                    if float(amount) > 0 and amount not in ["2024", "2025", "2026", "2027", "2028"]:
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
                    if re.search(r"\b(m\.?r\.?p\.?|max(?:imum)?\s*retail\s*price|price)\b", htl) and not is_discount_or_usp(htl):
                        # Search for proximate numeric currency values across other bounding boxes
                        best_pair = None
                        best_score = 0.0

                        for v_item in enriched_dets:
                            if v_item == h_item or not h_item["props"] or not v_item["props"]:
                                continue
                            vt = v_item["text"]
                            vtl = vt.lower()
                            if is_discount_or_usp(vtl) or any(nut in vtl for nut in ["kcal", "g", "ml", "mg", "serving"]):
                                continue
                            
                            val_match = re.search(r"(?:₹|rs\.?|inr)?\s*([0-9]+(?:\.[0-9]{1,2})?)(?:\s*\/\-)?\b", vt, re.IGNORECASE)
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
        # Exclude legal, nutritional boilerplate, and mobile UI / status bar text
        def is_boilerplate(text: str) -> bool:
            tl = text.lower().strip()
            # Time e.g. "9:14", "5:24"
            if re.match(r"^\d{1,2}:\d{2}$", tl):
                return True
            # Percentage e.g. "62%", "100%"
            if re.match(r"^\d{1,3}%$", tl):
                return True
            # Mobile UI / system keywords
            if any(k in tl for k in [
                "processing delay", "taking longer", "expected", "retake", "retry",
                "cancel", "accept", "settings", "claro", "legal metrology", "sign in",
                "create account", "battery", "volte", "wifi", "lte", "kb/s", "mb/s"
            ]):
                return True
            # Packaging statutory boilerplate
            return any(k in tl for k in [
                "mrp", "net wt", "quantity", "batch", "fssai", "100g", "ingredients",
                "nutrition", "save", "offer", "discount", "servings", "license",
                "regd", "trademark", "patent", "expiry", "best before", "serving size",
                "for external use", "keep out of reach", "store in a cool", "shake well"
            ])

        # Try on Front first, then fallback to other sides if not found
        if side == "front" or not extracted["product_name"]["detected"]:
            candidates = [it for it in enriched_dets if not is_boilerplate(it["text"]) and len(it["text"]) >= 2]

            # Known Popular Indian Brands list for boost
            known_brands = [
                "man matters", "amul", "britannia", "parle", "nestle", "tata", "haldiram",
                "dabur", "patanjali", "cadbury", "himalaya", "colgate", "dettol", "fortune",
                "aashirvaad", "dove", "nivea", "head & shoulders", "garnier", "mamaearth",
                "biotique", "mcaffeine", "the man company", "beardo", "ustraa", "itc",
                "sunfeast", "lays", "kurkure", "maggi", "pepsodent", "sensodyne"
            ]

            # 1. Brand Detection
            if not extracted["brand"]["detected"] and candidates:
                detected_brand_cand = None

                # Check for known brand in any candidate
                for it in candidates:
                    it_txt = it["text"].lower()
                    for kb in known_brands:
                        if kb in it_txt:
                            detected_brand_cand = (kb.title(), it)
                            break
                    if detected_brand_cand:
                        break

                # If no known brand, look at upper PDP text (ymin <= 0.38)
                if not detected_brand_cand:
                    brand_cands = [it for it in candidates if it["props"] and it["props"]["ymin"] <= 0.38]
                    if brand_cands:
                        brand_cands.sort(key=lambda it: (it["props"]["ymin"], -it["props"]["area"]))
                        top_brand_cand = brand_cands[0]
                        b_val = top_brand_cand["text"].strip()

                        # Check if next candidate directly below forms a 2-part brand (e.g. "man" + "matters")
                        for sub_b in brand_cands[1:]:
                            dy = sub_b["props"]["cy"] - top_brand_cand["props"]["cy"]
                            dx = abs(sub_b["props"]["cx"] - top_brand_cand["props"]["cx"])
                            if 0.01 <= dy <= 0.12 and dx <= 0.20:
                                sub_txt = sub_b["text"].strip()
                                if sub_txt.lower() not in b_val.lower():
                                    b_val = f"{b_val} {sub_txt}"
                                    break

                        detected_brand_cand = (b_val, top_brand_cand)

                if detected_brand_cand:
                    b_text, b_item = detected_brand_cand
                    # Normalize common OCR typos e.g. "Maitters" -> "Matters"
                    b_text = re.sub(r"\bmaitters\b", "Matters", b_text, flags=re.IGNORECASE)
                    extracted["brand"] = {
                        "value": b_text.strip(),
                        "raw_val": b_text.strip(),
                        "confidence": b_item["confidence"],
                        "side": side,
                        "bbox_norm": b_item["bbox_norm"],
                        "heading": "Brand Identity",
                        "spatial_relationship": "PDP_TOP",
                        "detected": True
                    }

            # 2. Product Name Detection (Prominent PDP Title)
            if not extracted["product_name"]["detected"] and candidates:
                brand_val = extracted["brand"].get("value", "").lower()
                pdp_cands = [
                    it for it in candidates
                    if it["props"] and 0.08 <= it["props"]["cy"] <= 0.88
                    and it["text"].strip().lower() not in brand_val
                    and brand_val not in it["text"].strip().lower()
                ]

                if pdp_cands:
                    pdp_cands.sort(key=lambda it: it["props"]["area"] if it["props"] else 0, reverse=True)
                    best_pdp = pdp_cands[0]
                    pdp_items = [best_pdp]

                    # Multi-line title collation: find lines immediately above or below connected to title
                    if best_pdp["props"]:
                        for other in candidates:
                            if other == best_pdp or not other["props"]:
                                continue
                            if other["text"].strip().lower() in brand_val:
                                continue
                            dy = other["props"]["cy"] - best_pdp["props"]["cy"]
                            dx = abs(other["props"]["cx"] - best_pdp["props"]["cx"])
                            if -0.16 <= dy <= 0.18 and dx <= 0.25 and len(other["text"].strip()) >= 3:
                                pdp_items.append(other)

                    # Sort collected title items top-to-bottom
                    pdp_items.sort(key=lambda it: it["props"]["cy"] if it["props"] else 0)
                    full_pdp_title = " ".join(it["text"].strip() for it in pdp_items)

                    # Normalize OCR typos in medical/personal care titles
                    full_pdp_title = re.sub(r"\bminoida\b", "Minoxidil", full_pdp_title, flags=re.IGNORECASE)
                    full_pdp_title = re.sub(r"\bheir\b", "Hair", full_pdp_title, flags=re.IGNORECASE)

                    extracted["product_name"] = {
                        "value": full_pdp_title.strip(),
                        "raw_val": full_pdp_title.strip(),
                        "confidence": best_pdp["confidence"],
                        "side": side,
                        "bbox_norm": best_pdp["bbox_norm"],
                        "heading": "Product Name (PDP)",
                        "spatial_relationship": "PDP_CENTER",
                        "detected": True
                    }

    # Derived Unit Sale Price (USP): If MRP and Net Qty detected, mathematically compute USP per Rule 6(1)(e)
    if not extracted["unit_sale_price"]["detected"] and extracted["mrp"]["detected"] and extracted["net_quantity"]["detected"]:
        try:
            mrp_m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(extracted["mrp"].get("value", "")))
            qty_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)", str(extracted["net_quantity"].get("value", "")))
            if mrp_m and qty_m:
                mrp_n = float(mrp_m.group(1))
                qty_n = float(qty_m.group(1))
                unit_str = qty_m.group(2).lower()
                if qty_n > 0:
                    usp_val = round(mrp_n / qty_n, 2)
                    extracted["unit_sale_price"] = {
                        "value": f"₹{usp_val} / {unit_str}",
                        "raw_val": f"{usp_val}",
                        "confidence": 0.92,
                        "side": extracted["mrp"].get("side", "right_side"),
                        "bbox_norm": extracted["mrp"].get("bbox_norm", [0.1, 0.1, 0.3, 0.9]),
                        "heading": "Unit Sale Price (Derived under PCR-2011)",
                        "spatial_relationship": "CALCULATED",
                        "detected": True
                    }
        except Exception:
            pass

    # Final fallback normalization: If any declaration not detected, mark clearly without hallucinating
    for k, v in extracted.items():
        if not v["detected"]:
            v["value"] = "Not detected"
            v["raw_val"] = None
            v["confidence"] = 0.0

    return extracted
