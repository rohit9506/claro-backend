import os
import re
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image, ImageOps
import io

class OCRService:
    """
    Modular OCR Service abstraction for Claro.
    Uses RapidOCR (PaddleOCR ONNX engine) for bilingual English and Devanagari (Hindi) recognition.
    Supports EXIF orientation correction and auto-rotation (0°, 90°, 180°, 270°) for sideways/rotated images.
    Extracts text lines, bounding boxes, and confidence metrics.
    """
    _instance = None
    _engine = None

    def __init__(self):
        if OCRService._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                OCRService._engine = RapidOCR()
            except Exception as e:
                print(f"[WARN] RapidOCR initialization error: {e}")
                OCRService._engine = None

    @staticmethod
    def _score_detections(detections: List[Dict[str, Any]]) -> float:
        if not detections:
            return 0.0
        score = 0.0
        statutory_tokens = {
            "mrp", "rs", "inr", "net", "wt", "weight", "qty", "quantity", "g", "gm", "ml", "kg", "l",
            "batch", "lot", "mfg", "exp", "expiry", "date", "best", "before", "use",
            "manufactured", "packed", "marketed", "fssai", "license", "lic",
            "email", "care", "consumer", "phone", "toll", "free", "india", "origin",
            "brand", "unit", "price", "usp", "per", "hair", "man", "solution", "growmax", "topical",
            "ingredients", "nutrition", "address", "company", "ltd", "pvt"
        }
        for d in detections:
            t = d.get("text", "").strip()
            conf = d.get("confidence", 0.5)
            # Penalize stray CJK or replacement characters (often produced by rotated/inverted Latin text)
            if any('\u4e00' <= c <= '\u9fff' for c in t):
                score -= 8.0
                continue
            words = re.findall(r'[a-zA-Z0-9]+', t.lower())
            for w in words:
                if len(w) >= 3:
                    score += conf * 1.5
                if w in statutory_tokens:
                    score += conf * 6.0
        return score

    def _ocr_numpy(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Runs OCR on a single numpy BGR image array."""
        if img is None or OCRService._engine is None:
            return []

        h, w = img.shape[:2]
        max_dim = max(h, w)
        scale = 1.0
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_to_ocr = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_to_ocr = img

        results = []
        try:
            ocr_out, _ = OCRService._engine(img_to_ocr)
            if ocr_out:
                for item in ocr_out:
                    box, text, score = item
                    if scale != 1.0:
                        box_orig = [[p[0] / scale, p[1] / scale] for p in box]
                    else:
                        box_orig = box

                    xs = [p[0] for p in box_orig]
                    ys = [p[1] for p in box_orig]
                    
                    xmin = max(0.0, min(xs) / w)
                    xmax = min(1.0, max(xs) / w)
                    ymin = max(0.0, min(ys) / h)
                    ymax = min(1.0, max(ys) / h)
                    
                    results.append({
                        "text": str(text).strip(),
                        "confidence": round(float(score), 3),
                        "bbox_raw": box_orig,
                        "bbox_norm": [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]
                    })
        except Exception as e:
            print(f"[ERROR] OCR processing failure: {e}")

        return results

    def extract_text_with_boxes(self, image_input: Any, auto_orient: bool = True) -> List[Dict[str, Any]]:
        """
        Runs OCR on image input (file path, bytes, or numpy array).
        Automatically handles EXIF orientation and scans rotations (0°, 90°, 180°, 270°)
        if the image is rotated left, right, or upside-down.
        """
        img = None
        loaded_from_disk_path = None

        if isinstance(image_input, str):
            if os.path.exists(image_input):
                loaded_from_disk_path = image_input
                try:
                    pil_im = Image.open(image_input)
                    pil_im = ImageOps.exif_transpose(pil_im)
                    img = cv2.cvtColor(np.array(pil_im.convert("RGB")), cv2.COLOR_RGB2BGR)
                except Exception:
                    img = cv2.imread(image_input)
        elif isinstance(image_input, bytes):
            try:
                pil_im = Image.open(io.BytesIO(image_input))
                pil_im = ImageOps.exif_transpose(pil_im)
                img = cv2.cvtColor(np.array(pil_im.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                nparr = np.frombuffer(image_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            img = image_input

        if img is None:
            return []

        # 1. Base OCR at 0 degrees
        base_dets = self._ocr_numpy(img)
        if not auto_orient:
            return base_dets

        base_score = self._score_detections(base_dets)

        # Fast exit: if 0 deg already has clear packaging text or statutory tokens, return immediately
        if (base_score >= 10.0 and len(base_dets) >= 3) or base_score >= 15.0 or len(base_dets) >= 6:
            return base_dets

        # 2. Fast Thumbnail Orientation Check for sideways or inverted images
        # Evaluates candidate rotations (90°, 270°, 180°) on a downscaled 380px thumbnail in ~0.04s
        h, w = img.shape[:2]
        thumb_scale = 380.0 / max(h, w)
        thumb_img = cv2.resize(img, (int(w * thumb_scale), int(h * thumb_scale)), interpolation=cv2.INTER_AREA)
        
        best_thumb_score = self._score_detections(self._ocr_numpy(thumb_img))
        best_angle = 0
        rot_flags = {
            90: cv2.ROTATE_90_CLOCKWISE,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180
        }

        for angle in [90, 270, 180]:
            r_thumb = cv2.rotate(thumb_img, rot_flags[angle])
            cand_thumb_dets = self._ocr_numpy(r_thumb)
            cand_thumb_score = self._score_detections(cand_thumb_dets)
            if cand_thumb_score > best_thumb_score * 1.3 and cand_thumb_score >= best_thumb_score + 2.5:
                best_thumb_score = cand_thumb_score
                best_angle = angle

        # If 0° is best or rotation showed no significant improvement, keep base detections
        if best_angle == 0:
            return base_dets

        # 3. Exactly ONE full-res OCR pass on the winning rotation
        best_rot_flag = rot_flags[best_angle]
        best_img = cv2.rotate(img, best_rot_flag)
        best_dets = self._ocr_numpy(best_img)

        # Overwrite disk file so previews, evidence crops, and PDF reports display upright packaging
        if loaded_from_disk_path is not None:
            try:
                cv2.imwrite(loaded_from_disk_path, best_img)
            except Exception as write_err:
                print(f"[WARN] Could not update rotated image on disk: {write_err}")

        return best_dets

    def get_full_text(self, detections: List[Dict[str, Any]]) -> str:
        """Concatenate detected lines into a readable text block."""
        return "\n".join(d["text"] for d in detections if d.get("text"))

# Singleton instance
ocr_service = OCRService()
