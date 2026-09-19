import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
import io

class OCRService:
    """
    Modular OCR Service abstraction for Claro.
    Uses RapidOCR (PaddleOCR ONNX engine) for bilingual English and Devanagari (Hindi) recognition.
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

    def extract_text_with_boxes(self, image_input: Any) -> List[Dict[str, Any]]:
        """
        Runs OCR on image input (file path, bytes, or numpy array).
        Returns list of detected elements:
        [
            {
                "text": "MRP Rs. 120.00",
                "confidence": 0.98,
                "bbox_raw": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "bbox_norm": [ymin, xmin, ymax, xmax]  # normalized to 0.0 - 1.0
            }
        ]
        """
        # Load image into numpy BGR array
        img = None
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                img = cv2.imread(image_input)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            img = image_input

        if img is None:
            return []

        h, w = img.shape[:2]
        results = []

        if OCRService._engine is not None:
            try:
                ocr_out, _ = OCRService._engine(img)
                if ocr_out:
                    for item in ocr_out:
                        # item format: [box_points, text, confidence]
                        box, text, score = item
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        
                        xmin = max(0.0, min(xs) / w)
                        xmax = min(1.0, max(xs) / w)
                        ymin = max(0.0, min(ys) / h)
                        ymax = min(1.0, max(ys) / h)
                        
                        results.append({
                            "text": str(text).strip(),
                            "confidence": round(float(score), 3),
                            "bbox_raw": box,
                            "bbox_norm": [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]
                        })
            except Exception as e:
                print(f"[ERROR] OCR processing failure: {e}")

        return results

    def get_full_text(self, detections: List[Dict[str, Any]]) -> str:
        """Concatenate detected lines into a readable text block."""
        return "\n".join(d["text"] for d in detections if d.get("text"))

# Singleton instance
ocr_service = OCRService()
