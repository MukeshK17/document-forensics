# #     main()
# from __future__ import annotations
# import argparse
# import json
# import re
# import sys
# from pathlib import Path
# from typing import Any
# import cv2
# import numpy as np

# ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(ROOT))

# PAN_TEMPLATE_HEADERS = {
#     "INCOME", "TAX", "DEPARTMENT", "GOVT", "INDIA", "GOVERNMENT",
#     "ACCOUNT", "NUMBER", "CARD", "MINISTRY", "PERMANENT", "NAME", "FATHER",
#     "BIRTH", "DATE", "OF", "THE", "SIGNATURE", "PERMANENT ACCOUNT CARD"
# }

# def clamp(v: int, lo: int, hi: int) -> int:
#     return max(lo, min(hi, v))

# def expand_box(box: list[int], pad_px: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
#     return (
#         clamp(box[0] - pad_px, 0, img_w - 1),
#         clamp(box[1] - pad_px, 0, img_h - 1),
#         clamp(box[2] + pad_px, 1, img_w),
#         clamp(box[3] + pad_px, 1, img_h),
#     )

# def analyze_document(image_path: Path, ocr_entry: dict[str, Any]) -> dict[str, Any]:
#     img = cv2.imread(str(image_path))
#     if img is None:
#         return {"forensic_risk_score": 0, "verdict": "CLEAN", "cv_issues": "image_load_failed"}

#     h, w = img.shape[:2]
#     words = ocr_entry.get("words", [])
#     boxes = ocr_entry.get("boxes", [])

#     forensic_risk = 0
#     cv_issues = []
#     white_overlay_flags = 0

#     # Step 1: Scan user content text pixels for illegal white mask covers
#     for word, box in zip(words, boxes):
#         raw_text = word.strip()
#         if not raw_text or len(raw_text) <= 1:
#             continue

#         upper_clean = raw_text.upper()

#         # Skip standard background label fields
#         if upper_clean in PAN_TEMPLATE_HEADERS:
#             continue
#         if any(h_word in upper_clean for h_word in ["INCOME", "TAX", "DEPARTMENT", "GOVT", "INDIA"]):
#             continue
#         if re.search(r"[\u0900-\u097F]", raw_text): # Ignore Hindi noise
#             continue
#         if not any(c.isalpha() for c in raw_text): # Skip standalone counts or periods
#             continue

#         # Target Pixel analysis inside field boundaries
#         x1, y1, x2, y2 = expand_box(box, 2, w, h)
#         if (x2 - x1) < 5 or (y2 - y1) < 5:
#             continue

#         crop = img[y1:y2, x1:x2]
#         gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#         # Isolate text content foreground channel values via Otsu
#         _, mask = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         text_pixels = crop[mask == 255]

#         if len(text_pixels) >= 5:
#             avg_b, avg_g, avg_r = np.mean(text_pixels, axis=0)
#             # Flag if fields are suspiciously covered with a bright white text block (All channels > 210)
#             if avg_r > 210 and avg_g > 210 and avg_b > 210:
#                 white_overlay_flags += 1
#                 cv_issues.append(f"white_text_overlay_detected:{raw_text}")

#     # Step 2: Clear Face Count Validation Checklist
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

#     if len(faces) == 0:
#         forensic_risk += 45
#         cv_issues.append("no_photo_detected")

#     if white_overlay_flags > 0:
#         forensic_risk += 45

#     return {
#         "doc_id": ocr_entry.get("doc_id", ""),
#         "forensic_risk_score": min(forensic_risk, 100),
#         "verdict": "SUSPICIOUS" if forensic_risk >= 30 else "CLEAN",
#         "cv_issues": "|".join(cv_issues)
#     }

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Boilerplate headers that are part of the printed template, not user data
PAN_TEMPLATE_HEADERS = {
    "INCOME",
    "TAX",
    "DEPARTMENT",
    "GOVT",
    "INDIA",
    "GOVERNMENT",
    "ACCOUNT",
    "NUMBER",
    "CARD",
    "MINISTRY",
    "PERMANENT",
    "NAME",
    "FATHER",
    "BIRTH",
    "DATE",
    "OF",
    "THE",
    "SIGNATURE",
    "PERMANENT ACCOUNT CARD",
}


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def expand_box(
    box: list[int], pad_px: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    return (
        clamp(box[0] - pad_px, 0, img_w - 1),
        clamp(box[1] - pad_px, 0, img_h - 1),
        clamp(box[2] + pad_px, 1, img_w),
        clamp(box[3] + pad_px, 1, img_h),
    )


def is_inside_box(x: float, y: float, box: list[int]) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def compute_laplacian_variance(region: np.ndarray, mask: np.ndarray = None) -> float:
    """Computes edge sharpness. Optionally uses a mask to calculate specific pixels."""
    if region.size == 0:
        return 0.0
    gray = (
        cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    )
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    if mask is not None:
        valid_pixels = lap[mask > 0]
        return float(np.var(valid_pixels)) if len(valid_pixels) > 10 else 0.0
    return float(lap.var())


def analyze_document(image_path: Path, ocr_entry: dict[str, Any]) -> dict[str, Any]:
    """
    Visual Forensics Engine:
    - Detects Solid Color/White text box overlays (Zero background variance)
    - Validates 1 or 2 Face counts
    - Detects High-Resolution Pasted Photos (Sharpness Mismatch)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "forensic_risk_score": 0,
            "verdict": "CLEAN",
            "cv_issues": "image_load_failed",
        }

    h, w = img.shape[:2]
    words = ocr_entry.get("words", [])
    boxes = ocr_entry.get("boxes", [])

    forensic_risk = 0
    cv_issues = []

    # Build signature exclusion zone to prevent false flags on handwritten cursives
    sig_exclusion_zone = None
    for word, box in zip(words, boxes):
        if "SIGN" in word.upper() or "SURE" in word.upper():
            sig_exclusion_zone = [
                clamp(box[0] - int(w * 0.05), 0, w),
                clamp(box[1] - int(h * 0.16), 0, h),
                clamp(box[2] + int(w * 0.05), 0, w),
                clamp(box[3] + int(h * 0.02), 0, h),
            ]
            break

    solid_overlay_flags = 0
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # =========================================================================
    # DETECTOR 1: SOLID TEXT BOX OVERLAYS (PowerPoint Backgrounds)
    # =========================================================================
    for word, box in zip(words, boxes):
        raw_text = word.strip()
        if not raw_text or len(raw_text) <= 1:
            continue

        upper_clean = raw_text.upper()

        # Skip template headers and Hindi translations
        if upper_clean in PAN_TEMPLATE_HEADERS or re.search(
            r"[\u0900-\u097F]", raw_text
        ):
            continue
        if any(
            h_word in upper_clean
            for h_word in ["INCOME", "TAX", "DEPARTMENT", "GOVT", "INDIA"]
        ):
            continue

        x_center = (box[0] + box[2]) / 2.0
        y_center = (box[1] + box[3]) / 2.0

        if sig_exclusion_zone and is_inside_box(x_center, y_center, sig_exclusion_zone):
            continue

        # Extract local region
        x1, y1, x2, y2 = expand_box(box, 3, w, h)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        crop = img[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Separate text strokes from the background paper
        _, text_mask = cv2.threshold(
            gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # The background is everything that is NOT text
        bg_mask = cv2.bitwise_not(text_mask)
        bg_pixels = gray_crop[bg_mask == 255]

        if len(bg_pixels) >= 15:
            bg_variance = np.var(bg_pixels)
            avg_color = np.mean(crop[bg_mask == 255], axis=0)  # BGR

            # Anomaly: A real card has paper texture/noise (variance > 10).
            # A pasted PowerPoint text box is a perfectly flat digital color (variance near 0).
            if bg_variance < 3.0:
                solid_overlay_flags += 1
                cv_issues.append(f"solid_digital_box_overlay:{raw_text}")

            # White Box Cover-up Check (Very bright background fills)
            elif avg_color[0] > 220 and avg_color[1] > 220 and avg_color[2] > 220:
                solid_overlay_flags += 1
                cv_issues.append(f"white_mask_overlay_detected:{raw_text}")

            # Black ink color consistency check
            text_pixels_hsv = hsv_img[y1:y2, x1:x2][text_mask == 255]
            if len(text_pixels_hsv) >= 5:
                # Check for non-black ink (Saturation/Brightness check)
                avg_h, avg_s, avg_v = np.mean(text_pixels_hsv, axis=0)
                if (avg_s > 45 and avg_v > 100) or (avg_v > 200):
                    solid_overlay_flags += 1
                    cv_issues.append(f"invalid_ink_color_detected:{raw_text}")

    # =========================================================================
    # DETECTOR 2 & 3: FACE COUNT & PASTED PHOTO SHARPNESS
    # =========================================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45)
    )

    face_count = len(faces)
    pasted_photos_detected = 0

    # Rule 2: Exactly 1 or 2 faces is legally acceptable. 0 or 3+ is fraud.
    if face_count == 0:
        forensic_risk += 45
        cv_issues.append("no_photo_detected")
    elif face_count > 2:
        forensic_risk += 45
        cv_issues.append(f"excessive_faces_detected:{face_count}")

    # Rule 4: Pasted Photo Sharpness Analysis
    # A pasted photo will have entirely different optical sharpness than the card it was pasted onto.
    for x, y, fw, fh in faces:
        # Calculate sharpness of the actual face region
        face_roi = gray[y : y + fh, x : x + fw]
        face_sharpness = compute_laplacian_variance(face_roi)

        # Extract a 25px local collar around the photo
        pad = 25
        nx1, ny1 = max(0, x - pad), max(0, y - pad)
        nx2, ny2 = min(w, x + fw + pad), min(h, y + fh + pad)
        collar_roi = gray[ny1:ny2, nx1:nx2]

        # Mask out the photo to calculate the sharpness of the immediate background ONLY
        mask = np.ones(collar_roi.shape, dtype=np.uint8) * 255
        ix1, iy1 = x - nx1, y - ny1
        ix2, iy2 = x + fw - nx1, y + fh - ny1
        mask[iy1:iy2, ix1:ix2] = 0

        collar_sharpness = compute_laplacian_variance(collar_roi, mask)

        # Avoid division by zero
        if collar_sharpness < 2.0:
            continue

        ratio = face_sharpness / collar_sharpness

        # Authentic photos blend slightly with the card print.
        # If the photo is 4.0x sharper than the card background right next to it, it was pasted digitally.
        if ratio > 4.0 and face_sharpness > 50.0:
            pasted_photos_detected += 1
            cv_issues.append("pasted_high_res_photo_detected")

    # =========================================================================
    # FINAL METRICS AGGREGATION
    # =========================================================================
    if solid_overlay_flags > 0:
        forensic_risk += 45
    if pasted_photos_detected > 0:
        forensic_risk += 50

    return {
        "doc_id": ocr_entry.get("doc_id", ""),
        "forensic_risk_score": min(forensic_risk, 100),
        "verdict": "SUSPICIOUS" if forensic_risk >= 30 else "CLEAN",
        "cv_issues": "|".join(cv_issues),
    }
