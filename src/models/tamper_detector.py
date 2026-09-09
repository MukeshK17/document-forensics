"""Three-signal tamper detector for PAN card fake detection."""

import statistics

import cv2
import numpy as np


def _lap_var(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _lap_var_masked(gray: np.ndarray, mask: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    pixels = lap[mask > 0]
    return float(np.var(pixels)) if len(pixels) >= 20 else 0.0


def _edge_gradient_variance(gray_crop: np.ndarray) -> float:
    """Measure variation in stroke-edge gradients after Otsu binarization."""
    if gray_crop.shape[0] < 6 or gray_crop.shape[1] < 6:
        return -1.0

    _, binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if int((binary == 255).sum()) < 15:
        return -1.0

    sx = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)
    gm = cv2.magnitude(sx, sy)

    stroke_gradients = gm[binary == 255]
    mean_grad = np.mean(stroke_gradients)
    if mean_grad < 1e-3:
        return 0.0

    return float(np.std(stroke_gradients) / mean_grad)


def detect_regional_tampering(
    image_path: str,
    boxes: list,
    scores: list = None,
) -> dict:
    """
    Args:
        image_path: path to PAN card image
        boxes:      list of [x1,y1,x2,y2] ABSOLUTE pixel coords from PaddleOCR
        scores:     OCR confidence scores, one per box

    Returns dict with tamper_score (0-100) and suspicious_regions list.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "error": "image_load_failed",
            "tamper_score": 0,
            "suspicious_regions": [],
        }

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if scores is None:
        scores = [None] * len(boxes)

    regions = []
    for box, conf in zip(boxes, scores):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        bw, bh = x2 - x1, y2 - y1

        if bw < 10 or bh < 6:
            continue
        if bw > 0.65 * W:
            continue

        gap = min(3, bh // 4)
        ix1 = max(0, x1 + gap)
        iy1 = max(0, y1 + gap)
        ix2 = min(W, x2 - gap)
        iy2 = min(H, y2 - gap)
        inner = gray[iy1:iy2, ix1:ix2]
        inner_sharp = _lap_var(inner) if inner.size > 0 else 0.0

        pad = max(10, bh)
        nx1 = max(0, x1 - pad)
        ny1 = max(0, y1 - pad)
        nx2 = min(W, x2 + pad)
        ny2 = min(H, y2 + pad)
        collar = gray[ny1:ny2, nx1:nx2]
        cmask = np.ones(collar.shape, dtype=np.uint8) * 255
        cmask[y1 - ny1 : y2 - ny1, x1 - nx1 : x2 - nx1] = 0
        collar_sharp = _lap_var_masked(collar, cmask)
        collar_ratio = inner_sharp / (collar_sharp + 1e-6)

        crop = gray[y1:y2, x1:x2]
        edge_var = _edge_gradient_variance(crop)

        regions.append(
            {
                "box": [x1, y1, x2, y2],
                "inner_sharp": inner_sharp,
                "collar_sharp": collar_sharp,
                "collar_ratio": collar_ratio,
                "conf": float(conf) if conf is not None else None,
                "edge_var": edge_var,
            }
        )

    if not regions:
        return {"tamper_score": 0, "suspicious_regions": [], "regions_analyzed": 0}

    med_sharp = statistics.median(r["inner_sharp"] for r in regions)

    conf_vals = [r["conf"] for r in regions if r["conf"] is not None]
    med_conf = statistics.median(conf_vals) if conf_vals else None

    valid_edge = [r["edge_var"] for r in regions if r["edge_var"] >= 0]
    med_edge = statistics.median(valid_edge) if valid_edge else None

    flagged = []
    for r in regions:
        signals_fired = []
        x1, y1, x2, y2 = r["box"]

        sharp_ratio = r["inner_sharp"] / (med_sharp + 1e-6)
        if (
            r["collar_ratio"] > 5.5
            and sharp_ratio > 2.2
            and r["inner_sharp"] > 80.0
            and r["collar_sharp"] >= 4.0
        ):
            signals_fired.append(
                f"sharpness(collar={r['collar_ratio']:.1f}x peer={sharp_ratio:.1f}x)"
            )

        if med_conf is not None and r["conf"] is not None and med_conf < 0.96:
            conf_delta = r["conf"] - med_conf
            if conf_delta > 0.10 and r["conf"] > 0.97:
                signals_fired.append(
                    f"conf(delta={conf_delta:.3f} val={r['conf']:.3f} med={med_conf:.3f})"
                )

        if (
            r["edge_var"] >= 0
            and med_edge is not None
            and r["edge_var"] < 0.28
            and r["edge_var"] < med_edge * 0.75
        ):
            signals_fired.append(
                f"edge_binarization(coef_var={r['edge_var']:.3f} med={med_edge:.3f})"
            )

        if signals_fired:
            flagged.append(
                {
                    "box": r["box"],
                    "inner_sharp": round(r["inner_sharp"], 2),
                    "collar_ratio": round(r["collar_ratio"], 2),
                    "conf": round(r["conf"], 3) if r["conf"] else None,
                    "edge_var": round(r["edge_var"], 3) if r["edge_var"] >= 0 else None,
                    "signals": signals_fired,
                }
            )

    tamper_score = 0
    if len(flagged) == 1:
        tamper_score = 45
    elif len(flagged) >= 2:
        tamper_score = 70

    return {
        "tamper_score": min(tamper_score, 100),
        "suspicious_regions": flagged,
        "regions_analyzed": len(regions),
        "med_sharpness": round(med_sharp, 2),
        "med_conf": round(med_conf, 3) if med_conf else None,
        "med_edge_var": round(med_edge, 3) if med_edge else None,
    }
