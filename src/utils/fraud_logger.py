import json
import logging
from datetime import datetime
from pathlib import Path

# Setup structured logger
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("fraud_detection")
logger.setLevel(logging.DEBUG)

# File handler — JSON lines format
file_handler = logging.FileHandler(
    LOG_DIR / f"fraud_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
)
file_handler.setLevel(logging.DEBUG)

# Console handler — human readable
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def log_fraud_assessment(
    doc_id: str,
    label: str,
    ocr_score: int,
    cv_score: int,
    combined_score: int,
    verdict: str,
    violations: list,
    cv_issues: list,
    suspicious_regions: list,
    avg_confidence: float,
    image_path: str = "",
):
    """
    Log a complete fraud assessment with all signals.
    Written as JSON lines for easy parsing later.
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "doc_id": doc_id,
        "label": label,
        "verdict": verdict,
        "scores": {
            "ocr_score": ocr_score,
            "cv_score": cv_score,
            "combined_score": combined_score,
            "threshold": 30,
            "threshold_crossed": combined_score >= 30,
        },
        "ocr_signals": {
            "violations": violations,
            "avg_confidence": avg_confidence,
            "violation_count": len(violations),
        },
        "cv_signals": {
            "issues": cv_issues,
        },
        "tamper_signals": {
            "suspicious_regions": suspicious_regions,
            "region_count": len(suspicious_regions),
        },
        "image_path": image_path,
        "explanation": _build_explanation(
            violations,
            cv_issues,
            suspicious_regions,
            ocr_score,
            cv_score,
            combined_score,
        ),
    }

    # Write JSON line to file
    file_handler.stream.write(json.dumps(record) + "\n")
    file_handler.stream.flush()

    # Human readable console output for suspicious docs
    if verdict == "SUSPICIOUS":
        logger.info(
            f"SUSPICIOUS | {doc_id} | score={combined_score} | "
            f"ocr={ocr_score} cv={cv_score} | "
            f"violations={violations} | "
            f"regions={len(suspicious_regions)}"
        )

    return record


def _build_explanation(
    violations, cv_issues, suspicious_regions, ocr_score, cv_score, combined_score
) -> str:
    """
    Build human-readable explanation of why document was flagged.
    This is what you show in the demo.
    """
    reasons = []

    # OCR reasons
    reason_map = {
        "missing_pan_number": "No valid PAN number found in document",
        "pan_ocr_correction_needed": "PAN number required OCR error correction",
        "invalid_pan_structure": "PAN number fails structural validation",
        "known_test_pan_number": "PAN number matches known dummy/test value",
        "multiple_different_pan_numbers": "Multiple different PAN numbers detected",
        "missing_date": "No date of birth found",
        "invalid_date": "Date of birth is impossible or invalid",
        "missing_required_keywords": "Required government keywords absent",
        "low_ocr_confidence": "Overall OCR confidence too low",
        "high_low_confidence_ratio": "High proportion of unreadable text",
        "confidence_outlier_detected": "Unusual confidence drop in specific regions",
    }

    for v in violations:
        if v in reason_map:
            reasons.append(f"[OCR] {reason_map[v]}")

    # CV reasons
    for issue in cv_issues:
        if "suspicious_screen_resolution" in issue:
            res = issue.split(":")[-1] if ":" in issue else "unknown"
            reasons.append(
                f"[CV] Image resolution {res} matches "
                f"known screenshot/template dimensions"
            )
        elif "uneven_noise_distribution" in issue:
            reasons.append(
                "[CV] Uneven noise distribution — possible image splicing or composite"
            )
        elif "high_ela_peak" in issue:
            reasons.append(
                "[CV] Error Level Analysis detected "
                "regions with different compression history"
            )
        elif "very_low_resolution" in issue:
            reasons.append("[CV] Image resolution too low for genuine submission")

    # Tamper reasons
    for region in suspicious_regions:
        box = region.get("box", [])
        ratio = region.get("ratio_to_bg", region.get("ratio_to_avg", 0))
        reasons.append(
            f"[TAMPER] Suspicious region at {box} — "
            f"sharpness {ratio:.1f}x above document baseline"
        )

    if not reasons:
        return "No specific violations detected"

    return " | ".join(reasons)


def log_batch_summary(results: list):
    """Log end-of-run summary statistics."""
    total = len(results)
    suspicious = sum(1 for r in results if r["verdict"] == "SUSPICIOUS")
    fake_caught = sum(
        1 for r in results if r["label"] == "FAKE_PAN" and r["verdict"] == "SUSPICIOUS"
    )
    false_positives = sum(
        1 for r in results if r["label"] == "PAN_CARD" and r["verdict"] == "SUSPICIOUS"
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "type": "batch_summary",
        "total_documents": total,
        "suspicious_flagged": suspicious,
        "fake_correctly_caught": fake_caught,
        "real_wrongly_flagged": false_positives,
    }

    file_handler.stream.write(json.dumps(summary) + "\n")
    file_handler.stream.flush()
    logger.info(
        f"BATCH COMPLETE | total={total} | "
        f"suspicious={suspicious} | "
        f"fake_caught={fake_caught} | "
        f"false_positives={false_positives}"
    )
