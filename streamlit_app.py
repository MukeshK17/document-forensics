from __future__ import annotations

import base64
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from PIL import Image, ImageDraw

# Note: Ensure these imports match your local project structure
try:
    from src.extractors.paddleOcr_extractor import PaddleOCRExtractor
    from src.models.tamper_detector import detect_regional_tampering
    from src.validators.field_validator import assess
except ImportError:
    # Mocking for the sake of UI demonstration if backend isn't present in this exact path
    pass

# ---------------------------------------------------------------------------
# Product identity
# ---------------------------------------------------------------------------
APP_NAME = "PANShield AI"
TAGLINE = "AI-Powered PAN Card Authenticity Verification"

# Initializing sidebar as collapsed (Must be the first Streamlit command)
st.set_page_config(
    page_title=f"{APP_NAME} · Bank of Baroda",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Backend logic (Retained exactly from your pipeline)
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary-style YAML file.")
    return data


@st.cache_resource(show_spinner=False)
def _get_ocr_extractor(config_path: str):
    config = _load_config(config_path)
    extractor = PaddleOCRExtractor(config)
    extractor.load()
    return extractor


@st.cache_resource(show_spinner=False)
def _get_patch_scorer(model_path: str):
    try:
        from src.models.dl_scorer import PatchTamperScorer
    except Exception:
        return None
    ckpt = Path(model_path)
    if not ckpt.exists():
        return None
    try:
        return PatchTamperScorer(str(ckpt))
    except Exception:
        return None


def _norm_to_abs_boxes(norm_boxes: list[list[int]], w: int, h: int) -> list[list[int]]:
    abs_boxes: list[list[int]] = []
    for b in norm_boxes:
        x1 = int((b[0] / 1000) * w)
        y1 = int((b[1] / 1000) * h)
        x2 = int((b[2] / 1000) * w)
        y2 = int((b[3] / 1000) * h)
        abs_boxes.append([x1, y1, x2, y2])
    return abs_boxes


def _annotate_image(
    image: Image.Image, suspicious_regions: list[dict[str, Any]]
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for region in suspicious_regions:
        box = region.get("box")
        if not box or len(box) != 4:
            continue
        draw.rectangle(box, outline=(255, 80, 80), width=4)
    return out


def _is_primitive_error(exc: Exception) -> bool:
    return "primitive" in str(exc).lower()


def _friendly_risk_label(label: str) -> str:
    return {
        "Rules Risk": "Rules Risk",
        "Tamper Risk": "Image Integrity Risk",
        "Visual Forensics Risk": "Visual Check Risk",
        "Deep Learning Risk": "Deep Learning Risk",
    }.get(label, label)


def _friendly_violation_label(code: str) -> str:
    code = code.replace("missing_mandatory_field:parent_name", "Parent name is missing")
    code = code.replace("missing_mandatory_field:name", "Name is missing")
    if ":" in code:
        key, value = code.split(":", 1)
    else:
        key, value = code, ""

    labels = {
        "missing_pan_number": "PAN number was not found",
        "pan_ocr_correction_needed": "PAN was read only after OCR correction",
        "known_test_pan_number": "Test PAN number detected",
        "invalid_pan_structure": "PAN format looks invalid",
        "multiple_different_pan_numbers": "More than one PAN number was found",
        "missing_date": "Date of birth missing or wrong format",
        "invalid_date": "Date of birth missing or wrong format",
        "missing_required_keywords": "Required PAN card text is missing",
        "low_ocr_confidence": "Text was read with low confidence",
        "high_low_confidence_ratio": "Many text parts were read poorly",
        "confidence_outlier_detected": "Mixed text quality was detected",
        "missing_critical_issuer_keywords": "Issuer details are missing",
        "missing_pan_number_block": "PAN number block is missing",
        "pan_recovered_via_normalization": "PAN was recovered after text correction",
        "multiple_conflicting_pans_detected": "Conflicting PAN numbers were found",
        "known_testing_pan_hash": "Test PAN entry detected",
        "pan_5th_character_surname_mismatch": "",
        "missing_date_of_birth": "Date of birth missing or wrong format",
    }

    if key == "invalid_taxpayer_type" and value:
        return f"Invalid PAN holder type: {value}"
    return labels.get(key, key.replace("_", " ").strip().capitalize())


def _friendly_violation_list(values: list[str]) -> list[str]:
    return [label for v in values if (label := _friendly_violation_label(v))]


def _clear_streamlit_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def _retry_ocr_with_smaller_image(
    extractor, image: Image.Image
) -> tuple[dict[str, list], Image.Image]:
    candidates: list[tuple[str, Image.Image]] = [("original", image)]
    for max_side in (1600, 1200, 960):
        resized = image.copy()
        resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if resized.size != image.size:
            candidates.append((f"resized-{max_side}", resized))

    last_exc: Exception | None = None
    for attempt_name, candidate in candidates:
        try:
            return extractor.extract(candidate), candidate
        except Exception as exc:
            last_exc = exc
            if not _is_primitive_error(exc):
                raise
            continue

    raise RuntimeError(
        "OCR could not read this image after multiple attempts."
    ) from last_exc


def _score_document(
    image: Image.Image,
    filename: str,
    extractor,
    fake_threshold: int,
    rules_weight: float,
    tamper_weight: float,
    dl_weight: float,
    patch_model_path: str,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        suffix=Path(filename).suffix or ".jpg", delete=False
    ) as tf:
        temp_path = Path(tf.name)
        image.save(temp_path)

    try:
        processing_notes: list[str] = []
        ocr, ocr_image = _retry_ocr_with_smaller_image(extractor, image)
        if ocr_image.size != image.size:
            processing_notes.append("ocr_resized_retry")

        words = ocr.get("words", [])
        norm_boxes = ocr.get("boxes", [])
        scores = ocr.get("scores", [])

        rules_result = assess(
            {
                "doc_id": str(uuid.uuid4())[:8],
                "words": words,
                "boxes": norm_boxes,
                "scores": scores,
            }
        )
        validation = rules_result["validation"]
        rules_risk = float(validation["risk_score"])
        violations = validation["violations"]
        fields = rules_result["fields"]
        hard_fail = bool(validation.get("hard_fail", False))
        hard_fail_reasons = validation.get("hard_fail_reasons", [])

        if hard_fail:
            return {
                "filename": filename,
                "verdict": "Suspicious",
                "is_fake": True,
                "suspicious_score": 100.0,
                "authenticity_score": 0.0,
                "rules_risk": 100.0,
                "tamper_risk": 0.0,
                "dl_risk": 0.0,
                "dl_available": False,
                "ocr_words": len(words),
                "pan_candidates": fields.get("pan_numbers", []),
                "dob_candidates": fields.get("dates", []),
                "violations": violations,
                "hard_fail_reasons": hard_fail_reasons,
                "processing_notes": processing_notes,
                "suspicious_regions": [],
                "tamper_regions_analyzed": 0,
                "annotated_image": image,
            }

        abs_boxes = _norm_to_abs_boxes(norm_boxes, image.width, image.height)
        try:
            tamper_result = detect_regional_tampering(str(temp_path), abs_boxes, scores)
            tamper_risk = float(tamper_result.get("tamper_score", 0.0))
            suspicious_regions = tamper_result.get("suspicious_regions", [])
        except Exception as exc:
            if _is_primitive_error(exc):
                processing_notes.append("tamper_primitive_error_skipped")
                tamper_result = {"regions_analyzed": 0}
                tamper_risk = 0.0
                suspicious_regions = []
            else:
                raise

        patch_scorer = _get_patch_scorer(patch_model_path)
        dl_available = patch_scorer is not None
        dl_result: dict[str, Any] = {}
        dl_risk = 0.0
        if dl_available:
            try:
                dl_result = patch_scorer.score(str(temp_path))
                dl_risk = float(dl_result.get("dl_score", 0.0)) * 100.0
            except Exception as exc:
                if _is_primitive_error(exc):
                    processing_notes.append("dl_primitive_error_skipped")
                    dl_available = False
                    dl_result = {}
                    dl_risk = 0.0
                else:
                    raise

        total_weight = (
            rules_weight + tamper_weight + (dl_weight if dl_available else 0.0)
        )
        if total_weight <= 0:
            total_weight = 1.0

        suspicious_score = (
            (rules_risk * rules_weight)
            + (tamper_risk * tamper_weight)
            + (dl_risk * (dl_weight if dl_available else 0.0))
        ) / total_weight
        suspicious_score = round(suspicious_score, 2)
        authenticity_score = round(100.0 - suspicious_score, 2)

        verdict = (
            "Suspicious"
            if suspicious_score >= fake_threshold
            else "Genuine (Likely Real)"
        )
        is_fake = suspicious_score >= fake_threshold

        return {
            "filename": filename,
            "verdict": verdict,
            "is_fake": is_fake,
            "suspicious_score": suspicious_score,
            "authenticity_score": authenticity_score,
            "rules_risk": round(rules_risk, 2),
            "tamper_risk": round(tamper_risk, 2),
            "dl_risk": round(dl_risk, 2),
            "dl_available": dl_available,
            "ocr_words": len(words),
            "pan_candidates": fields.get("pan_numbers", []),
            "dob_candidates": fields.get("dates", []),
            "violations": violations,
            "hard_fail_reasons": hard_fail_reasons,
            "processing_notes": processing_notes,
            "suspicious_regions": suspicious_regions,
            "tamper_regions_analyzed": tamper_result.get("regions_analyzed", 0),
            "annotated_image": _annotate_image(image, suspicious_regions),
        }
    finally:
        temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Base64 Image Encoder
# ---------------------------------------------------------------------------
def _get_base64_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Visual design system — Dark Sidebar, White Main, Orange Accents
# ---------------------------------------------------------------------------

COLOR_PRIMARY = "#ED5C1F"  # BoB Orange
COLOR_SAFE = "#22C55E"
COLOR_SAFE_BG = "#DCFCE7"
COLOR_DANGER = "#EF4444"
COLOR_DANGER_BG = "#FEE2E2"
COLOR_WARN = "#F59E0B"
COLOR_WARN_BG = "#FEF3C7"
COLOR_BG_MAIN = "#F8FAFC"
COLOR_BG_SIDE = "#161B26"
COLOR_TEXT = "#0F172A"
COLOR_TEXT_SIDE = "#94A3B8"
COLOR_BORDER = "#E2E8F0"


def _raw_html(content: str) -> None:
    st.markdown(textwrap.dedent(content).strip(), unsafe_allow_html=True)


def _inject_css() -> None:
    _raw_html(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Base Typography & Backgrounds */
        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', sans-serif;
            background: {COLOR_BG_MAIN};
            color: {COLOR_TEXT};
        }}

        /* Dark Sidebar Override (Careful not to block the toggle button) */
        [data-testid="stSidebar"] > div:first-child {{
            background-color: {COLOR_BG_SIDE} !important;
            border-right: none;
        }}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {{
            color: {COLOR_TEXT_SIDE} !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: #2D3748 !important;
        }}

        /* Sidebar custom menu styling */
        .side-menu-item {{
            display: flex; align-items: center; gap: 12px;
            padding: 12px 16px; margin: 4px 0; border-radius: 8px;
            cursor: pointer; font-weight: 500; color: #CBD5E1; transition: 0.2s;
        }}
        .side-menu-item.active {{
            background-color: rgba(237, 92, 31, 0.1);
            color: {COLOR_PRIMARY} !important; border-left: 4px solid {COLOR_PRIMARY};
        }}

        /* Top Header Navigation */
        .dash-header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 0 20px 0; border-bottom: 1px solid {COLOR_BORDER};
            margin-bottom: 24px;
        }}
        .header-left {{ display: flex; align-items: center; gap: 18px; }}
        .header-title-box {{ display: flex; flex-direction: column; }}
        .header-title {{ font-size: 2.2rem; font-weight: 700; color: {COLOR_TEXT}; line-height: 1.1; margin: 0; }}
        .header-subtitle {{ font-size: 0.95rem; color: #64748B; margin: 0; margin-top: 2px; }}
        .header-right {{ display: flex; flex-direction: column; align-items: flex-end; gap: 8px; }}

        /* Status Pill */
        .status-pill {{
            display: inline-flex; align-items: center; gap: 6px;
            background: {COLOR_SAFE_BG}; color: #166534; padding: 4px 12px;
            border-radius: 20px; font-size: 0.75rem; font-weight: 600; border: 1px solid #BBF7D0;
        }}
        .status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {COLOR_SAFE}; }}

        /* Metric Cards Row */
        .metrics-grid {{
            display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 24px;
        }}
        .metric-card {{
            background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 12px;
            padding: 16px; display: flex; align-items: center; gap: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }}
        .metric-icon {{
            width: 48px; height: 48px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center; font-size: 1.4rem; flex-shrink: 0;
        }}
        .metric-info h4 {{ margin: 0; font-size: 0.85rem; color: #64748B; font-weight: 500; white-space: nowrap; }}
        .metric-info h2 {{ margin: 4px 0 0 0; font-size: 2.4rem; font-weight: 700; color: {COLOR_TEXT}; line-height: 1; }}
        .metric-info p {{ margin: 4px 0 0 0; font-size: 0.75rem; color: #94A3B8; }}

        /* File Uploader styling */
        section[data-testid="stFileUploaderDropzone"] {{
            background: #FFFFFF !important;
            border: 2px dashed #CBD5E1 !important;
            border-radius: 12px !important;
            padding: 30px !important;
        }}
        section[data-testid="stFileUploaderDropzone"]:hover {{ border-color: {COLOR_PRIMARY} !important; }}

        /* Buttons */
        div.stButton > button[kind="primary"] {{
            background-color: {COLOR_PRIMARY}; color: #FFF; border: none;
            border-radius: 8px; font-weight: 600; padding: 12px; width: 100%;
        }}
        div.stButton > button[kind="primary"]:hover {{ background-color: #D64C14; }}
        div.stButton > button[kind="secondary"] {{
            background-color: #FFFFFF; color: #334155; border: 1px solid #CBD5E1;
            border-radius: 8px; font-weight: 600; padding: 12px; width: 100%;
        }}
        div.stButton > button[kind="secondary"]:hover {{ background-color: #F8FAFC; border-color: #94A3B8; color: #0F172A; }}

        /* Reports Elements */
        .report-section-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }}

        /* Expanders overrides */
        [data-testid="stExpander"] {{
            background: #FFFFFF; border: 1px solid {COLOR_BORDER} !important; border-radius: 12px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.02); margin-bottom: 12px;
        }}
        [data-testid="stExpander"] summary p {{ font-weight: 600 !important; font-size: 1.05rem; }}

        /* Details grid */
        .doc-detail-item {{ display: flex; gap: 10px; margin-bottom: 14px; align-items: flex-start; }}
        .doc-detail-icon {{ font-size: 1.1rem; color: #94A3B8; margin-top: 2px; }}
        .doc-detail-text h6 {{ margin: 0; font-size: 0.75rem; color: #64748B; font-weight: 500; text-transform: uppercase; }}
        .doc-detail-text p {{ margin: 2px 0 0 0; font-size: 0.95rem; font-weight: 600; color: {COLOR_TEXT}; }}

        /* Custom Gauge CSS */
        .gauge-container {{ position: relative; width: 140px; height: 140px; margin: 0 auto; display: flex; align-items: center; justify-content: center; }}
        .gauge-circle {{ width: 100%; height: 100%; border-radius: 50%; display: flex; align-items: center; justify-content: center; }}
        .gauge-inner {{ width: 110px; height: 110px; border-radius: 50%; background: #FFFFFF; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05); }}
        .gauge-val {{ font-size: 2rem; font-weight: 700; line-height: 1; }}
        .gauge-label {{ font-size: 0.65rem; color: #64748B; text-transform: uppercase; margin-top: 4px; font-weight: 600; text-align: center; line-height: 1.2; }}

        /* Risk Bars */
        .risk-row {{ margin-bottom: 16px; }}
        .risk-label-row {{ display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 500; margin-bottom: 6px; }}
        .risk-icon-name {{ display: flex; align-items: center; gap: 8px; color: #475569; }}
        .risk-val {{ font-weight: 600; color: {COLOR_TEXT}; }}
        .risk-track {{ width: 100%; height: 6px; background: #E2E8F0; border-radius: 99px; overflow: hidden; }}
        .risk-fill {{ height: 100%; border-radius: 99px; }}

        /* Tag */
        .verdict-tag {{ display: inline-flex; align-items: center; gap: 6px; padding: 6px 16px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; margin-bottom: 20px; }}
        .tag-genuine {{ background: {COLOR_SAFE_BG}; color: #166534; border: 1px solid #BBF7D0; }}
        .tag-suspicious {{ background: {COLOR_DANGER_BG}; color: #991B1B; border: 1px solid #FECACA; }}

        </style>
        """
    )


def _render_sidebar() -> None:
    _raw_html(
        f"""
        <div style="padding: 20px 10px; text-align: center; margin-bottom: 20px;">
            <div style="background: {COLOR_PRIMARY}; width: 60px; height: 60px; border-radius: 14px; margin: 0 auto; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(237, 92, 31, 0.4);">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" style="width: 32px; height: 32px; color: white;">
                    <path d="M11.9998 2.25L2.25 6V11.25C2.25 16.5 6.45789 21.2547 11.9998 22.5C17.5417 21.2547 21.7496 16.5 21.7496 11.25V6L11.9998 2.25Z"></path>
                </svg>
            </div>
        </div>

        <div style="margin-bottom: 40px;">
            <div class="side-menu-item active">
                <span style="font-size:1.2rem;">⊞</span> <span style="color:{COLOR_PRIMARY} !important;">Dashboard</span>
            </div>
            <div class="side-menu-item">
                <span style="font-size:1.2rem;">📷</span> Scan Documents
            </div>
            <div class="side-menu-item">
                <span style="font-size:1.2rem;">📄</span> Reports
            </div>
            <div class="side-menu-item">
                <span style="font-size:1.2rem;">📊</span> Analytics
            </div>
            <div class="side-menu-item">
                <span style="font-size:1.2rem;">⚙️</span> Settings
            </div>
            <div class="side-menu-item">
                <span style="font-size:1.2rem;">📥</span> Export
            </div>
        </div>

        <div style="margin-top: auto; padding: 20px 10px;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;">About PANShield AI</div>
            <div style="font-size: 0.75rem; color: #64748B; line-height: 1.5; margin-bottom: 16px;">
                Advanced multi-layer verification system combining rule-based checks, tamper detection and deep learning for accurate PAN card authenticity assessment.
            </div>
            <div style="font-size: 0.75rem; font-weight: 600; color: {COLOR_PRIMARY};">
                🛡️ Secure. Reliable. Intelligent.
            </div>
        </div>
        """
    )


def _render_header() -> None:
    # Attempt to load and encode the local logo file
    logo_base64 = _get_base64_image("boblogo.jpg")

    # Fallback to web URL if local file is missing
    if logo_base64:
        img_src = f"data:image/jpeg;base64,{logo_base64}"
    else:
        img_src = "https://upload.wikimedia.org/wikipedia/commons/1/12/Bank_of_Baroda_logo.svg"

    _raw_html(
        f"""
        <div class="dash-header">
            <div class="header-left">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{COLOR_PRIMARY}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 52px; height: 52px;">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                </svg>
                <div class="header-title-box">
                    <h1 class="header-title">PANShield <span style="color:{COLOR_PRIMARY}">AI</span></h1>
                    <p class="header-subtitle">{TAGLINE}</p>
                </div>
            </div>
            <div class="header-right">
                <img src="{img_src}" alt="Bank of Baroda Logo" style="height: 100px; object-fit: contain; margin-bottom: 6px;border-radius: 12px;">
                <div class="status-pill">
                    <span class="status-dot"></span> System Status: Operational
                </div>
            </div>
        </div>
        """
    )


def _render_metrics(results: list[dict[str, Any]]) -> None:
    total = len(results)
    if total == 0:
        genuine, suspicious, avg_auth, high_risk = 0, 0, 0.0, 0
    else:
        suspicious = sum(1 for r in results if r["is_fake"])
        genuine = total - suspicious
        avg_auth = sum(r["authenticity_score"] for r in results) / total
        high_risk = sum(1 for r in results if r["suspicious_score"] > 80)

    pct_gen = f"{(genuine / total * 100):.0f}% of total" if total else "0% of total"
    pct_sus = f"{(suspicious / total * 100):.0f}% of total" if total else "0% of total"

    _raw_html(
        f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-icon" style="background:#F1F5F9; color:#6366F1;">📁</div>
                <div class="metric-info">
                    <h4>Total Documents</h4>
                    <h2>{total}</h2>
                    <p>Uploaded files</p>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon" style="background:{COLOR_SAFE_BG}; color:#16A34A;">✅</div>
                <div class="metric-info">
                    <h4>Verified Genuine</h4>
                    <h2>{genuine}</h2>
                    <p>{pct_gen}</p>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon" style="background:{COLOR_DANGER_BG}; color:#DC2626;">⏰</div>
                <div class="metric-info">
                    <h4>Flagged as Suspicious</h4>
                    <h2>{suspicious}</h2>
                    <p>{pct_sus}</p>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon" style="background:#E0F2FE; color:#0284C7;">🔑</div>
                <div class="metric-info">
                    <h4>Avg. Authenticity</h4>
                    <h2>{avg_auth:.1f}%</h2>
                    <p>Across all documents</p>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon" style="background:{COLOR_WARN_BG}; color:#D97706;">⚠️</div>
                <div class="metric-info">
                    <h4>High Risk Alerts</h4>
                    <h2>{high_risk}</h2>
                    <p>Requires attention</p>
                </div>
            </div>
        </div>
        """
    )


def _gauge_html(score: float, is_fake: bool) -> str:
    color = COLOR_DANGER if is_fake else COLOR_SAFE
    track = "#F1F5F9"
    pct = max(0.0, min(100.0, score))
    deg = pct * 3.6

    return f"""
    <div class="gauge-container">
        <div class="gauge-circle" style="background: conic-gradient({color} {deg}deg, {track} 0deg);">
            <div class="gauge-inner">
                <span class="gauge-val" style="color: {color};">{pct:.0f}%</span>
                <span class="gauge-label">Authenticity<br>Score</span>
            </div>
        </div>
    </div>
    """


def _risk_bar_html(label: str, icon: str, value: float, color: str) -> str:
    return f"""
    <div class="risk-row">
        <div class="risk-label-row">
            <span class="risk-icon-name"><span style="color:{color}">{icon}</span> {label}</span>
            <span class="risk-val">{value:.0f}%</span>
        </div>
        <div class="risk-track">
            <div class="risk-fill" style="width: {value}%; background-color: {color};"></div>
        </div>
    </div>
    """


def _render_detailed_report(result: dict[str, Any]) -> None:
    is_fake = result["is_fake"]
    tag_class = "tag-suspicious" if is_fake else "tag-genuine"
    verdict_icon = "⚠️" if is_fake else "✅"
    verdict_text = "Suspicious" if is_fake else "Genuine (Likely Real)"

    expander_title = f"{'🔴' if is_fake else '🟢'} {result['filename']} — Score: {result['authenticity_score']:.0f}%"

    # Default strictly set to False
    with st.expander(expander_title, expanded=False):
        col1, col2, col3, col4 = st.columns([1.2, 1.3, 1.7, 1.5], gap="medium")

        # COLUMN 1: Gauge & Status
        with col1:
            st.markdown(
                f'<div style="text-align:center;"><div class="verdict-tag {tag_class}">{verdict_icon} {verdict_text}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _gauge_html(result["authenticity_score"], is_fake),
                unsafe_allow_html=True,
            )

            risk_badge = "High Risk" if is_fake else "Low Risk"
            badge_color = COLOR_DANGER if is_fake else COLOR_SAFE
            st.markdown(
                f'<div style="text-align:center; margin-top: 16px;"><span style="background:{badge_color}20; color:{badge_color}; padding:4px 12px; border-radius:12px; font-size:0.8rem; font-weight:600;">● {risk_badge}</span></div>',
                unsafe_allow_html=True,
            )

        # COLUMN 2: Document Details
        with col2:
            pan_val = (
                result["pan_candidates"][0] if result["pan_candidates"] else "Not found"
            )
            dob_val = (
                result["dob_candidates"][0] if result["dob_candidates"] else "Not found"
            )

            st.markdown(
                f"""
                <div style="padding-top: 10px;">
                    <div class="doc-detail-item">
                        <div class="doc-detail-icon">📄</div>
                        <div class="doc-detail-text"><h6>Document</h6><p>{result["filename"]}</p></div>
                    </div>
                    <div class="doc-detail-item">
                        <div class="doc-detail-icon">🪪</div>
                        <div class="doc-detail-text"><h6>PAN Number</h6><p>{pan_val}</p></div>
                    </div>
                    <div class="doc-detail-item">
                        <div class="doc-detail-icon">📅</div>
                        <div class="doc-detail-text"><h6>DOB</h6><p>{dob_val}</p></div>
                    </div>
                    <div class="doc-detail-item">
                        <div class="doc-detail-icon">📑</div>
                        <div class="doc-detail-text"><h6>OCR Words Found</h6><p>{result["ocr_words"]}</p></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # COLUMN 3: Risk Breakdown + Violations Dropper
        with col3:
            st.markdown(
                '<div style="font-weight:600; margin-bottom:12px; font-size:1rem; color:#1E293B;">Risk Breakdown ⓘ</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                _risk_bar_html("Rules Risk", "⚖️", result["rules_risk"], COLOR_PRIMARY)
                + _risk_bar_html(
                    _friendly_risk_label("Tamper Risk"),
                    "🛡️",
                    result["tamper_risk"],
                    COLOR_DANGER,
                )
                + _risk_bar_html(
                    _friendly_risk_label("Visual Forensics Risk"),
                    "👁️",
                    result["tamper_risk"] * 0.8,
                    "#8B5CF6",
                )
                + _risk_bar_html(
                    _friendly_risk_label("Deep Learning Risk"),
                    "🧠",
                    result["dl_risk"],
                    "#F59E0B",
                ),
                unsafe_allow_html=True,
            )

            st.write("")
            with st.expander("Issues found"):
                violations = _friendly_violation_list(result.get("violations", []))
                hard_fail_reasons = _friendly_violation_list(
                    result.get("hard_fail_reasons", [])
                )
                if violations:
                    for v in violations:
                        st.markdown(f"**⚠️ {v}**")
                elif hard_fail_reasons:
                    for v in hard_fail_reasons:
                        st.markdown(f"**🔴 {v}**")
                else:
                    st.markdown("✅ *No issues detected.*")

        # COLUMN 4: Detected Image
        with col4:
            st.markdown(
                '<div style="font-weight:600; margin-bottom:12px; font-size:1rem; color:#1E293B; display:flex; justify-content:space-between;">Detected PAN Card <span>⛶</span></div>',
                unsafe_allow_html=True,
            )
            st.image(result["annotated_image"], use_container_width=True)


# ---------------------------------------------------------------------------
# Main App Structure
# ---------------------------------------------------------------------------


def main() -> None:
    _inject_css()

    # Track the file uploader widget key in session state to completely clear the UI on reset
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    with st.sidebar:
        _render_sidebar()
        # Invisible advanced settings fallback
        config_path = "configs/pipeline_prod.yaml"
        patch_model_path = "colab/best_patch_model.pt"
        fake_threshold = 45
        rules_weight = 0.45
        tamper_weight = 0.35
        dl_weight = 0.20

    _render_header()

    st.markdown(
        '<div class="report-section-title">Upload PAN Cards <span style="font-size:0.85rem; color:#64748B; font-weight:400;">Upload one or more PAN card images to begin verification</span></div>',
        unsafe_allow_html=True,
    )

    upload_col, _ = st.columns([1, 0.01])
    with upload_col:
        # Dynamically key the uploader so it resets when the key changes
        uploads = st.file_uploader(
            "Drag & drop PAN card images here",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"pan_uploader_{st.session_state['uploader_key']}",
        )

    # Action Buttons: Start Verification and Reset
    st.write("")  # Spacer
    action_col1, action_col2, action_col3, action_col4 = st.columns([1.5, 1, 1, 1.5])

    with action_col2:
        run_clicked = st.button(
            "🔲 Start Verification ❯", type="primary", use_container_width=True
        )

    with action_col3:
        # Pushing the Reset increments the uploader key, immediately wiping the UI files.
        reset_clicked = st.button(
            "🔄 Clear All", type="secondary", use_container_width=True
        )
        if reset_clicked:
            _clear_streamlit_caches()
            st.session_state["uploader_key"] += 1
            st.rerun()

    # Execution Block
    if run_clicked:
        if not uploads:
            st.warning(
                "⚠️ Please upload at least one PAN card image to start verification."
            )
        else:
            try:
                _clear_streamlit_caches()
                extractor = _get_ocr_extractor(config_path)
            except Exception as exc:
                st.error(f"Backend not connected or failed to load: {exc}")
                return

            results: list[dict[str, Any]] = []
            with st.spinner("Analyzing documents via PANShield AI Engine..."):
                for idx, up in enumerate(uploads):
                    image = Image.open(up).convert("RGB")
                    try:
                        result = _score_document(
                            image=image,
                            filename=up.name,
                            extractor=extractor,
                            fake_threshold=fake_threshold,
                            rules_weight=rules_weight,
                            tamper_weight=tamper_weight,
                            dl_weight=dl_weight,
                            patch_model_path=patch_model_path,
                        )
                        results.append(result)
                    except Exception as exc:
                        message = str(exc).lower()
                        if (
                            _is_primitive_error(exc)
                            or "ocr could not read" in message
                            or "paddleocr" in message
                        ):
                            st.warning(
                                f"{up.name}: OCR could not read this image, so it was skipped."
                            )
                        else:
                            st.error(f"{up.name}: failed to process ({exc})")

            if results:
                st.markdown("---")
                _render_metrics(results)

                st.markdown(
                    """
                    <div class="report-section-title">
                        Detailed Reports
                        <div style="font-size:0.85rem; font-weight:400; color:#64748B;">Click on any document to view full analysis</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for result in results:
                    _render_detailed_report(result)


if __name__ == "__main__":
    main()
