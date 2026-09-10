from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image

# Disable the experimental Paddle IR compiler to fix the OneDNN C++ bug
os.environ["FLAGS_enable_pir_api"] = "0"
# Disable MKLDNN fallback path that may fail with "could not execute a primitive"
os.environ.setdefault("FLAGS_use_mkldnn", "0")

logger = logging.getLogger(__name__)

BoundingBox = list[int]  # [x_min, y_min, x_max, y_max] in [0, 1000]
_Polygon = list[list[float]]  # 4 corner points from PaddleOCR


class PaddleOCRExtractor:
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("paddleocr", {})

        self._lang = str(cfg.get("lang", "en"))
        self._use_angle_cls = bool(cfg.get("use_angle_cls", True))
        self._min_score = float(cfg.get("min_score", 0.35))
        # use_gpu / det_db_thresh / rec_batch_num removed in PaddleOCR >= 2.8
        # GPU is now auto-detected by PaddlePaddle at runtime

        raw_cache = cfg.get("cache_dir")
        self._cache_dir: Path | None = Path(raw_cache) if raw_cache else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._ocr: Any = None
        logger.info(
            "PaddleOCRExtractor | lang=%s | min_score=%.2f | cache=%s",
            self._lang,
            self._min_score,
            self._cache_dir,
        )

    @property
    def is_loaded(self) -> bool:
        return self._ocr is not None

    def load(self) -> None:
        self._ensure_numpy_compat()
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pip install paddleocr paddlepaddle") from exc

        logger.info("Loading PaddleOCR (lang=%s)…", self._lang)
        self._ocr = PaddleOCR(
            use_textline_orientation=self._use_angle_cls,
            lang=self._lang,
            text_det_thresh=0.3,
        )
        logger.info("PaddleOCR loaded.")

    def extract(self, image: Image.Image) -> dict[str, list]:
        if not self.is_loaded:
            raise RuntimeError("Call load() before extract().")
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL.Image.Image, got {type(image)}")

        image = image.convert("RGB")

        if self._cache_dir:
            cached = self._read_cache(image)
            if cached:
                logger.debug("OCR cache hit.")
                return cached

        img_w, img_h = image.size
        words, boxes, scores = [], [], []

        raw = self._run_ocr_with_fallback(image)
        for polygon, text, score in self._iter_ocr_lines(raw):
            if not text.strip() or score < self._min_score:
                continue
            words.append(text)
            boxes.append(self._norm_box(polygon, img_w, img_h))
            scores.append(float(score))

        result = {"words": words, "boxes": boxes, "scores": scores}
        logger.debug("Extracted %d tokens.", len(words))

        if self._cache_dir:
            self._write_cache(image, result)
        return result

    def extract_batch(self, images: list[Image.Image]) -> list[dict[str, list]]:
        return [self.extract(img) for img in images]

    # Private helpers

    @staticmethod
    def _norm_box(polygon: _Polygon, img_w: int, img_h: int) -> BoundingBox:
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        x_min = max(0, int(round(min(xs) / img_w * 1000)))
        y_min = max(0, int(round(min(ys) / img_h * 1000)))
        x_max = min(1000, int(round(max(xs) / img_w * 1000)))
        y_max = min(1000, int(round(max(ys) / img_h * 1000)))
        x_max = max(x_max, x_min + 1)
        y_max = max(y_max, y_min + 1)
        return [x_min, y_min, min(x_max, 1000), min(y_max, 1000)]

    @staticmethod
    def _img_hash(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    def _cache_path(self, image: Image.Image) -> Path:
        return self._cache_dir / f"{self._img_hash(image)}.json"  # type: ignore[operator]

    def _read_cache(self, image: Image.Image) -> dict[str, list] | None:
        p = self._cache_path(image)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _write_cache(self, image: Image.Image, result: dict[str, list]) -> None:
        self._cache_path(image).write_text(json.dumps(result), encoding="utf-8")

    @staticmethod
    def _iter_ocr_lines(raw: Any):
        """Yield polygons, text, and scores from PaddleOCR 2.x or 3.x output."""
        if not raw or raw[0] is None:
            return

        first = raw[0]
        try:
            texts = first["rec_texts"]
            scores = first["rec_scores"]
            polygons = first["rec_polys"]
        except (KeyError, TypeError, IndexError):
            for line in first:
                yield line[0], str(line[1][0]), float(line[1][1])
            return

        for polygon, text, score in zip(polygons, texts, scores):
            yield polygon, str(text), float(score)

    @staticmethod
    def _ensure_numpy_compat() -> None:
        """
        PaddleOCR dependencies may access `np.sctypes`, removed in NumPy 2.0.
        Re-create it to keep older downstream code working.
        """
        import numpy as np

        if hasattr(np, "sctypes"):
            return

        np.sctypes = {  # type: ignore[attr-defined]
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_, np.str_, np.void],
        }

    def _run_ocr_with_fallback(self, image: Image.Image):
        import numpy as np

        try:
            method = getattr(self._ocr, "predict", self._ocr.ocr)
            return method(np.array(image, dtype=np.uint8))
        except Exception as exc:
            message = str(exc).lower()
            if "primitive" not in message:
                raise

            # Some Paddle/OneDNN builds fail on large input tensors;
            # retry once with a bounded resolution image.
            retry_image = image.copy()
            retry_image.thumbnail((1600, 1600), Image.Resampling.BILINEAR)
            logger.warning(
                "PaddleOCR primitive execution failed; retrying with resized image (%sx%s).",
                retry_image.width,
                retry_image.height,
            )
            try:
                method = getattr(self._ocr, "predict", self._ocr.ocr)
                return method(np.array(retry_image, dtype=np.uint8))
            except Exception as retry_exc:
                raise RuntimeError(
                    "PaddleOCR failed after primitive-error retry."
                ) from retry_exc
