from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

logger = logging.getLogger(__name__)

BoundingBox = list[int]
ClassificationResult = dict[str, Any]

_REQUIRED_LM_KEYS = {
    "pretrained_model_name_or_path",
    "id2label",
    "processor",
    "inference",
}


class LayoutLMv3Classifier:
    def __init__(self, config: dict[str, Any]) -> None:
        if "layoutlmv3" not in config:
            raise ValueError("Config missing required key: 'layoutlmv3'")
        missing = _REQUIRED_LM_KEYS - config["layoutlmv3"].keys()
        if missing:
            raise ValueError(f"'layoutlmv3' config missing key(s): {missing}")

        self._cfg = config["layoutlmv3"]
        self._rt_cfg = config.get("runtime", {})
        self.model: LayoutLMv3ForSequenceClassification | None = None
        self.processor: LayoutLMv3Processor | None = None
        self.device: torch.device = self._resolve_device()

        self.id2label: dict[int, str] = {
            int(k): v for k, v in self._cfg["id2label"].items()
        }
        self.label2id: dict[str, int] = {v: k for k, v in self.id2label.items()}

        logger.info(
            "LayoutLMv3Classifier | device=%s | labels=%s",
            self.device,
            list(self.id2label.values()),
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        path = self._cfg["pretrained_model_name_or_path"]
        logger.info("Loading LayoutLMv3 from '%s'…", path)
        self.processor = LayoutLMv3Processor.from_pretrained(
            path,
            revision=self._cfg["revision"],
            apply_ocr=self._cfg["processor"]["apply_ocr"],
        )
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            path,
            revision=self._cfg["revision"],
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device).eval()
        logger.info("LayoutLMv3 loaded on '%s'.", self.device)

    def inference(
        self,
        image: Image.Image,
        words: list[str],
        boxes: list[BoundingBox],
    ) -> ClassificationResult:
        if not self.is_loaded:
            raise RuntimeError("Call load() before inference().")
        self._validate_ocr_inputs(words, boxes)

        proc_cfg = self._cfg["processor"]
        encoding = self.processor(
            images=image.convert("RGB"),
            text=words,
            boxes=boxes,
            max_length=proc_cfg["max_seq_length"],
            padding=proc_cfg["padding"],
            truncation=proc_cfg["truncation"],
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        use_amp = (
            self._rt_cfg.get("mixed_precision", False) and self.device.type == "cuda"
        )
        with torch.no_grad():
            ctx = torch.autocast(device_type="cuda") if use_amp else torch.no_grad()
            with ctx:
                logits = self.model(**encoding).logits

        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_id = int(probs.argmax().item())
        confidence = round(float(probs[pred_id].item()), 4)
        label = self.id2label.get(pred_id, "UNKNOWN")

        logger.debug("Inference | label=%s confidence=%.4f", label, confidence)
        return {
            "label": label,
            "confidence": confidence,
            "scores": {
                self.id2label[i]: round(float(probs[i].item()), 4)
                for i in range(len(self.id2label))
            },
            "label_id": pred_id,
        }

    # Private

    def _resolve_device(self) -> torch.device:
        d = self._rt_cfg.get("device", "auto")
        if d != "auto":
            return torch.device(d)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _validate_ocr_inputs(words: list[str], boxes: list[BoundingBox]) -> None:
        if len(words) != len(boxes):
            raise ValueError(
                f"words({len(words)}) and boxes({len(boxes)}) length mismatch."
            )
        for i, box in enumerate(boxes):
            if len(box) != 4:
                raise ValueError(f"Box[{i}] must have 4 coords, got {len(box)}.")
            if not all(0 <= c <= 1000 for c in box):
                raise ValueError(f"Box[{i}] coords out of [0,1000]: {box}")
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError(f"Box[{i}] is degenerate: {box}")
