"""
src/models/dl_scorer.py

EfficientNet-B0 patch classifier inference wrapper.
Trained on annotated tamper patches from 213 fake PAN cards.
Best val AUC: 0.732, threshold: 0.952

Usage:
    scorer = PatchTamperScorer('colab/best_patch_model.pt')
    result = scorer.score('data/raw/fake_pan_card/FAKE_PAN_083.JPG')
    print(result['dl_score'], result['dl_verdict'])
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── model definition (must match training exactly) ────────────────────────────


class PatchClassifier(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm not installed: pip install timm")
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 1280
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)


# ── inference transform ───────────────────────────────────────────────────────


def _get_transform(patch_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# ── sliding window ────────────────────────────────────────────────────────────


def _extract_patches(
    img_rgb: np.ndarray,
    patch_size: int,
    stride: int,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    h, w = img_rgb.shape[:2]
    patches, coords = [], []
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            patch = img_rgb[y : y + patch_size, x : x + patch_size]
            if patch.shape[:2] != (patch_size, patch_size):
                continue
            patches.append(patch)
            coords.append((y, x))
    return patches, coords


# ── main scorer class ─────────────────────────────────────────────────────────


class PatchTamperScorer:
    """
    Loads the trained EfficientNet-B0 checkpoint and scores a full card image
    by running a sliding window and aggregating patch scores.

    Args:
        checkpoint_path: path to best_patch_model.pt
        device:          'cpu' or 'cuda' (auto-detected if not specified)
    """

    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.patch_size = int(ckpt.get("patch_size", 128))
        self.threshold = float(ckpt["metrics"]["thr"])

        self.model = PatchClassifier(dropout=0.4)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.transform = _get_transform(self.patch_size)

        print(
            f"PatchTamperScorer loaded | device={self.device} "
            f"patch_size={self.patch_size} threshold={self.threshold:.3f}"
        )

    @torch.no_grad()
    def score(self, image_path: str) -> dict[str, Any]:
        """
        Score a single PAN card image.

        Returns:
            dl_score:   float 0-1  (max patch score across sliding window)
            dl_verdict: 'FAKE' or 'REAL'
            dl_fires:   bool (True if dl_score >= threshold)
            patch_scores_mean: mean of all patch scores
            n_patches:  number of patches evaluated
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {
                "dl_score": 0.0,
                "dl_verdict": "REAL",
                "dl_fires": False,
                "patch_scores_mean": 0.0,
                "n_patches": 0,
                "error": "image_load_failed",
            }

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        stride = self.patch_size // 2

        patches, _ = _extract_patches(img_rgb, self.patch_size, stride)
        if not patches:
            return {
                "dl_score": 0.0,
                "dl_verdict": "REAL",
                "dl_fires": False,
                "patch_scores_mean": 0.0,
                "n_patches": 0,
            }

        # batch inference
        tensors = [self.transform(image=p)["image"] for p in patches]
        all_scores = []
        batch_size = 32
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(self.device)
            logits = self.model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(probs.tolist())

        all_scores = np.array(all_scores)
        max_score = float(all_scores.max())
        mean_score = float(all_scores.mean())
        dl_fires = max_score >= self.threshold

        return {
            "dl_score": round(max_score, 4),
            "dl_verdict": "FAKE" if dl_fires else "REAL",
            "dl_fires": dl_fires,
            "patch_scores_mean": round(mean_score, 4),
            "n_patches": len(patches),
        }
