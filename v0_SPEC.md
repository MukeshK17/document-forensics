# V0 Spec: Fake PAN Card Detection — Baseline Ensemble + VLM Complementarity Gate

Status: **Complete**
Owner: Mukesh K.
Context: Bank of Baroda Risk Management internship deliverable, extended post-internship as a long-term learning project.

---

## 1. Purpose

V0 establishes two things before any further architecture is built:

1. A working, evaluated baseline detection pipeline (four independent specialists + majority voting).
2. A scientific gate answering: *does a general-purpose VLM add real detection signal beyond this baseline, and does it justify building agentic/router infrastructure on top?*

No router, agent, or LangGraph code exists at this stage. V0 is intentionally a floor, not a final architecture.

---

## 2. System under test

### 2.1 Four-stage pipeline

| Stage | Method | File |
|---|---|---|
| OCR Rule Validation | PaddleOCR + PAN format rules | `field_validator.py` |
| Tamper Detection | OCR bounding boxes + Laplacian variance (sharpness discontinuity) | `tamper_detector.py` |
| Visual Forensics | HSV ink-colour analysis, photo/face presence | `visual_forensics.py` |
| DL Patch Classifier | EfficientNet-B0, 128×128 patches, inference only | `dl_scorer.py` |

**Important:** the DL component has **no training pipeline in this repo**. `dl_scorer.py` loads a frozen pre-trained checkpoint (`colab/best_patch_model.pt`, trained separately in Google Colab during the internship). This checkpoint is a fixed asset for V0 and is not retrained, fine-tuned, or regenerated as part of this or later phases unless explicitly scoped.

### 2.2 Decision rule

2-of-4 majority voting across the four stages' binary outputs (`collective_fake`).

### 2.3 Baseline performance

**IMPORTANT: two different numbers exist for this system and must not be conflated.**

**(a) Production baseline — full dataset, natural class imbalance (1,284 real / 213 fake, ~86%/14% split):**

| Metric | Value |
|---|---|
| Precision | 0.325 |
| Recall | 0.451 |
| F1 | 0.378 |
| FPR | 0.155 |

This is the number that reflects actual deployment conditions and is the one every future phase must beat or explain, in a defensible way, before adding complexity.

**(b) 340-image balanced subset (170 real / 170 fake) — used only for the VLM complementarity experiment below:**

| Metric | Value |
|---|---|
| Precision | 0.544 |
| Recall | 0.871 |
| F1 | 0.670 |

This subset was constructed specifically to give the VLM experiment a class-balanced, apples-to-apples comparison (equal number of real and fake images). It is **not** a substitute for the full-dataset evaluation — artificially balancing the class ratio inflates precision relative to the true ~86/14 real/fake base rate, because the model sees far fewer real cards to false-positive against. All comparisons in Section 3 (VLM vs. existing system) are internally consistent with each other since both ran on the same 340-image split, but neither number should be read as "the system's real-world performance." That is (a) above.

**Known gap / incomplete work:** V0 does not yet report FPR, precision, or recall for the existing ensemble on a full-dataset or larger-sample basis broken down by fake-type (overlay, photo-replacement, typed-text vs. semantic single-character edits). This is needed before the V1 calibration model can be properly validated and should be treated as a prerequisite task, not deferred indefinitely.

---

## 3. VLM Complementarity Experiment

### 3.1 Question

Does Qwen2-VL-7B-Instruct (zero-shot, 4-bit quantized, local inference — no external API, no PII sent to third parties) add detection signal the four existing specialists miss? If yes, agentic/router infrastructure is justified. If no, pivot to calibrated/selective verification using the existing specialists.

### 3.2 Constraints driving the design

- No GPU locally → inference run on Kaggle T4.
- No budget for commercial VLM APIs → open-weight model only, run locally in-session (Qwen2-VL-7B-Instruct, 4-bit via bitsandbytes).
- Real identification documents must not be sent to third-party inference endpoints, even though this is internship/practice data, not live bank customer data — treated as sensitive on principle.
- Same 340-image balanced split used for both systems, so results are directly comparable.

### 3.3 Method

For each image, VLM was prompted to return structured JSON: `decision` (FAKE / REAL / INSUFFICIENT_EVIDENCE), `confidence` (0–1), `reason` (free text). Compared against the same ground truth and same test split as the existing ensemble.

### 3.4 Results

All figures below are on the 340-image **balanced** subset (Section 2.3b), not the full-dataset production baseline (Section 2.3a). The existing-ensemble row here (F1=0.670) is higher than the true production F1=0.378 purely because of class balancing — it is reported only so the VLM comparison is apples-to-apples, not as a claim about real-world performance.

| System | Precision | Recall | F1 |
|---|---|---|---|
| Existing 4-stage ensemble (on 340-balanced subset) | 0.544 | 0.871 | **0.670** |
| VLM standalone | 0.521 | 0.371 | **0.433** |
| Naive OR-ensemble (either flags fake) | 0.533 | 0.912 | 0.672 |

**Complementarity crosstab (existing correct × VLM correct):**

| | VLM wrong | VLM correct |
|---|---|---|
| Existing wrong | 61 | 85 |
| Existing correct | 104 | 90 |

- 85/340 cases where VLM was correct and existing was wrong.
- Of these, ~78 are explained by the VLM defaulting to REAL on cases where the existing ensemble's low precision (0.544) produced a false positive — not genuine forensic reasoning.
- Only **7/340** are genuine fake catches via specific, non-boilerplate visual reasoning (inconsistent text alignment, photo blending, edge artifacts) — and these cluster specifically where the existing ensemble's vote count was weakest (0–1 votes out of 4).
- VLM confidence was miscalibrated: clustered almost entirely at 0.9–1.0 regardless of correctness.
- VLM never produced `INSUFFICIENT_EVIDENCE` despite the prompt explicitly offering it as a valid, encouraged output.
- 212/340 (62%) of all VLM responses collapsed into four near-identical boilerplate reason strings, indicating the model frequently defaults to template answers rather than grounding output in per-image pixel evidence.

### 3.5 Conclusion

A general-purpose VLM used naively (as a fifth equal-weight voter, or via naive OR-combination) is **not justified**:
- Standalone F1 is well below the existing ensemble.
- OR-ensemble F1 gain (0.670 → 0.672) is marginal and comes with a precision cost.
- Confidence output is not trustworthy as a calibration signal in its current form.

A **narrow, evidence-backed role for the VLM is justified**: as an escalation specialist invoked only on cases where the existing ensemble's confidence is already low (0–1 votes), where it demonstrated real, non-templated detection capability on 7 otherwise-missed fakes.

Full experiment artifacts, code, and raw results: `experiments/vlm_complementarity/`.

---

## 4. Decision: what V0 does NOT justify

- Adaptive Visual Intelligence Router (multi-specialist routing with LLM-driven tool selection)
- LangGraph / agentic workflow
- VLM as an equal-weight voting specialist
- Any architecture whose value depends on VLM detection quality exceeding what was measured here

## 5. Decision: what comes next (V1)

Per the complementarity data, V1 direction is **calibrated confidence scoring + selective escalation**, not a general router:

1. Replace 2-of-4 hard voting with a calibrated probability (logistic regression over the four raw continuous risk scores already produced by the pipeline).
2. Define a narrow, low-confidence escalation zone based on calibrated score (empirically, where existing fake-vote-count is 0–1).
3. Invoke VLM only within that zone — not on every image — keeping inference cost near-zero in aggregate.
4. Three-way output: FAKE / REAL / INSUFFICIENT_EVIDENCE, with INSUFFICIENT_EVIDENCE triggered when both the calibrated CV score and VLM output are ambiguous or boilerplate (detected via reason-string matching against known templates).
5. Evaluate via a coverage–risk curve (how much does automated-decision risk drop as more cases are escalated to human review), not F1 alone.

This is the selective-prediction / reject-option framing: a well-calibrated system should say "I don't know" rather than fabricate confidence — now backed by measured evidence of exactly where and how the current system's confidence fails.

---

## 6. Open gaps (not yet resolved in V0)

- No full-dataset (imbalanced, 1,284/213) evaluation of the VLM or the OR-ensemble — only the existing system has this number (Section 2.3a). Before any V1 calibration model is trusted, it needs to be validated against the full imbalanced set, not just the 340-balanced subset.
- No per-fake-type breakdown (overlay vs. photo-replacement vs. typed-text vs. semantic single-character edits) for either system. The thesis identifies these as having very different detectability; aggregate F1 hides this.
- No confidence interval / variance estimate on any reported metric — all numbers are point estimates from a single train/test split.
- Sample size (340, or 213 for full-dataset fakes) is small; metric estimates carry meaningful uncertainty that should be reported alongside point values in the technical report, not omitted.

## 7. Artifacts

- `tools/run_baseline.py` — existing ensemble evaluation
- `experiments/vlm_complementarity/vlm_experiment.py` — reproducible VLM experiment (data prep → inference → analysis)
- `experiments/vlm_complementarity/vlm_results.csv` — raw VLM outputs, 340 images
- `experiments/vlm_complementarity/existing_results.csv` — raw existing-system outputs, same split
- `experiments/vlm_complementarity/merged_results.csv` — merged crosstab data
- `colab/best_patch_model.pt` — frozen DL checkpoint (inference only, not retrained in this repo)