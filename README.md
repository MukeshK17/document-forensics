# Fake Document Detection Pipeline

A modular machine learning pipeline for detecting suspicious PAN-card
documents by combining OCR-based validation, regional tamper analysis,
visual forensics, and patch-level deep learning.

The system was developed for document-fraud review in a banking
risk-management setting, where sensitive documents can be processed
locally without sending document images to external APIs.

## Overview

Fake documents can contain different types of inconsistencies: invalid
or OCR-corrupted identity fields, locally edited text regions, pasted
images, or other visual artifacts. A single detector is therefore prone
to missing certain forgery patterns.

This project combines four complementary signals:

1.  **OCR and rule validation** --- extracts text and bounding boxes
    with PaddleOCR and validates PAN structure, dates, required fields,
    OCR corrections, and other document-level inconsistencies.
2.  **Regional tamper detection** --- compares local sharpness, OCR
    confidence, and text-edge characteristics against surrounding
    document regions.
3.  **Visual forensics** --- checks for suspicious digital overlays,
    unusual text/ink appearance, missing or excessive faces, and pasted
    high-resolution photos.
4.  **Patch-level deep learning** --- applies an EfficientNet-B0
    classifier over sliding image patches to detect localized tampering.

A document is marked **Suspicious when at least two of the four stages
fire**, reducing dependence on any single detector.

## Pipeline

``` text
                         PAN card image
                               │
                               ▼
                         ┌───────────┐
                         │  PaddleOCR │
                         └─────┬─────┘
                               │
                  words + boxes + confidence
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      OCR / Rule         Regional Tamper    Visual Forensics
       Validation          Detection          Analysis
             │                 │                 │
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                     EfficientNet-B0
                     Patch Classifier
                               │
                               ▼
                       2-of-4 Voting
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
                Suspicious              Clean
```

## Stage 1 --- OCR and Rule Validation

`PaddleOCR` provides text, normalized bounding boxes, and OCR confidence
scores.

The validation layer performs deterministic checks including:

-   PAN structure and taxpayer-entity code validation
-   OCR-aware PAN normalization for common character confusions
-   detection of conflicting PAN numbers
-   date extraction and calendar validation
-   mandatory PAN-card issuer/template checks
-   surname-initial consistency checks
-   known test/dummy PAN detection
-   OCR confidence tracking

The validation logic is implemented in
`src/validators/field_validator.py`.

## Stage 2 --- Regional Tamper Detection

The regional detector uses OCR bounding boxes to inspect text regions
against their local document context.

It combines:

-   Laplacian sharpness inside the text region versus its surrounding
    collar
-   OCR-confidence differences
-   edge-gradient variation after binarization

Regions that exhibit multiple suspicious signals are flagged and
returned with their bounding boxes for visualization.

Implementation: `src/models/tamper_detector.py`.

## Stage 3 --- Visual Forensics

The visual-forensics module analyzes image-level and field-level
artifacts, including:

-   flat digital text-box / overlay regions
-   white-mask cover-ups
-   unusual ink characteristics
-   face-count inconsistencies
-   sharpness mismatch between detected photos and surrounding card
    regions

These checks are implemented in `tools/visual_forensics.py`.

Additional image-level analysis in `src/models/cv_analyzer.py`
evaluates:

-   unusually low resolution
-   suspicious screen-like resolutions
-   unusual aspect ratios
-   uneven image noise
-   JPEG Error Level Analysis (ELA)

## Stage 4 --- Patch-Level EfficientNet-B0

A patch classifier based on **EfficientNet-B0** scores localized regions
of the PAN card using a sliding-window strategy.

The inference wrapper:

-   loads the trained checkpoint
-   extracts overlapping patches
-   runs batched inference
-   records the maximum patch-level tamper probability
-   returns the mean patch score and number of evaluated patches

Implementation: `src/models/dl_scorer.py`.

The trained model was developed using annotated tampered PAN-card
regions. The training data and model checkpoints are intentionally
excluded from the public repository.

## Decision Logic

The final evaluation uses a simple **2-of-4 voting scheme**:

``` text
OCR / Rules       ──┐
Regional Tamper   ──┤
Visual Forensics  ──┼──► >= 2 stages fire ──► Suspicious
Patch Classifier  ──┘
```

This is intended to reduce false positives caused by relying on a single
signal.

The evaluation workflow in `tools/run_baseline.py` computes precision,
recall, F1, and false-positive rate across train/validation/test splits.

## Streamlit Demonstration

The repository includes a local Streamlit interface used to demonstrate
the pipeline:

``` bash
streamlit run streamlit_app.py
```

The interface supports:

-   batch PAN-card image upload
-   OCR and rule validation
-   regional tamper analysis
-   optional EfficientNet patch scoring
-   final suspicious/authenticity scores
-   highlighted suspicious regions
-   downloadable JSON-style assessment output

The Streamlit application is a demonstration layer; the core detection
components remain usable independently.

## Repository Structure

``` text
fake-document-/
├── configs/
│   └── pipeline_prod.yaml       # Runtime/model configuration
├── src/
│   ├── data/                    # Ingestion, preprocessing, datasets, loaders
│   ├── extractors/
│   │   └── paddleOcr_extractor.py
│   ├── models/
│   │   ├── cv_analyzer.py       # Image-level forensic checks
│   │   ├── dl_scorer.py         # EfficientNet-B0 patch scoring
│   │   ├── layoutlm_classifier.py
│   │   └── tamper_detector.py   # Regional tamper detection
│   ├── validators/
│   │   ├── field_validator.py   # PAN/document validation
│   │   └── rules_engine.py
│   └── utils/
│       └── fraud_logger.py
├── tools/
│   ├── build_splits.py
│   ├── build_ocr_cache.py
│   ├── build_field_crops.py
│   └── visual_forensics.py
├── tests/
├── streamlit_app.py             # Local demonstration UI
├── annotation.py                # Annotation/template utility
├── dataset.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Configuration

Pipeline configuration is stored in:

``` text
configs/pipeline_prod.yaml
```

It contains runtime settings, model configuration, OCR settings,
preprocessing parameters, and risk thresholds.

The repository also contains a more general
`DocumentVerificationPipeline` in `src/pipeline.py` with
LayoutLMv3-based document classification. The PAN-card four-stage
evaluation path described above is implemented through the validation,
tamper, forensic, and patch-scoring components and is used by the
evaluation tooling and demonstration interface.

## Data and Model Artifacts

Sensitive document datasets, generated outputs, notebooks, and trained
checkpoints are not included in the public repository.

Expected local artifacts include:

``` text
data/
colab/
outputs/
reports/
```

These paths are excluded from version control.

## Development

Install the Python dependencies:

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-runtime.txt
```

Run Ruff locally:

``` bash
ruff check .
ruff format --check .
```

Run the test suite:

``` bash
pytest
```

## License

MIT License. See `LICENSE`.
