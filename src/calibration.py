"""Train and apply logistic-regression score calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

DEFAULT_FEATURE_COLUMNS = (
    "rules_risk",
    "tamper_risk",
    "forensic_risk",
    "dl_risk",
)
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_MODEL_PATH = Path("outputs/calibrator.joblib")


def _validate_columns(
    columns: Sequence[str], feature_columns: Sequence[str], label_column: str
) -> None:
    expected = set(feature_columns) | {label_column}
    actual = set(columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected columns: {', '.join(unexpected)}")
        raise ValueError(
            "CSV columns do not match expected schema (" + "; ".join(details) + ")"
        )


def _validate_feature_names(
    feature_columns: Sequence[str], label_column: str
) -> tuple[str, ...]:
    feature_columns = tuple(feature_columns)
    if len(feature_columns) != 4 or len(set(feature_columns)) != 4:
        raise ValueError(
            "feature_columns must contain exactly four unique column names"
        )
    if label_column in feature_columns:
        raise ValueError("label_column must not also be a feature column")
    return feature_columns


def train_calibrator(
    csv_path: str | Path,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    label_column: str = DEFAULT_LABEL_COLUMN,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Path:
    """Fit a calibrator from four 0-100 risk columns and save it with joblib.

    Args:
        csv_path: CSV containing exactly the feature and label columns.
        feature_columns: Four score column names, in model input order.
        label_column: Binary ground-truth column, where 1 means fake.
        model_path: Destination for the fitted joblib model.

    Returns:
        The path of the saved model.

    Raises:
        ValueError: If the schema or labels do not meet the requirements.
    """
    feature_columns = _validate_feature_names(feature_columns, label_column)
    data = pd.read_csv(csv_path)
    _validate_columns(data.columns, feature_columns, label_column)

    labels = data[label_column]
    if labels.isna().any() or not set(labels).issubset({0, 1}):
        raise ValueError(f"{label_column!r} must contain only binary labels 0 and 1")

    try:
        features = data.loc[:, feature_columns].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature columns must contain numeric values") from exc
    if features.isna().any().any():
        raise ValueError("feature columns must not contain missing values")

    model = LogisticRegression()
    model.fit(features, labels.astype(int))
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def predict_calibrated(
    rules_risk: float,
    tamper_risk: float,
    forensic_risk: float,
    dl_risk: float,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> float:
    """Return the calibrated fake probability for four 0-100 risk scores."""
    model = joblib.load(model_path)
    values = [[rules_risk, tamper_risk, forensic_risk, dl_risk]]
    feature_names = getattr(model, "feature_names_in_", None)
    features = pd.DataFrame(values, columns=feature_names)
    probability = model.predict_proba(features)[0, 1]
    return float(probability)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", metavar="CSV", help="fit a calibrator from CSV")
    mode.add_argument(
        "--predict",
        nargs=4,
        type=float,
        metavar=("RULES", "TAMPER", "FORENSIC", "DL"),
        help="predict fake probability from four risk scores",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        type=Path,
        help=f"joblib model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--feature-columns",
        nargs=4,
        default=DEFAULT_FEATURE_COLUMNS,
        metavar=("RULES", "TAMPER", "FORENSIC", "DL"),
        help="four CSV feature columns in prediction argument order",
    )
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    return parser.parse_args()


def main() -> None:
    """Run calibration training or prediction from the command line."""
    args = _parse_args()
    try:
        if args.train:
            path = train_calibrator(
                args.train,
                feature_columns=args.feature_columns,
                label_column=args.label_column,
                model_path=args.model_path,
            )
            print(f"Saved calibrator to {path}")
        else:
            probability = predict_calibrated(*args.predict, model_path=args.model_path)
            print(f"Fake probability: {probability:.6f}")
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
