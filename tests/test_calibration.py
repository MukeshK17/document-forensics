from pathlib import Path

import pandas as pd
import pytest

from src.calibration import predict_calibrated, train_calibrator


@pytest.fixture
def calibration_csv(tmp_path: Path) -> Path:
    data = pd.DataFrame(
        {
            "rules": [0, 90, 10, 80, 20, 70],
            "tamper": [0, 80, 20, 70, 10, 90],
            "forensic": [5, 85, 15, 75, 25, 95],
            "deep_learning": [0, 95, 5, 65, 30, 88],
            "is_fake": [0, 1, 0, 1, 0, 1],
        }
    )
    csv_path = tmp_path / "scores.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


def test_train_calibrator_produces_saved_model(
    calibration_csv: Path, tmp_path: Path
) -> None:
    model_path = tmp_path / "calibrator.joblib"

    saved_path = train_calibrator(
        calibration_csv,
        feature_columns=["rules", "tamper", "forensic", "deep_learning"],
        label_column="is_fake",
        model_path=model_path,
    )

    assert saved_path == model_path
    assert model_path.is_file()


def test_predict_calibrated_returns_probability(
    calibration_csv: Path, tmp_path: Path
) -> None:
    model_path = tmp_path / "calibrator.joblib"
    train_calibrator(
        calibration_csv,
        feature_columns=["rules", "tamper", "forensic", "deep_learning"],
        label_column="is_fake",
        model_path=model_path,
    )

    probability = predict_calibrated(35, 0, 45, 39.23, model_path=model_path)

    assert isinstance(probability, float)
    assert 0 <= probability <= 1


def test_train_calibrator_raises_for_missing_column(
    calibration_csv: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="missing columns: missing"):
        train_calibrator(
            calibration_csv,
            feature_columns=["rules", "tamper", "forensic", "missing"],
            label_column="is_fake",
            model_path=tmp_path / "calibrator.joblib",
        )
