
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_anomalies import (  # noqa: E402
    load_artifacts,
    predict_anomalies,
    prepare_features,
)


@pytest.fixture
def fitted_artifacts():
    """Create a small model without using the project dataset."""
    rng = np.random.default_rng(42)

    training_data = pd.DataFrame({
        "Air_Pressure": rng.normal(10, 1, 100),
        "Oil_Pressure": rng.normal(5, 0.5, 100),
    })

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "detector",
            IsolationForest(
                n_estimators=30,
                contamination=0.1,
                random_state=42,
            ),
        ),
    ])
    pipeline.fit(training_data)

    metadata = {
        "feature_columns": [
            "Air_Pressure",
            "Oil_Pressure",
        ],
        "threshold": 0.0,
    }

    return pipeline, metadata, training_data


def test_prepare_features_preserves_feature_order(fitted_artifacts):
    _, metadata, data = fitted_artifacts

    reversed_data = data[["Oil_Pressure", "Air_Pressure"]]
    prepared = prepare_features(
        reversed_data,
        metadata["feature_columns"],
    )

    assert list(prepared.columns) == metadata["feature_columns"]


def test_missing_feature_is_rejected(fitted_artifacts):
    _, metadata, data = fitted_artifacts

    incomplete = data.drop(columns=["Oil_Pressure"])

    with pytest.raises(ValueError, match="missing model features"):
        prepare_features(incomplete, metadata["feature_columns"])


def test_infinity_is_converted_to_missing_value(fitted_artifacts):
    _, metadata, data = fitted_artifacts

    data_with_infinity = data.copy()
    data_with_infinity.loc[0, "Air_Pressure"] = np.inf

    prepared = prepare_features(
        data_with_infinity,
        metadata["feature_columns"],
    )

    assert pd.isna(prepared.loc[0, "Air_Pressure"])


def test_prediction_output_has_expected_format(fitted_artifacts):
    pipeline, metadata, data = fitted_artifacts

    predictions = predict_anomalies(data, pipeline, metadata)

    assert list(predictions.columns) == [
        "anomaly_score",
        "is_anomaly",
    ]
    assert len(predictions) == len(data)
    assert np.isfinite(predictions["anomaly_score"]).all()
    assert predictions["is_anomaly"].dtype == bool


def test_flags_follow_saved_threshold(fitted_artifacts):
    pipeline, metadata, data = fitted_artifacts

    predictions = predict_anomalies(data, pipeline, metadata)

    expected = (
        predictions["anomaly_score"].to_numpy()
        >= metadata["threshold"]
    )

    np.testing.assert_array_equal(
        predictions["is_anomaly"].to_numpy(),
        expected,
    )


def test_saved_artifacts_can_be_loaded(tmp_path, fitted_artifacts):
    pipeline, metadata, _ = fitted_artifacts

    model_path = tmp_path / "pipeline.joblib"
    metadata_path = tmp_path / "metadata.joblib"

    joblib.dump(pipeline, model_path)
    joblib.dump(metadata, metadata_path)

    loaded_pipeline, loaded_metadata = load_artifacts(
        model_path,
        metadata_path,
    )

    assert loaded_metadata["threshold"] == metadata["threshold"]
    assert loaded_metadata["feature_columns"] == metadata["feature_columns"]
    assert hasattr(loaded_pipeline, "decision_function")


def test_invalid_metadata_is_rejected(tmp_path):
    model_path = tmp_path / "pipeline.joblib"
    metadata_path = tmp_path / "metadata.joblib"

    joblib.dump(object(), model_path)
    joblib.dump({"threshold": 0.0}, metadata_path)

    with pytest.raises(ValueError, match="metadata is missing"):
        load_artifacts(model_path, metadata_path)
