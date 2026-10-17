
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "marine_propulsion.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_PATH = MODELS_DIR / "anomaly_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "anomaly_metadata.joblib"

TIME_COLUMN = "Timestamp"


def load_artifacts(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    """Load the trained anomaly-detection pipeline and its metadata."""
    pipeline = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    required_metadata = {"feature_columns", "threshold"}
    missing = required_metadata - set(metadata.keys())

    if missing:
        raise ValueError(
            f"Model metadata is missing: {sorted(missing)}"
        )

    return pipeline, metadata


def prepare_features(data, feature_columns):
    """Validate, clean, and order input features for inference."""
    missing = set(feature_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"Input is missing model features: {sorted(missing)}"
        )

    # Preserve the feature order used during training.
    X = data[feature_columns].copy()

    # Replace infinite values and convert features to numeric.
    X = X.replace([np.inf, -np.inf], np.nan)

    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # Reject columns that contain no usable numeric values.
    all_missing = X.columns[X.isna().all()].tolist()

    if all_missing:
        raise ValueError(
            f"Features contain no valid numeric values: {all_missing}"
        )

    return X


def predict_anomalies(data, pipeline, metadata):
    """Calculate anomaly scores and classify rows using a saved threshold."""
    feature_columns = metadata["feature_columns"]
    threshold = float(metadata["threshold"])

    X = prepare_features(data, feature_columns)

    # Isolation Forest's decision_function is higher for normal samples.
    # Negating it makes higher scores indicate greater anomaly.
    scores = -pipeline.decision_function(X)
    flags = scores >= threshold

    return pd.DataFrame({
        "anomaly_score": scores,
        "is_anomaly": flags,
    })


def main():
    required_paths = (DATA_PATH, MODEL_PATH, METADATA_PATH)

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    pipeline, metadata = load_artifacts()

    df = pd.read_csv(DATA_PATH)

    if TIME_COLUMN not in df.columns:
        raise ValueError(
            f"Input dataset is missing the '{TIME_COLUMN}' column."
        )

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        errors="coerce",
    )

    df = (
        df.dropna(subset=[TIME_COLUMN])
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )

    predictions = predict_anomalies(df, pipeline, metadata)

    results = pd.concat(
        [df[[TIME_COLUMN]], predictions],
        axis=1,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "anomaly_inference_demo.csv"
    results.to_csv(output_path, index=False)

    print("Saved anomaly-detection model loaded successfully.")
    print(f"Threshold: {float(metadata['threshold']):.6f}")
    print(f"Rows scored: {len(results)}")
    print(f"Flagged anomalies: {int(results['is_anomaly'].sum())}")
    print(f"Flag rate: {results['is_anomaly'].mean():.2%}")
    print(f"Output saved to: {output_path}")

    print("\nFirst five predictions:")
    print(results.head().to_string(index=False))


if __name__ == "__main__":
    main()