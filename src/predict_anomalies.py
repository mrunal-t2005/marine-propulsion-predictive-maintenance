
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
LABEL_COLUMN = "Fault_Label"


def main():
    for path in (DATA_PATH, MODEL_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load the fitted pipeline and its matching threshold.
    pipeline = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    feature_columns = metadata["feature_columns"]
    threshold = float(metadata["threshold"])

    df = pd.read_csv(DATA_PATH)
    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN], errors="coerce"
    )
    df = (
        df.dropna(subset=[TIME_COLUMN])
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )

    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Input is missing model features: {sorted(missing)}"
        )

    # Use the same feature order used during model training.
    X = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # No fit() call: the saved preprocessing and model are reused.
    scores = -pipeline.decision_function(X)
    predictions = scores >= threshold

    results = pd.DataFrame({
        TIME_COLUMN: df[TIME_COLUMN],
        "anomaly_score": scores,
        "is_anomaly": predictions,
    })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "anomaly_inference_demo.csv"
    results.to_csv(output_path, index=False)

    print("Saved model loaded successfully.")
    print(f"Threshold: {threshold:.6f}")
    print(f"Rows scored: {len(results)}")
    print(f"Flagged anomalies: {int(predictions.sum())}")
    print(f"Flag rate: {predictions.mean():.2%}")
    print(f"Output: {output_path}")

    print("\nFirst five predictions:")
    print(results.head().to_string(index=False))


if __name__ == "__main__":
    main()
