
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "marine_propulsion.csv"
MODELS_DIR = ROOT / "models"

TIME_COLUMN = "Timestamp"
LABEL_COLUMN = "Fault_Label"
NORMAL_LABEL = 0

RANDOM_STATE = 42
CONTAMINATION = 0.08
TRAIN_RATIO = 0.70
VALIDATION_END_RATIO = 0.85


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required = {TIME_COLUMN, LABEL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN], errors="coerce"
    )
    df = (
        df.dropna(subset=[TIME_COLUMN, LABEL_COLUMN])
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )

    feature_columns = [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column != LABEL_COLUMN
    ]

    if not feature_columns:
        raise ValueError("No numeric sensor features found.")

    X = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    y = (
        df[LABEL_COLUMN].astype(int) != NORMAL_LABEL
    ).astype(int)

    train_end = int(len(df) * TRAIN_RATIO)
    validation_end = int(
        len(df) * VALIDATION_END_RATIO
    )

    if not (0 < train_end < validation_end < len(df)):
        raise ValueError(
            "Dataset is too small for the chronological split."
        )

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:validation_end]
    y_val = y.iloc[train_end:validation_end].to_numpy()

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=200,
                    contamination=CONTAMINATION,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training on the first 70% of chronological data...")
    pipeline.fit(X_train)

    # Larger scores indicate more unusual observations.
    validation_scores = -pipeline.decision_function(X_val)

    precision, recall, thresholds = precision_recall_curve(
        y_val, validation_scores
    )

    average_precision = average_precision_score(
        y_val, validation_scores
    )

    # Match each threshold to its corresponding precision/recall.
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_scores))
    threshold = float(thresholds[best_index])

    # Keep the exact fitted model paired with its threshold.
    model_path = MODELS_DIR / "anomaly_pipeline.joblib"
    metadata_path = MODELS_DIR / "anomaly_metadata.joblib"

    metadata = {
        "feature_columns": feature_columns,
        "threshold": threshold,
        "normal_label": NORMAL_LABEL,
        "random_state": RANDOM_STATE,
        "contamination": CONTAMINATION,
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "validation_average_precision": float(
            average_precision
        ),
        "validation_precision": float(
            precision[best_index]
        ),
        "validation_recall": float(
            recall[best_index]
        ),
        "validation_f1": float(f1_scores[best_index]),
        "threshold_selection": (
            "Maximum F1 on the middle 15% validation segment"
        ),
    }

    joblib.dump(pipeline, model_path)
    joblib.dump(metadata, metadata_path)

    print("\nModel saved successfully.")
    print(f"Pipeline: {model_path}")
    print(f"Metadata: {metadata_path}")

    print("\nValidation-selected threshold:")
    print(f"Threshold: {threshold:.6f}")
    print(f"Average precision: {average_precision:.3f}")
    print(f"Precision: {precision[best_index]:.3f}")
    print(f"Recall: {recall[best_index]:.3f}")
    print(f"F1 score: {f1_scores[best_index]:.3f}")

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Features:", len(feature_columns))

    print(
        "\nThe saved pipeline and threshold belong together. "
        "The final 15% test segment was not used."
    )


if __name__ == "__main__":
    main()
