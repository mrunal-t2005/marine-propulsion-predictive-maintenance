
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "marine_propulsion.csv"
REPORTS_DIR = ROOT / "reports"

TIME_COLUMN = "Timestamp"
LABEL_COLUMN = "Fault_Label"

# Treat class 0 as normal and classes 1–7 as known fault categories.
# This mapping follows the project's current dataset assumptions.
NORMAL_LABEL = 0

RANDOM_STATE = 42
CONTAMINATION = 0.08


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {TIME_COLUMN, LABEL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN], errors="coerce"
    )
    df = df.dropna(subset=[TIME_COLUMN, LABEL_COLUMN])
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    feature_columns = [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column != LABEL_COLUMN
    ]

    if not feature_columns:
        raise ValueError("No numeric sensor features were found.")

    X = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    y = df[LABEL_COLUMN].astype(int)

    # Fit only on the earliest 85% of observations to avoid
    # using the final chronological evaluation segment.
    split_index = int(len(df) * 0.85)
    if split_index == 0 or split_index == len(df):
        raise ValueError("Not enough rows for a chronological split.")

    X_development = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

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

    print("Training Isolation Forest on the development segment...")
    pipeline.fit(X_development)

    # Isolation Forest returns -1 for anomalies and +1 for inliers.
    test_predictions = pipeline.predict(X_test)
    anomaly_scores = -pipeline.decision_function(X_test)
    is_anomaly = test_predictions == -1

    results = df.iloc[split_index:][
        [TIME_COLUMN, LABEL_COLUMN]
    ].copy()
    results["anomaly_score"] = anomaly_scores
    results["is_anomaly"] = is_anomaly
    results["known_fault"] = (
        results[LABEL_COLUMN] != NORMAL_LABEL
    )

    output_csv = REPORTS_DIR / "anomaly_detection_results.csv"
    results.to_csv(output_csv, index=False)

    print(f"\nTest observations: {len(results)}")
    print(f"Flagged anomalies: {int(is_anomaly.sum())}")
    print(f"Flag rate: {is_anomaly.mean():.2%}")
    print(f"Known fault-label rate: {results['known_fault'].mean():.2%}")

    # This comparison is exploratory: it is not proof that every
    # flagged anomaly is a real fault or every fault is detectable.
    actual_fault = results["known_fault"].to_numpy()
    predicted_anomaly = results["is_anomaly"].to_numpy()

    true_positives = int(
        np.sum(actual_fault & predicted_anomaly)
    )
    false_positives = int(
        np.sum(~actual_fault & predicted_anomaly)
    )
    false_negatives = int(
        np.sum(actual_fault & ~predicted_anomaly)
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )

    print("\nExploratory comparison with known fault labels:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")

    plt.figure(figsize=(12, 5))
    plt.plot(
        results[TIME_COLUMN],
        results["anomaly_score"],
        linewidth=0.8,
        label="Anomaly score",
    )
    plt.scatter(
        results.loc[results["is_anomaly"], TIME_COLUMN],
        results.loc[results["is_anomaly"], "anomaly_score"],
        s=14,
        label="Flagged anomaly",
    )
    plt.xlabel("Timestamp")
    plt.ylabel("Anomaly score (higher = more unusual)")
    plt.title("Isolation Forest: Chronological Test Segment")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_plot = REPORTS_DIR / "anomaly_detection_scores.png"
    plt.savefig(output_plot, dpi=160)
    plt.close()

    print(f"\nSaved results: {output_csv}")
    print(f"Saved plot:    {output_plot}")


if __name__ == "__main__":
    main()
