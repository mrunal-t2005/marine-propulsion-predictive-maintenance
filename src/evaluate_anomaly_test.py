
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "marine_propulsion.csv"
REPORTS_DIR = ROOT / "reports"

TIME_COLUMN = "Timestamp"
LABEL_COLUMN = "Fault_Label"
NORMAL_LABEL = 0

RANDOM_STATE = 42
CONTAMINATION = 0.08

# Selected on the validation segment. Do not tune on the test segment.
FROZEN_THRESHOLD = -0.063393


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
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

    X = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    y = (
        df[LABEL_COLUMN].astype(int) != NORMAL_LABEL
    ).astype(int)

    split = int(len(df) * 0.85)
    X_development = X.iloc[:split]
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

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

    print("Training on the first 85% of chronological observations...")
    pipeline.fit(X_development)

    # The threshold is fixed before inspecting test results.
    test_scores = -pipeline.decision_function(X_test)
    y_pred = (test_scores >= FROZEN_THRESHOLD).astype(int)
    y_true = y_test.to_numpy()

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nFinal chronological test results")
    print(f"Frozen threshold: {FROZEN_THRESHOLD}")
    print(f"Test observations: {len(y_true)}")
    print(f"Fault-label prevalence: {y_true.mean():.2%}")
    print(f"Flag rate: {y_pred.mean():.2%}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 score: {f1:.3f}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Fault-labelled"],
            zero_division=0,
        )
    )

    results = df.iloc[split:][
        [TIME_COLUMN, LABEL_COLUMN]
    ].copy()
    results["anomaly_score"] = test_scores
    results["actual_fault"] = y_true
    results["predicted_anomaly"] = y_pred

    results.to_csv(
        REPORTS_DIR / "anomaly_final_test_predictions.csv",
        index=False,
    )

    metrics_text = (
        "Isolation Forest — final chronological test\n"
        f"Frozen threshold: {FROZEN_THRESHOLD}\n"
        f"Observations: {len(y_true)}\n"
        f"Fault-label prevalence: {y_true.mean():.4f}\n"
        f"Flag rate: {y_pred.mean():.4f}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1: {f1:.4f}\n"
        f"Confusion matrix:\n{matrix}\n"
    )

    (REPORTS_DIR / "anomaly_final_test_metrics.txt").write_text(
        metrics_text, encoding="utf-8"
    )

    print("\nSaved reports/anomaly_final_test_predictions.csv")
    print("Saved reports/anomaly_final_test_metrics.txt")
    print(
        "\nThis is the final test evaluation. "
        "Do not use these results to tune the threshold."
    )


if __name__ == "__main__":
    main()
