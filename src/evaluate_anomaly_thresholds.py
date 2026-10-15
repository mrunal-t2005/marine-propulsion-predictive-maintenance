
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
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
        c for c in df.select_dtypes(include=np.number).columns
        if c != LABEL_COLUMN
    ]

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = (df[LABEL_COLUMN].astype(int) != NORMAL_LABEL).astype(int)

    split = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    X_train = X.iloc[:split]
    X_val = X.iloc[split:validation_end]
    y_val = y.iloc[split:validation_end]

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

    print("Training on the first 70% of chronological observations...")
    pipeline.fit(X_train)

    # Higher scores mean more unusual observations.
    scores = -pipeline.decision_function(X_val)
    precision, recall, thresholds = precision_recall_curve(
        y_val, scores
    )
    ap = average_precision_score(y_val, scores)

    # precision_recall_curve returns one more precision/recall point
    # than threshold values. Use only points associated with thresholds.
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_idx = int(np.argmax(f1))
    selected_threshold = float(thresholds[best_idx])

    # Evaluate the fixed Isolation Forest decision threshold as well.
    default_scores = scores
    default_pred = pipeline.predict(X_val) == -1
    default_tp = int(np.sum(default_pred & y_val.to_numpy().astype(bool)))
    default_fp = int(np.sum(default_pred & ~y_val.to_numpy().astype(bool)))
    default_fn = int(np.sum(~default_pred & y_val.to_numpy().astype(bool)))

    default_precision = (
        default_tp / (default_tp + default_fp)
        if default_tp + default_fp else 0.0
    )
    default_recall = (
        default_tp / (default_tp + default_fn)
        if default_tp + default_fn else 0.0
    )

    print("\nValidation results (15% middle segment):")
    print(f"Fault-label prevalence: {y_val.mean():.2%}")
    print(f"Average precision:      {ap:.3f}")
    print("\nDefault Isolation Forest threshold:")
    print(f"Precision: {default_precision:.3f}")
    print(f"Recall:    {default_recall:.3f}")

    print("\nBest F1 threshold on validation scores:")
    print(f"Threshold: {selected_threshold:.6f}")
    print(f"Precision: {precision[best_idx]:.3f}")
    print(f"Recall:    {recall[best_idx]:.3f}")
    print(f"F1:        {f1[best_idx]:.3f}")

    report = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "f1": f1,
    })
    report.to_csv(
        REPORTS_DIR / "anomaly_threshold_metrics.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label=f"Validation AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Anomaly Score: Precision–Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "anomaly_precision_recall_curve.png",
        dpi=160,
    )
    plt.close()

    print("\nSaved reports/anomaly_threshold_metrics.csv")
    print("Saved reports/anomaly_precision_recall_curve.png")
    print(
        "\nThreshold selection is for validation only. "
        "Do not use the final 15% test segment to tune it."
    )


if __name__ == "__main__":
    main()
