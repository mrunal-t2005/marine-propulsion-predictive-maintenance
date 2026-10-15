
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "marine_propulsion.csv"
REPORTS_DIR = PROJECT_DIR / "reports"

TARGET = "Fault_Label"
TIME = "Timestamp"


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n--- {name} ---")
    print(f"Accuracy:          {accuracy_score(y_test, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )
    print(f"Macro F1:          {f1_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1:       {f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
    print("\nPer-class report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=sorted(y_train.unique()),
            zero_division=0,
        )
    )

    return model, predictions


def main():
    df = pd.read_csv(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME], errors="raise")
    df = df.sort_values(TIME).reset_index(drop=True)

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing labels.")

    feature_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col != TARGET
    ]

    # Chronological 70% train, 15% validation, 15% test.
    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]
    test = df.iloc[validation_end:]

    # Use validation for initial model comparison.
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_val = validation[feature_cols]
    y_val = validation[TARGET]

    print("Features:", feature_cols)
    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Final test rows reserved: {len(test)}")

    dummy = DummyClassifier(strategy="most_frequent")

    forest = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )

    _, dummy_predictions = evaluate_model(
        "Dummy baseline", dummy, X_train, y_train, X_val, y_val
    )

    forest, forest_predictions = evaluate_model(
        "Random Forest", forest, X_train, y_train, X_val, y_val
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_val,
        forest_predictions,
        labels=sorted(y_train.unique()),
        xticks_rotation="vertical",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Random Forest — Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / "baseline_confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    print("\nValidation experiment complete.")
    print("Confusion matrix saved to reports/baseline_confusion_matrix.png")
    print("The final test set has not been used.")


if __name__ == "__main__":
    main()
