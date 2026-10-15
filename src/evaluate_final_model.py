
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "marine_propulsion.csv"
REPORTS_DIR = PROJECT_DIR / "reports"

TARGET = "Fault_Label"
TIME = "Timestamp"

CLASS1_BOOST = 2.4
OTHER_FAULT_BOOST = 1.3

CLASS_NAMES = [
    "Normal",
    "Fuel injection fault",
    "Cylinder pressure loss",
    "Exhaust overheating",
    "Bearing/vibration fault",
    "Lubrication degradation",
    "Turbocharger failure",
    "Mixed fault",
]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and chronologically sort observations.
    df = pd.read_csv(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME], errors="raise")
    df = df.sort_values(TIME).reset_index(drop=True)

    feature_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col != TARGET
    ]

    # Freeze the final split: first 85% for fitting,
    # last 15% for one final evaluation.
    split_index = int(len(df) * 0.85)

    development = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = development[feature_cols]
    y_train = development[TARGET].astype(int)

    X_test = test[feature_cols]
    y_test = test[TARGET].astype(int)

    print(f"Total rows: {len(df)}")
    print(f"Development rows: {len(development)}")
    print(f"Final test rows: {len(test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Test period: {test[TIME].min()} to {test[TIME].max()}")

    # Train the fixed XGBoost configuration on development data.
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=8,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    print("\nTraining final candidate...")
    model.fit(X_train, y_train, sample_weight=weights)

    # Apply the multipliers selected on validation data.
    probabilities = model.predict_proba(X_test)
    classes = model.classes_
    adjusted = probabilities.copy()

    for column, class_label in enumerate(classes):
        if class_label == 1:
            adjusted[:, column] *= CLASS1_BOOST
        elif class_label != 0:
            adjusted[:, column] *= OTHER_FAULT_BOOST

    adjusted /= adjusted.sum(axis=1, keepdims=True)
    predictions = classes[np.argmax(adjusted, axis=1)]

    # Evaluate once on the reserved chronological test segment.
    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(8)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print("\nPer-class report:")
    print(report)

    cm = confusion_matrix(y_test, predictions, labels=list(range(8)))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    # Save the final test confusion matrix.
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(8),
        yticks=np.arange(8),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Final Test Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    matrix_path = REPORTS_DIR / "final_test_confusion_matrix.png"
    fig.savefig(matrix_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    metrics_path = REPORTS_DIR / "final_test_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as file:
        file.write("Final chronological test evaluation\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Balanced accuracy: {balanced_acc:.4f}\n")
        file.write(f"Macro F1: {macro_f1:.4f}\n")
        file.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        file.write(report)
        file.write("\nClass-1 multiplier: 2.4\n")
        file.write("Other fault multiplier: 1.3\n")

    print(f"\nSaved confusion matrix: {matrix_path}")
    print(f"Saved metrics report: {metrics_path}")
    print("\nFinal test evaluation complete.")
    print("Do not tune the model using these test results.")


if __name__ == "__main__":
    main()
