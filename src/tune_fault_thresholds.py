

from pathlib import Path

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

TARGET = "Fault_Label"
TIME = "Timestamp"


def main():
    # 1. Load and prepare the dataset
    df = pd.read_csv(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME], errors="raise")
    df = df.sort_values(TIME).reset_index(drop=True)

    feature_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col != TARGET
    ]

    # 2. Chronological split: 70% training, 15% validation, 15% test
    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]

    X_train = train[feature_cols]
    y_train = train[TARGET].astype(int)

    X_val = validation[feature_cols]
    y_val = validation[TARGET].astype(int)

    print(f"Dataset rows: {n}")
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Features: {len(feature_cols)}")

    # 3. Configure and train class-weighted XGBoost
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=df[TARGET].nunique(),
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

    print("\nTraining class-weighted XGBoost...")
    model.fit(X_train, y_train, sample_weight=weights)

    probabilities = model.predict_proba(X_val)
    classes = model.classes_

    # 4. Evaluate standard predictions
    baseline_predictions = classes[np.argmax(probabilities, axis=1)]

    print("\n--- Standard weighted XGBoost ---")
    print(f"Accuracy:          {accuracy_score(y_val, baseline_predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_val, baseline_predictions):.4f}"
    )
    print(
        "Macro F1:          "
        f"{f1_score(y_val, baseline_predictions, average='macro', zero_division=0):.4f}"
    )

    print(
        classification_report(
            y_val,
            baseline_predictions,
            labels=list(classes),
            zero_division=0,
        )
    )

    # 5. Tune class 1 separately.
    # Keep the other fault classes at a multiplier of 1.30.
    OTHER_FAULT_BOOST = 1.30
    best = None

    for class1_boost in np.round(np.arange(1.0, 3.01, 0.1), 2):
        adjusted = probabilities.copy()

        for column, class_label in enumerate(classes):
            if class_label == 1:
                adjusted[:, column] *= class1_boost
            elif class_label != 0:
                adjusted[:, column] *= OTHER_FAULT_BOOST

        adjusted /= adjusted.sum(axis=1, keepdims=True)
        predictions = classes[np.argmax(adjusted, axis=1)]

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )
        balanced_acc = balanced_accuracy_score(y_val, predictions)
        accuracy = accuracy_score(y_val, predictions)

        # Select by macro F1; use balanced accuracy to break ties.
        score = (macro_f1, balanced_acc)

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "class1_boost": float(class1_boost),
                "predictions": predictions,
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_acc,
                "accuracy": accuracy,
            }

    # 6. Report the best validation configuration
    print("\n--- Best class-1 multiplier experiment ---")
    print(f"Other fault multiplier: {OTHER_FAULT_BOOST:.2f}")
    print(f"Class-1 multiplier:     {best['class1_boost']:.2f}")
    print(f"Accuracy:               {best['accuracy']:.4f}")
    print(f"Balanced accuracy:      {best['balanced_accuracy']:.4f}")
    print(f"Macro F1:               {best['macro_f1']:.4f}")

    print(
        classification_report(
            y_val,
            best["predictions"],
            labels=list(classes),
            zero_division=0,
        )
    )

    # 7. Print the confusion matrix once
    cm = confusion_matrix(
        y_val,
        best["predictions"],
        labels=list(classes),
    )

    print("\n--- Confusion matrix: class-1 tuned ---")
    print("Rows = actual class; columns = predicted class")
    print(pd.DataFrame(cm, index=classes, columns=classes))

    print("\nValidation-based model selection is complete.")
    print("The final test segment has not been evaluated.")


if __name__ == "__main__":
    main()
