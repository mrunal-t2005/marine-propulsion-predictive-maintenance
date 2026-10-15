
from pathlib import Path
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "marine_propulsion.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
TARGET = "Fault_Label"
TIME = "Timestamp"


def main():
    df = pd.read_csv(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME], errors="raise")
    df = df.sort_values(TIME).reset_index(drop=True)

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    feature_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col != TARGET
    ]

    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_val = validation[feature_cols]
    y_val = validation[TARGET]

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

    
    print("Training XGBoost with balanced class weights...")

    sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

    model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
)

    predictions = model.predict(X_val)


    print("\n--- XGBoost validation results ---")
    print(f"Accuracy:          {accuracy_score(y_val, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_val, predictions):.4f}"
    )
    print(
        "Macro F1:          "
        f"{f1_score(y_val, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        "Weighted F1:       "
        f"{f1_score(y_val, predictions, average='weighted', zero_division=0):.4f}"
    )

    print("\nPer-class report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=sorted(y_train.unique()),
            zero_division=0,
        )
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_val,
        predictions,
        labels=sorted(y_train.unique()),
        xticks_rotation="vertical",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("XGBoost — Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / "xgboost_confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols,
        name="Importance",
    ).sort_values(ascending=False)

    print("\n--- Top 10 XGBoost features ---")
    print(importance.head(10).to_string())

    ax = importance.head(10).sort_values().plot(
        kind="barh", figsize=(9, 6)
    )
    ax.set_title("XGBoost — Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "xgboost_feature_importance.png",
        dpi=150,
    )
    plt.close()

    print("\nSaved validation confusion matrix and feature importance.")
    print("The final test set has not been used.")


if __name__ == "__main__":
    main()
