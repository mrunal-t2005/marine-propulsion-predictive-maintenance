
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "marine_propulsion.csv"
REPORTS_DIR = PROJECT_DIR / "reports"

TARGET = "Fault_Label"
TIME = "Timestamp"

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

    # Load and sort chronologically.
    df = pd.read_csv(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME], errors="raise")
    df = df.sort_values(TIME).reset_index(drop=True)

    feature_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col != TARGET
    ]

    # Use only the development segment.
    # The final 15% remains excluded from this explanation workflow.
    development_end = int(len(df) * 0.85)
    development = df.iloc[:development_end]

    X = development[feature_cols]
    y = development[TARGET].astype(int)

    # Train the selected weighted XGBoost configuration.
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
        y=y,
    )

    print("Training XGBoost on the development segment...")
    model.fit(X, y, sample_weight=weights)

    # A representative sample keeps SHAP computation manageable.
    background = X.sample(
        n=min(100, len(X)),
        random_state=42,
    )
    explanation_data = X.sample(
        n=min(300, len(X)),
        random_state=7,
    )

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(explanation_data)

    # XGBoost multiclass SHAP commonly returns:
    # (rows, features, classes). Handle class-first arrays too.
    if isinstance(shap_values, list):
        values = [
            pd.DataFrame(
                class_values,
                columns=feature_cols,
            )
            for class_values in shap_values
        ]
        mean_abs = pd.concat(
            [frame.abs().mean(axis=0) for frame in values],
            axis=1,
        )
        mean_abs.columns = CLASS_NAMES
        mean_abs["Overall"] = mean_abs.mean(axis=1)

    else:
        import numpy as np

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            if shap_values.shape[1] == len(feature_cols):
                # rows x features x classes
                mean_abs = pd.DataFrame(
                    np.abs(shap_values).mean(axis=0),
                    index=feature_cols,
                )
            elif shap_values.shape[2] == len(feature_cols):
                # rows x classes x features
                mean_abs = pd.DataFrame(
                    np.abs(shap_values).mean(axis=0).T,
                    index=feature_cols,
                )
            else:
                raise ValueError(
                    f"Unexpected SHAP array shape: {shap_values.shape}"
                )

            mean_abs.columns = CLASS_NAMES
            mean_abs["Overall"] = mean_abs.mean(axis=1)

        elif shap_values.ndim == 2:
            # Single-output fallback.
            mean_abs = pd.DataFrame(
                {"Overall": np.abs(shap_values).mean(axis=0)},
                index=feature_cols,
            )
        else:
            raise ValueError(
                f"Unexpected SHAP array shape: {shap_values.shape}"
            )

    mean_abs = mean_abs.sort_values("Overall", ascending=False)
    csv_path = REPORTS_DIR / "shap_feature_importance.csv"
    mean_abs.to_csv(csv_path, index_label="Feature")

    print("\nTop 10 features by mean absolute SHAP value:")
    print(mean_abs["Overall"].head(10).to_string())

    # Save global feature importance plot.
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_abs["Overall"].head(15).sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global SHAP Feature Importance")
    fig.tight_layout()

    plot_path = REPORTS_DIR / "shap_feature_importance.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Explain one example from the sampled development data.
    row = explanation_data.iloc[[0]]
    predicted_class = int(model.predict(row)[0])
    print("\nExample observation:")
    print(f"Timestamp: {df.loc[row.index[0], TIME]}")
    print(f"Actual class: {int(y.loc[row.index[0]])} "
          f"({CLASS_NAMES[int(y.loc[row.index[0]])]})")
    print(f"Predicted class: {predicted_class} "
          f"({CLASS_NAMES[predicted_class]})")

    print(f"\nSaved: {plot_path}")
    print(f"Saved: {csv_path}")
    print("\nSHAP explanation complete.")
    print("No final-test observations were used.")


if __name__ == "__main__":
    main()
