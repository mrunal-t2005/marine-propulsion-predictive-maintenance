
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "marine_propulsion.csv"

TARGET = "Fault_Label"
TIME = "Timestamp"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME], errors="coerce")

    feature_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col != TARGET
    ]

    print("\n--- 1. Target transition behaviour ---")
    labels = df[TARGET]
    changes = labels.ne(labels.shift()).iloc[1:]

    print(f"Adjacent label changes: {changes.sum():,}")
    print(f"Adjacent label change rate: {changes.mean():.2%}")

    run_id = labels.ne(labels.shift()).cumsum()
    runs = df.groupby(run_id, sort=False).agg(
        label=(TARGET, "first"),
        length=(TARGET, "size"),
    )

    print(f"Number of consecutive label runs: {len(runs):,}")
    print("Run length summary:")
    print(runs["length"].describe().to_string())

    print("\nMedian run length by label:")
    print(runs.groupby("label")["length"].median().to_string())

    print("\n--- 2. Sensor correlations ---")
    corr = df[feature_cols].corr().abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    pairs = upper.stack().sort_values(ascending=False)

    print("Top 10 absolute sensor correlations:")
    print(pairs.head(10).to_string())

    high_corr = pairs[pairs >= 0.95]
    print("\nPairs with absolute correlation >= 0.95:")
    print(high_corr.to_string() if not high_corr.empty else "None")

    print("\n--- 3. Sensor summaries by target label ---")
    grouped = df.groupby(TARGET)[feature_cols].mean().T
    print("Feature means by label:")
    print(grouped.round(3).to_string())

    # Normalised difference between class means and overall means.
    overall_mean = df[feature_cols].mean()
    overall_std = df[feature_cols].std(ddof=0).replace(0, np.nan)

    print("\nLargest absolute standardised mean deviations by label:")
    for label in sorted(df[TARGET].dropna().unique()):
        class_mean = df.loc[df[TARGET] == label, feature_cols].mean()
        deviation = ((class_mean - overall_mean) / overall_std).abs()
        print(f"\nLabel {label}:")
        print(deviation.sort_values(ascending=False).head(5).round(3).to_string())

    print("\n--- 4. Chronological split preview (70/15/15) ---")
    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    splits = {
        "Train": df.iloc[:train_end],
        "Validation": df.iloc[train_end:validation_end],
        "Test": df.iloc[validation_end:],
    }

    for name, part in splits.items():
        print(f"\n{name}: {len(part):,} rows")
        print(f"Time range: {part[TIME].min()} to {part[TIME].max()}")
        print(
            part[TARGET].value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
            .to_string()
        )

    print("\n--- 5. Basic data integrity ---")
    print(f"Non-finite numeric feature values: "
          f"{(~np.isfinite(df[feature_cols].to_numpy())).sum():,}")
    print(f"Unique target labels: {sorted(df[TARGET].dropna().unique())}")

    print("\nQuality checks complete.")


if __name__ == "__main__":
    main()
