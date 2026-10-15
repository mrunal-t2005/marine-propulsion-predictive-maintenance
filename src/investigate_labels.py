
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "marine_propulsion.csv"

TARGET = "Fault_Label"
TIME = "Timestamp"


def main():
    df = pd.read_csv(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME], errors="raise")
    df = df.sort_values(TIME).reset_index(drop=True)

    print("--- Consecutive label run lengths ---")
    run_id = df[TARGET].ne(df[TARGET].shift()).cumsum()
    runs = df.groupby(run_id).agg(
        label=(TARGET, "first"),
        start=(TIME, "first"),
        end=(TIME, "last"),
        length=(TARGET, "size"),
    )

    print("\nRun-length summary by label:")
    print(
        runs.groupby("label")["length"]
        .agg(["count", "median", "mean", "max"])
        .to_string()
    )

    print("\n--- Label transitions ---")
    transitions = pd.crosstab(
        df[TARGET].shift().rename("previous"),
        df[TARGET].rename("current"),
    )
    print(transitions.to_string())

    print("\n--- Sensor means by label ---")
    sensors = df.select_dtypes(include="number").columns.drop(TARGET)
    means = df.groupby(TARGET)[sensors].mean().round(3)
    print(means.to_string())

    print("\nInvestigation complete. No model was trained.")


if __name__ == "__main__":
    main()
