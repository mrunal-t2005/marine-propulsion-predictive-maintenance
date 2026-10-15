
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

    df["Time_Segment"] = pd.cut(
        range(len(df)),
        bins=[-1, 6999, 8499, 9999],
        labels=["Train", "Validation", "Test"],
    )

    features = [
        "Fuel_Flow",
        "Air_Pressure",
        "Oil_Temp",
        "Oil_Pressure",
        "Vibration_X",
        "Vibration_Y",
        "Vibration_Z",
        "Cylinder1_Pressure",
        "Cylinder1_Exhaust_Temp",
    ]

    means = df.groupby(
        ["Time_Segment", TARGET], observed=True
    )[features].mean().round(3)

    print("--- Feature means by chronological segment and label ---")
    print(means.to_string())

    print("\n--- Row counts by segment and label ---")
    print(
        pd.crosstab(df["Time_Segment"], df[TARGET]).to_string()
    )


if __name__ == "__main__":
    main()
