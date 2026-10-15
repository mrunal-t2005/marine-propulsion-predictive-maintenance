
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "marine_propulsion.csv"
REPORTS_DIR = PROJECT_DIR / "reports"

TARGET = "Fault_Label"


def main():
    df = pd.read_csv(DATA_PATH)

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

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Feature means by label ---")
    print(df.groupby(TARGET)[features].mean().round(3).to_string())

    print("\n--- Feature standard deviations by label ---")
    print(df.groupby(TARGET)[features].std().round(3).to_string())

    for feature in features:
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df, x=TARGET, y=feature, hue=TARGET, legend=False)
        plt.title(f"{feature} by fault label")
        plt.tight_layout()
        filename = feature.lower() + "_by_label.png"
        plt.savefig(REPORTS_DIR / filename, dpi=150)
        plt.close()

    print(f"\nSaved {len(features)} plots to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
