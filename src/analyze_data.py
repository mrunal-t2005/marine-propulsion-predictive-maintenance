
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "marine_propulsion.csv"
REPORTS_DIR = PROJECT_DIR / "reports"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Validate and parse timestamps.
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce"
    )

    print("\n--- Timestamp validation ---")
    print(f"Invalid timestamps: {df['Timestamp'].isna().sum()}")

    valid_times = df["Timestamp"].dropna()

    if not valid_times.empty:
        print(f"First timestamp: {valid_times.min()}")
        print(f"Last timestamp:  {valid_times.max()}")
        print(f"Timestamp span:   {valid_times.max() - valid_times.min()}")
        print(f"Duplicate timestamps: {valid_times.duplicated().sum()}")
        print(
            "Already chronological: "
            f"{df['Timestamp'].is_monotonic_increasing}"
        )

        ordered_times = valid_times.sort_values()
        intervals = ordered_times.diff().dropna()

        print("\nMost common time intervals:")
        print(intervals.value_counts().head(10).to_string())

        print(
            "\nIntervals not equal to one second: "
            f"{(intervals != pd.Timedelta(seconds=1)).sum():,}"
        )

    # Inspect the target distribution.
    print("\n--- Fault label distribution ---")

    counts = df["Fault_Label"].value_counts(dropna=False).sort_index()
    percentages = (
        df["Fault_Label"].value_counts(normalize=True, dropna=False)
        .sort_index()
        .mul(100)
    )

    distribution = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2),
    })

    print(distribution.to_string())
    print(f"\nNumber of distinct labels: {df['Fault_Label'].nunique()}")

    if df["Fault_Label"].isna().any():
        print("WARNING: The target contains missing labels.")

    # Examine how labels are distributed over chronological segments.
    print("\n--- Label distribution over five time segments ---")

    chronological = (
        df.dropna(subset=["Timestamp"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if len(chronological) >= 5:
        chronological["Time_Segment"] = pd.qcut(
            range(len(chronological)),
            q=5,
            labels=["Segment 1", "Segment 2", "Segment 3",
                    "Segment 4", "Segment 5"],
        )

        segment_counts = pd.crosstab(
            chronological["Time_Segment"],
            chronological["Fault_Label"],
        )

        print(segment_counts.to_string())

    # Save a class distribution chart.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ax = counts.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Fault Label Distribution")
    ax.set_xlabel("Fault_Label")
    ax.set_ylabel("Number of observations")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()

    chart_path = REPORTS_DIR / "fault_label_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nChart saved to: {chart_path}")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
