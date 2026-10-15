
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def main() -> None:
    files = [
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("~$")
    ]

    if not files:
        print(f"No CSV or Excel dataset found in: {DATA_DIR}")
        print("Place your dataset in data/raw/ and run this script again.")
        return

    if len(files) > 1:
        print("Multiple datasets found:")
        for index, path in enumerate(files, start=1):
            print(f"{index}. {path.name}")

        print("For now, keep only the dataset you want to inspect in data/raw/.")
        return

    path = files[0]
    print(f"\nDataset: {path.name}")

    try:
        df = load_dataset(path)
    except Exception as exc:
        print(f"Could not load dataset: {exc}")
        return

    print("\n--- Dataset dimensions ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\n--- Column names and data types ---")
    print(df.dtypes.to_string())

    print("\n--- First five rows ---")
    print(df.head().to_string())

    print("\n--- Missing values ---")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.to_string() if not missing.empty else "No missing values.")

    print("\n--- Duplicate rows ---")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    print("\n--- Numeric summary ---")
    print(df.describe(include="number").T.to_string())

    print("\n--- Non-numeric summary ---")
    non_numeric = df.select_dtypes(exclude="number")
    if non_numeric.shape[1] > 0:
        print(non_numeric.describe().T.to_string())
    else:
        print("No non-numeric columns.")

    print("\n--- Candidate target columns ---")
    candidates = [
        column
        for column in df.columns
        if any(term in str(column).lower()
               for term in ("fault", "label", "target", "class", "status"))
    ]
    print(candidates if candidates else "No obvious target column found.")

    print("\n--- Dataset inspection complete ---")


if __name__ == "__main__":
    main()
