from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def discover_input_csv(data_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in data_dir.glob("*.csv")
        if not path.stem.endswith(("_processed", "_reduced", "_regimes"))
    )
    if not candidates:
        raise FileNotFoundError(f"No raw CSV files found in {data_dir}")
    return candidates[0]


input_csv = discover_input_csv(DATA_DIR)
df = pd.read_csv(input_csv)

total_rows = len(df)
rows_with_nan = df.isna().any(axis=1).sum()
percent_nan_rows = (rows_with_nan / total_rows) * 100

print(f"Input CSV: {input_csv}")
print(f"{percent_nan_rows:.2f}% of rows contain at least one NaN value")
