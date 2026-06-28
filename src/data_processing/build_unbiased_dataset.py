from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("raw_data")
BASE_VINTAGE = "1999-08.csv"


def parse_vintage_date(filename: str) -> tuple[int, int]:
    """Return (year, month) for a vintage filename."""
    name = Path(filename).stem
    # FRED-MD_2024m03
    m = re.match(r"FRED-MD_(\d{4})m(\d{2})", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    # 2025-04-MD or 1999-08
    m = re.match(r"(\d{4})-(\d{2})", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot parse vintage date from: {filename}")


def read_vintage(path: Path, last_row_only: bool) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, low_memory=False)
    # Row 0 is the "Transform:" row — drop it, keep data rows
    data = df.iloc[1:].copy()
    if last_row_only:
        data = data.iloc[[-1]]
    date_col = df.columns[0]
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(date_col)
    data.index.name = "sasdate"
    data = data.apply(pd.to_numeric, errors="coerce")
    return data


def build_unbiased_dataset(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    files = sorted(raw_data_dir.glob("*.csv"), key=lambda p: parse_vintage_date(p.name))

    chunks: list[pd.DataFrame] = []
    for path in files:
        is_base = path.name == BASE_VINTAGE
        chunk = read_vintage(path, last_row_only=not is_base)
        chunks.append(chunk)

    combined = pd.concat(chunks, join="inner", axis=0)
    # Keep first occurrence of each date (base file has the biased history;
    # subsequent vintages may overlap on the same date — the last-row extraction
    # already ensures each vintage contributes only its newest observation).
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


def check_last_rows_present(dataset: pd.DataFrame, raw_data_dir: Path = RAW_DATA_DIR) -> None:
    files = [p for p in raw_data_dir.glob("*.csv") if p.name != BASE_VINTAGE]
    failures: list[str] = []
    for path in files:
        last_row = read_vintage(path, last_row_only=True)
        date = last_row.index[0]
        if date not in dataset.index:
            failures.append(f"{path.name}: date {date.date()} not found in dataset")

    if failures:
        raise AssertionError("Missing rows:\n" + "\n".join(failures))
    print(f"OK — all {len(files)} vintages have their last row in the dataset")


if __name__ == "__main__":
    output_path = Path("data/macro_unbiased.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_unbiased_dataset()
    df.to_csv(output_path)
    print(f"Saved {df.shape[0]} rows × {df.shape[1]} columns to {output_path}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    check_last_rows_present(df)
