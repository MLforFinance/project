from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .mrsq import mrsq
    from .factors_em import impute_by_method, IMPUTATION_METHODS
    from .remove_outliers import remove_outliers_by_method, OUTLIER_METHODS
    from .prepare_missing import prepare_missing, compute_NA
except ImportError:  # pragma: no cover - supports direct script execution
    from mrsq import mrsq
    from factors_em import impute_by_method, IMPUTATION_METHODS
    from remove_outliers import remove_outliers_by_method, OUTLIER_METHODS
    from prepare_missing import prepare_missing, compute_NA


DEFAULT_CSV_PATH = Path("data/2026-02-MD.csv")
DEMEAN = 2
jj = 2
kmax = 8

# Group 6: Interest and exchange rates
GROUP_6_COLUMNS = (
    "FEDFUNDS",
    "CP3Mx",
    "TB3MS",
    "TB6MS",
    "GS1",
    "GS5",
    "GS10",
    "AAA",
    "BAA",
    "COMPAPFFx",
    "TB3SMFFM",
    "TB6SMFFM",
    "T1YFFM",
    "T5YFFM",
    "T10YFFM",
    "AAAFFM",
    "BAAFFM",
    "TWEXAFEGSMTHx",
    "EXSZUSx",
    "EXJPUSx",
    "EXUSUKx",
    "EXCAUSx",
)


def infer_burn_in_rows(tcode) -> int:
    burn_in_by_tcode = {
        1: 0,
        2: 1,
        3: 2,
        4: 0,
        5: 1,
        6: 2,
        7: 2,
    }
    return max((burn_in_by_tcode.get(int(code), 0) for code in tcode), default=0)


def preprocess_fred_md(
    input_path: str | Path,
    output_path: str | Path | None = None,
    trim_rows: int | None = None,
    excluded_columns: tuple[str, ...] = (),
    demean: int = DEMEAN,
    jj_value: int = jj,
    kmax_value: int = kmax,
    outlier_method: str = "global",
    imputation_method: str = "em",
    imputation_burn_in: int = 60,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    input_path = Path(input_path)
    df = pd.read_csv(input_path)
    date_column = df.columns[0]

    excluded_present = [column for column in excluded_columns if column in df.columns]
    if excluded_present:
        df = df.drop(columns=excluded_present)

    series_names = df.columns[1:].values
    tcode = df.iloc[0, 1:].values.astype(int)
    rawdata = df.iloc[1:, 1:].values.astype(float)

    burn_in_rows = infer_burn_in_rows(tcode) if trim_rows is None else trim_rows
    date_index = pd.to_datetime(df.iloc[1 + burn_in_rows :, 0])
    date_index.name = date_column

    if verbose:
        print((compute_NA(df) == 0).all())

    yt = prepare_missing(rawdata, tcode)
    yt = yt[burn_in_rows:, :]

    n_missing_after_prepare_missing = int(pd.DataFrame(yt).isna().sum().sum())
    data, n_outliers = remove_outliers_by_method(yt, method=outlier_method)
    n_missing_after_remove_outliers = int(pd.DataFrame(data).isna().sum().sum())

    _, Fhat, lamhat, ve2, x2 = impute_by_method(
        data,
        method=imputation_method,
        kmax=kmax_value,
        jj=jj_value,
        demean_type=demean,
        burn_in=imputation_burn_in,
    )
    R2, mR2, mR2_F, R2_T, t10_s, t10_mR2 = mrsq(Fhat, lamhat, ve2, series_names)

    if verbose:
        print(f"Total variance explained: {R2_T:.2%}")

    transformed_df = pd.DataFrame(x2, columns=series_names, index=date_index)

    if verbose:
        print((compute_NA(transformed_df) == 0).all())

    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(transformed_df),
        columns=series_names,
        index=date_index,
    )
    scaled_data.index.name = date_column

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scaled_data.to_csv(output_path)

    preprocessing_info = {
        "input_path": input_path,
        "output_path": Path(output_path) if output_path is not None else None,
        "burn_in_rows": burn_in_rows,
        "excluded_columns": excluded_present,
        "n_missing_after_prepare_missing": n_missing_after_prepare_missing,
        "n_missing_after_remove_outliers": n_missing_after_remove_outliers,
        "n_outliers_total": int(n_outliers.sum()),
        "outlier_method": outlier_method,
        "imputation_method": imputation_method,
        "imputation_burn_in": imputation_burn_in if imputation_method == "em_burnin" else None,
        "output_shape": scaled_data.shape,
        "variance_explained": float(R2_T),
        "mrsq": {
            "R2": R2,
            "mR2": mR2,
            "mR2_F": mR2_F,
            "top_10_series": t10_s,
            "top_10_marginal_r2": t10_mR2,
        },
    }
    return scaled_data, preprocessing_info


def main() -> tuple[pd.DataFrame, dict]:
    output_path = DEFAULT_CSV_PATH.with_name(f"{DEFAULT_CSV_PATH.stem}_processed.csv")
    processed_df, preprocessing_info = preprocess_fred_md(
        DEFAULT_CSV_PATH,
        output_path,
        verbose=True,
    )
    print(f"Processed data saved to: {output_path}")
    return processed_df, preprocessing_info


if __name__ == "__main__":
    main()
