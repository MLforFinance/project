"""
Downloads international macroeconomic data from the ECB Data Portal and OECD
Data Explorer and merges it with the FRED-MD dataset.

Sources
-------
* ECB Data Portal         : https://data.ecb.europa.eu/
* OECD CLI                : DSD_STES@DF_CLI  (OECD.SDD.STES)
* OECD KEI – Mfg. Prod.  : DSD_KEI@DF_KEI   (OECD.SDD.STES)
* OECD Prices (CPI)       : DSD_PRICES@DF_PRICES_ALL (OECD.SDD.TPS)
* OECD Labour – Unemp.    : DSD_LFS@DF_IALFS_INDIC   (OECD.SDD.TPS)
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRED_MD_PATH = PROJECT_ROOT / "data" / "2026-02-MD.csv"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "enriched_MD.csv"

# Restrict external series to this time window so the merged file stays aligned
# with the FRED-MD period.  Earlier data is sparse for ECB series anyway.
START_PERIOD = "1999-01"

# ── ECB configuration ─────────────────────────────────────────────────────────
# Actual ECB REST API endpoint (data-api.ecb.europa.eu, not data.ecb.europa.eu)
ECB_BASE = "https://data-api.ecb.europa.eu/service/data"

# Tuples: (dataset_id, series_key, output_column, transform_code)
#
# transform_code follows FRED-MD convention:
#   1 = level (xt)
#   2 = first difference (xt − xt-1)
#   3 = second difference
#   4 = log level (ln xt)
#   5 = first log-difference / growth rate (Δ ln xt)
#   6 = second log-difference
#   7 = first difference of change in log
#
# Keys verified against https://data-api.ecb.europa.eu/service/data/{dataset}/{key}
# If a series returns no data the script logs a warning and continues.
ECB_SERIES: list[tuple[str, str, str, int]] = [
    # Harmonised CPI – annual rate of change (already a growth-rate level → Δ)
    ("ICP", "M.U2.N.000000.4.ANR",            "ECB_HICP_YOY",   2),
    # EUR/USD monthly average exchange rate
    ("EXR", "M.USD.EUR.SP00.A",                "ECB_EURUSD",     5),
    # EURIBOR 3-month (European short-term benchmark rate)
    ("FM",  "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", "ECB_EURIBOR3M",  2),
    # ECB deposit facility rate (key monetary policy rate since 2022)
    ("FM",  "M.U2.EUR.MR.FR.DKEMPM.LEV.HSTA",  "ECB_DEP_RATE",   2),
    # M3 broad money supply – annual growth rate
    ("BSI", "M.U2.N.A.L20.A.1.U2.2300.Z01.E",  "ECB_M3_YOY",     2),
]

# ── OECD configuration ────────────────────────────────────────────────────────
OECD_BASE = "https://sdmx.oecd.org/public/rest/data"

# Keep only these country codes; the raw OECD responses cover all members.
KEEP_COUNTRIES: set[str] = {
    "USA", "GBR", "DEU", "FRA", "JPN", "CAN", "ITA",
    "EA20",   # Euro-area 20
    "OECD",   # OECD total
    "G7M",    # G7 average
}

# Keys come from the dq= parameter of the OECD Data Explorer URLs.  The country
# dimension is left as "." (= all countries), then filtered to KEEP_COUNTRIES
# after download.
OECD_DATASETS: list[dict] = [
    {
        "agency":   "OECD.SDD.STES",
        "dataflow": "DSD_STES@DF_CLI",
        "key":      ".M.LI...AA...H",
        "prefix":   "CLI",
        "tcode":    1,
        "desc":     "Composite Leading Indicator (amplitude-adjusted)",
    },
    {
        "agency":   "OECD.SDD.STES",
        "dataflow": "DSD_KEI@DF_KEI",
        "key":      ".M.PRVM.IX.BTE..",
        "prefix":   "MFG_PROD",
        "tcode":    5,
        "desc":     "Manufacturing production volume index",
    },
    {
        "agency":   "OECD.SDD.TPS",
        "dataflow": "DSD_PRICES@DF_PRICES_ALL",
        "key":      ".M.N.CPI.._T.N.GY",
        "prefix":   "CPI_YOY",
        "tcode":    2,
        "desc":     "CPI all items – annual growth rate",
    },
    {
        "agency":   "OECD.SDD.TPS",
        "dataflow": "DSD_LFS@DF_IALFS_INDIC",
        "key":      ".UNE_LF_M...Y._T.Y_GE15..M",
        "prefix":   "UNEMP",
        "tcode":    2,
        "desc":     "Unemployment rate (% labour force, 15+)",
    },
]


# ── Low-level HTTP helper ─────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; macro-enrichment/1.0; research)",
    "Accept": "text/csv, application/vnd.sdmx.data+csv, */*",
}


def _get(url: str, timeout: int = 60) -> requests.Response | None:
    """GET with 3 exponential-backoff retries; returns None on persistent failure."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            # Reject HTML responses (e.g. ECB's rate-limit / error pages)
            ct = r.headers.get("Content-Type", "")
            if "html" in ct.lower():
                log.warning("  received HTML instead of CSV (rate-limited?)")
                if attempt < 2:
                    wait = 4 ** attempt + 2
                    log.warning(f"  retry in {wait}s")
                    time.sleep(wait)
                    continue
                return None
            return r
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning(f"  attempt {attempt + 1} failed: {exc}  (retry in {wait}s)")
            time.sleep(wait)
    return None


# ── ECB fetch ─────────────────────────────────────────────────────────────────

def fetch_ecb_series(
    dataset_id: str, key: str, col: str
) -> pd.Series | None:
    """
    Download one ECB series and return it as a monthly-indexed Series.
    Returns None if the request fails or the response is malformed.
    """
    url = (
        f"{ECB_BASE}/{dataset_id}/{key}"
        f"?format=csvdata&startPeriod={START_PERIOD}"
    )
    log.info(f"ECB  {col}")
    log.info(f"     {url}")

    r = _get(url)
    if r is None:
        log.warning(f"ECB  {col}: skipped (request failed)")
        return None

    try:
        # ECB returns JSON 404 for unknown series keys
        if r.text.strip().startswith("{"):
            log.warning(f"ECB  {col}: series not found (check key)")
            return None
        df = pd.read_csv(io.StringIO(r.text))
        # ECB SDMX-CSV contains TIME_PERIOD and OBS_VALUE columns
        time_col = next((c for c in df.columns if "TIME_PERIOD" in c.upper()), None)
        val_col  = next((c for c in df.columns if "OBS_VALUE"   in c.upper()), None)
        if time_col is None or val_col is None:
            log.warning(
                f"ECB  {col}: unexpected columns {df.columns.tolist()}"
            )
            return None

        s = (
            df[[time_col, val_col]]
            .assign(**{time_col: pd.to_datetime(df[time_col])})
            .set_index(time_col)[val_col]
        )
        s = pd.to_numeric(s, errors="coerce")
        s.name = col
        log.info(f"     → {len(s)} observations")
        return s

    except Exception as exc:
        log.warning(f"ECB  {col}: parse error ({exc})")
        return None


# ── OECD fetch ────────────────────────────────────────────────────────────────

def fetch_oecd_dataset(
    agency: str, dataflow: str, key: str, prefix: str, tcode: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Download one OECD dataflow, pivot to wide format (one column per country),
    and return (DataFrame, tcode_dict).  Returns empty objects on failure.
    """
    url = (
        f"{OECD_BASE}/{agency},{dataflow}/{key}"
        f"?format=csv&startPeriod={START_PERIOD}"
    )
    log.info(f"OECD {prefix}")
    log.info(f"     {url}")

    r = _get(url, timeout=120)
    if r is None:
        log.warning(f"OECD {prefix}: skipped (request failed)")
        return pd.DataFrame(), {}

    try:
        df = pd.read_csv(io.StringIO(r.text))

        # Identify key columns robustly
        col_upper = {c.upper(): c for c in df.columns}
        time_col = (
            col_upper.get("TIME_PERIOD")
            or col_upper.get("TIME PERIOD")
            or col_upper.get("PERIOD")
        )
        val_col = (
            col_upper.get("OBS_VALUE")
            or col_upper.get("VALUE")
            or col_upper.get("OBSERVATION VALUE")
        )
        ref_col = (
            col_upper.get("REF_AREA")
            or col_upper.get("REFERENCE AREA")
            or col_upper.get("COUNTRY")
        )

        if not all([time_col, val_col, ref_col]):
            log.warning(
                f"OECD {prefix}: cannot identify standard columns; "
                f"found {df.columns.tolist()}"
            )
            return pd.DataFrame(), {}

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[time_col])

        # Filter to countries of interest
        present  = set(df[ref_col].unique())
        selected = present & KEEP_COUNTRIES
        if not selected:
            log.warning(
                f"OECD {prefix}: none of the target countries found; "
                f"sample present: {sorted(present)[:12]}"
            )
            return pd.DataFrame(), {}

        df = df[df[ref_col].isin(selected)]
        pivot = df.pivot_table(
            index=time_col, columns=ref_col, values=val_col, aggfunc="first"
        )
        pivot.columns = [f"OECD_{prefix}_{c}" for c in pivot.columns]
        tcodes = {c: tcode for c in pivot.columns}

        log.info(
            f"     → {pivot.shape[1]} country columns, {pivot.shape[0]} periods"
        )
        return pivot, tcodes

    except Exception as exc:
        log.warning(f"OECD {prefix}: parse error ({exc})")
        return pd.DataFrame(), {}


# ── FRED-MD loader ────────────────────────────────────────────────────────────

def load_fred_md(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Parse a FRED-MD CSV file.

    Returns
    -------
    data : pd.DataFrame
        Monthly observations indexed by month-start Timestamps.
    tcode_dict : dict[str, int]
        Transform code for each column.
    """
    raw = pd.read_csv(path, dtype=str)
    date_col = raw.columns[0]

    # Row 0 (after the header) is the "Transform:" row with integer codes
    tcode_series = raw.iloc[0, 1:]
    tcode_dict: dict[str, int] = {
        col: int(float(v))
        for col, v in tcode_series.items()
        if v not in ("", "nan", "None")
    }

    data = raw.iloc[1:].copy()
    data[date_col] = pd.to_datetime(data[date_col], format="%m/%d/%Y")
    data = data.set_index(date_col)
    data.index.name = "date"
    data = data.apply(pd.to_numeric, errors="coerce")

    # Normalise all timestamps to month-start (consistent with ECB/OECD)
    data.index = data.index.to_period("M").to_timestamp()

    return data, tcode_dict


# ── Date formatter ────────────────────────────────────────────────────────────

def _fmt_fred_date(ts: pd.Timestamp) -> str:
    """Format a Timestamp as M/1/YYYY (FRED-MD style, no zero-padding)."""
    return f"{ts.month}/{ts.day}/{ts.year}"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def create_enriched_dataset(
    fred_md_path: Path = FRED_MD_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Build the enriched dataset and write it to *output_path*.

    The output CSV has the same layout as FRED-MD:
      row 1 (after header): transform codes
      row 2+:               monthly observations, date in column 0
    """
    log.info("=" * 60)
    log.info("Step 1 – Loading FRED-MD …")
    fred_df, fred_tcodes = load_fred_md(fred_md_path)
    log.info(
        f"  {fred_df.shape[0]} rows × {fred_df.shape[1]} columns  "
        f"({fred_df.index[0].date()} – {fred_df.index[-1].date()})"
    )

    extra_frames: list[pd.DataFrame] = []
    extra_tcodes: dict[str, int] = {}

    # ── ECB ───────────────────────────────────────────────────────────────────
    log.info("\nStep 2 – Fetching ECB series …")
    ecb_parts: list[pd.Series] = []
    for dataset_id, key, col, tcode in ECB_SERIES:
        s = fetch_ecb_series(dataset_id, key, col)
        if s is not None:
            # Align to month-start timestamps
            s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
            ecb_parts.append(s)
            extra_tcodes[col] = tcode
        time.sleep(0.5)          # polite rate-limiting

    if ecb_parts:
        ecb_df = pd.concat(ecb_parts, axis=1)
        extra_frames.append(ecb_df)
        log.info(f"\nECB total: {ecb_df.shape[1]} series fetched successfully")

    # ── OECD ──────────────────────────────────────────────────────────────────
    log.info("\nStep 3 – Fetching OECD datasets …")
    for ds in OECD_DATASETS:
        frame, tc = fetch_oecd_dataset(
            ds["agency"], ds["dataflow"], ds["key"],
            ds["prefix"], ds["tcode"],
        )
        if not frame.empty:
            frame.index = pd.to_datetime(frame.index).to_period("M").to_timestamp()
            extra_frames.append(frame)
            extra_tcodes.update(tc)
        time.sleep(1.5)          # OECD responses are large; be gentle

    # ── Merge ─────────────────────────────────────────────────────────────────
    log.info("\nStep 4 – Merging …")
    enriched = fred_df.copy()
    for frame in extra_frames:
        enriched = enriched.join(frame, how="left")

    new_cols = enriched.shape[1] - fred_df.shape[1]
    log.info(
        f"  Enriched: {enriched.shape[0]} rows × {enriched.shape[1]} columns "
        f"(+{new_cols} new)"
    )

    # ── Rebuild transform-code row ─────────────────────────────────────────────
    all_tcodes = {**fred_tcodes, **extra_tcodes}
    tcode_row = pd.DataFrame(
        [[str(all_tcodes.get(c, 1)) for c in enriched.columns]],
        columns=enriched.columns,
        index=["Transform:"],
    )

    # ── Re-format date index to FRED-MD style (M/D/YYYY) ──────────────────────
    enriched.index = [_fmt_fred_date(ts) for ts in enriched.index]
    enriched.index.name = "sasdate"
    tcode_row.index.name = "sasdate"

    final = pd.concat([tcode_row, enriched])

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path)
    log.info(f"\nSaved → {output_path}")
    return enriched


# ── Public entry-point for other modules ─────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Convenience wrapper used by other modules in this project."""
    return create_enriched_dataset()


if __name__ == "__main__":
    create_enriched_dataset()
