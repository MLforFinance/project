"""Live FRED data fetching, transformations, and scaling."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Literal, Protocol

import numpy as np
import pandas as pd
from fredapi import Fred
from sklearn.preprocessing import StandardScaler

Transformation = Literal[
    "level",
    "diff",
    "pct_change",
    "log",
    "log_diff",
    "yoy_pct_change",
    "yoy_log_diff",
]
Frequency = Literal["daily_or_weekly", "monthly", "quarterly"]
MonthlyAggregation = Literal["mean", "last", "sum", "min", "max"]


@dataclass(frozen=True)
class FredSeriesSpec:
    """Configuration for one FRED series used as a model feature.

    ``frequency`` controls how observations are converted to monthly rows before
    any feature transformation is applied. Daily/weekly series use
    ``monthly_aggregation``. Quarterly series are repeated into all three months
    of their quarter.
    """

    series_id: str
    name: str | None = None
    transformation: Transformation = "level"
    frequency: Frequency = "monthly"
    monthly_aggregation: MonthlyAggregation = "mean"

    @property
    def column_name(self) -> str:
        return self.name or self.series_id


class FredLike(Protocol):
    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.Series:
        ...


class FredFeatureBuilder:
    """Fetch FRED series with fredapi, transform them, and standard-scale features.

    Use ``fit_transform`` on the training window. Then use ``transform`` on later
    data with the same fitted scaler to avoid look-ahead leakage.

    ``vintage_date`` is passed to FRED as both ``realtime_start`` and
    ``realtime_end`` so the returned observations reflect values available as of
    that date, assuming the FRED endpoint supports the requested realtime query.
    """

    def __init__(
        self,
        series: Iterable[FredSeriesSpec],
        *,
        fred: FredLike | None = None,
        api_key: str | None = None,
        standardize: bool = True,
        drop_missing: bool = True,
    ) -> None:
        self.series = list(series)
        if not self.series:
            raise ValueError("At least one FRED series must be configured.")
        self.fred = fred
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.standardize = standardize
        self.drop_missing = drop_missing
        self.scaler: StandardScaler | None = None
        self.columns_: list[str] | None = None

    def fetch_raw(
        self,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
        vintage_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch configured FRED series and convert them to monthly rows."""

        if self.fred is None:
            self.fred = Fred(api_key=self.api_key)

        request_args = {
            "observation_start": observation_start,
            "observation_end": observation_end,
        }
        if vintage_date is not None:
            request_args["realtime_start"] = vintage_date
            request_args["realtime_end"] = vintage_date

        monthly_series = []
        for spec in self.series:
            values = self.fred.get_series(spec.series_id, **request_args)
            monthly = to_monthly(
                values,
                frequency=spec.frequency,
                aggregation=spec.monthly_aggregation,
            )
            monthly.name = spec.column_name
            monthly_series.append(monthly)
        return pd.concat(monthly_series, axis=1).sort_index()

    def transform_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Apply per-series transformations without fitting or scaling."""

        features = pd.DataFrame(index=raw.index)
        for spec in self.series:
            column = spec.column_name
            if column not in raw.columns:
                raise ValueError(f"Raw data is missing expected column: {column}")
            features[column] = apply_transformation(raw[column], spec.transformation)

        features = features.replace([np.inf, -np.inf], np.nan)
        if self.drop_missing:
            features = features.dropna(axis=0, how="any")
        return features.astype(float)

    def fit(self, raw: pd.DataFrame) -> "FredFeatureBuilder":
        transformed = self.transform_raw(raw)
        if self.standardize:
            self.scaler = StandardScaler().fit(transformed)
        self.columns_ = list(transformed.columns)
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transform_raw(raw)
        if self.columns_ is not None:
            transformed = transformed.reindex(columns=self.columns_)
        if not self.standardize:
            return transformed
        if self.scaler is None:
            raise RuntimeError("FredFeatureBuilder must be fit before transform.")
        values = self.scaler.transform(transformed)
        return pd.DataFrame(values, index=transformed.index, columns=transformed.columns)

    def fit_transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        self.fit(raw)
        return self.transform(raw)

    def fetch_transform(
        self,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
        vintage_date: str | None = None,
        fit: bool = False,
    ) -> pd.DataFrame:
        """Fetch raw FRED data, then transform and optionally fit the scaler."""

        raw = self.fetch_raw(
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_date=vintage_date,
        )
        if fit:
            return self.fit_transform(raw)
        return self.transform(raw)


def to_monthly(
    series: pd.Series,
    *,
    frequency: Frequency,
    aggregation: MonthlyAggregation = "mean",
) -> pd.Series:
    """Convert a FRED series to one month-end value per month."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    values.index = pd.to_datetime(values.index)
    values = values.sort_index()

    if frequency == "daily_or_weekly":
        return aggregate_to_month(values, aggregation)
    if frequency == "monthly":
        values.index = values.index.to_period("M").to_timestamp("M")
        return values.groupby(level=0).last()
    if frequency == "quarterly":
        return expand_quarterly_to_monthly(values)
    raise ValueError(f"Unsupported frequency: {frequency}")


def aggregate_to_month(series: pd.Series, aggregation: MonthlyAggregation) -> pd.Series:
    """Aggregate higher-frequency observations to monthly observations."""

    grouped = series.resample("ME")
    if aggregation == "mean":
        return grouped.mean()
    if aggregation == "last":
        return grouped.last()
    if aggregation == "sum":
        return grouped.sum()
    if aggregation == "min":
        return grouped.min()
    if aggregation == "max":
        return grouped.max()
    raise ValueError(f"Unsupported monthly aggregation: {aggregation}")


def expand_quarterly_to_monthly(series: pd.Series) -> pd.Series:
    """Repeat each quarterly observation into every month of that quarter."""

    monthly_values: dict[pd.Timestamp, float] = {}
    for date, value in series.items():
        quarter = pd.Timestamp(date).to_period("Q")
        months = pd.period_range(
            quarter.asfreq("M", "start"),
            quarter.asfreq("M", "end"),
            freq="M",
        )
        for month in months:
            monthly_values[month.to_timestamp("M")] = value
    return pd.Series(monthly_values, dtype=float).sort_index()


def apply_transformation(series: pd.Series, transformation: Transformation) -> pd.Series:
    """Apply one feature transformation to a FRED series."""

    x = pd.to_numeric(series, errors="coerce").astype(float)
    if transformation == "level":
        return x
    if transformation == "diff":
        return x.diff()
    if transformation == "pct_change":
        return x.pct_change()
    if transformation == "log":
        return np.log(x)
    if transformation == "log_diff":
        return np.log(x).diff()
    if transformation == "yoy_pct_change":
        return x.pct_change(12)
    if transformation == "yoy_log_diff":
        return np.log(x).diff(12)
    raise ValueError(f"Unsupported transformation: {transformation}")


def default_fredmd_representative_series() -> list[FredSeriesSpec]:
    """A compact FRED-MD-inspired macro feature set.

    This is not the full FRED-MD panel. It is a small representative basket for
    first-pass regime modeling: activity, labor, housing, consumption, prices,
    money/credit, and rates. Transformations are chosen to make most series more
    stationary before standard scaling.
    """

    return [
        # Output / income / production
        FredSeriesSpec("RPI", transformation="log_diff", frequency="monthly"),
        FredSeriesSpec("INDPRO", transformation="log_diff", frequency="monthly"),
        FredSeriesSpec("CUMFNS", transformation="diff", frequency="monthly"),

        # Labor market
        FredSeriesSpec("PAYEMS", transformation="log_diff", frequency="monthly"),
        FredSeriesSpec("UNRATE", transformation="diff", frequency="monthly"),
        FredSeriesSpec("ICSA", transformation="log_diff", frequency="daily_or_weekly", monthly_aggregation="mean"),

        # Housing
        FredSeriesSpec("HOUST", transformation="log_diff", frequency="monthly"),
        FredSeriesSpec("PERMIT", transformation="log_diff", frequency="monthly"),

        # Consumption / orders
        FredSeriesSpec("RSAFS", transformation="log_diff", frequency="monthly"),
        FredSeriesSpec("DGORDER", transformation="log_diff", frequency="monthly"),

        # Prices
        FredSeriesSpec("CPIAUCSL", transformation="yoy_log_diff", frequency="monthly"),
        FredSeriesSpec("PCEPI", transformation="yoy_log_diff", frequency="monthly"),
        FredSeriesSpec("PPIACO", transformation="yoy_log_diff", frequency="monthly"),

        # Money / credit
        FredSeriesSpec("M2SL", transformation="log_diff", frequency="monthly"),
        FredSeriesSpec("BUSLOANS", transformation="log_diff", frequency="daily_or_weekly", monthly_aggregation="mean"),

        # Rates / spreads
        FredSeriesSpec("FEDFUNDS", transformation="level", frequency="monthly"),
        FredSeriesSpec("GS10", transformation="level", frequency="monthly"),
        FredSeriesSpec("T10Y3M", transformation="level", frequency="daily_or_weekly", monthly_aggregation="mean"),
    ]
