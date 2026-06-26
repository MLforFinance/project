"""Usage example for FredFeatureBuilder."""

import pandas as pd

from macro_prediction.fred import FredFeatureBuilder, FredSeriesSpec


class ExampleFredClient:
    """Tiny fredapi stand-in so the example runs without network access."""

    def get_series(
        self,
        series_id,
        observation_start=None,
        observation_end=None,
        realtime_start=None,
        realtime_end=None,
    ):
        if series_id == "DAILY_RATE":
            return pd.Series(
                [1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
                index=pd.to_datetime([
                    "2020-01-03",
                    "2020-01-31",
                    "2020-02-07",
                    "2020-02-28",
                    "2020-03-06",
                    "2020-03-27",
                ]),
            )
        if series_id == "MONTHLY_INDEX":
            return pd.Series(
                [100.0, 105.0, 111.0],
                index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            )
        if series_id == "QUARTERLY_GDP":
            return pd.Series(
                [1000.0],
                index=pd.to_datetime(["2020-01-01"]),
            )
        raise KeyError(series_id)


def test_fred_feature_builder_usage_example() -> None:
    series = [
        FredSeriesSpec(
            "DAILY_RATE",
            transformation="level",
            frequency="daily_or_weekly",
            monthly_aggregation="mean",
        ),
        FredSeriesSpec(
            "MONTHLY_INDEX",
            transformation="log_diff",
            frequency="monthly",
        ),
        FredSeriesSpec(
            "QUARTERLY_GDP",
            transformation="level",
            frequency="quarterly",
        ),
    ]
    builder = FredFeatureBuilder(series, fred=ExampleFredClient())

    raw = builder.fetch_raw(
        observation_start="2020-01-01",
        vintage_date="2024-12-31",
    )

    assert raw.loc["2020-01-31", "DAILY_RATE"] == 1.5
    assert raw.loc["2020-02-29", "DAILY_RATE"] == 3.5
    assert raw.loc["2020-01-31", "QUARTERLY_GDP"] == 1000.0
    assert raw.loc["2020-02-29", "QUARTERLY_GDP"] == 1000.0
    assert raw.loc["2020-03-31", "QUARTERLY_GDP"] == 1000.0

    features = builder.fit_transform(raw)

    assert list(features.columns) == ["DAILY_RATE", "MONTHLY_INDEX", "QUARTERLY_GDP"]
    assert features.index.tolist() == [pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31")]
