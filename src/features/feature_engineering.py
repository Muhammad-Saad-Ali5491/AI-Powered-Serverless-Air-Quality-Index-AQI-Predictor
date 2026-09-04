"""
Turn raw weather + pollutant rows into the model-ready feature table.

Computes:
  * time-based features (hour, day, month, day_of_week, is_weekend, day_of_year)
  * AQI (via EPA breakpoints) from raw pollutant concentrations
  * lag features (1h, 24h, 72h)
  * rolling statistics (24h mean/std)
  * AQI change rate (1h, 24h)
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from src.utils.aqi_calc import compute_aqi
from src import config


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(int)  # Fri/Sat weekend in Pakistan
    df["day_of_year"] = ts.dt.dayofyear
    return df


def add_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the AQI target column row-by-row from pollutant concentrations."""
    df = df.copy()

    def _row_aqi(row):
        concs = {p: row.get(p) for p in config.POLLUTANTS}
        return compute_aqi(concs)

    df["aqi"] = df.apply(_row_aqi, axis=1)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, group_col: str = "city", time_col: str = "timestamp") -> pd.DataFrame:
    """
    Add lag features and rolling stats, computed per-city on a time-sorted
    series. Assumes roughly hourly cadence; lag windows are expressed in
    "rows" which is acceptable because the feature pipeline runs hourly.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.sort_values([group_col, time_col])

    grouped = df.groupby(group_col, group_keys=False)

    df["aqi_lag_1h"] = grouped["aqi"].shift(1)
    df["aqi_lag_24h"] = grouped["aqi"].shift(24)
    df["aqi_lag_72h"] = grouped["aqi"].shift(72)

    df["aqi_rolling_mean_24h"] = grouped["aqi"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    df["aqi_rolling_std_24h"] = grouped["aqi"].transform(lambda s: s.rolling(24, min_periods=1).std())

    df["aqi_change_rate_1h"] = df["aqi"] - df["aqi_lag_1h"]
    df["aqi_change_rate_24h"] = df["aqi"] - df["aqi_lag_24h"]

    return df


def build_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw rows -> engineered feature table ready for the feature store."""
    df = add_time_features(raw_df)
    df = add_aqi_column(df)
    df = add_lag_and_rolling_features(df)

    # Fill remaining gaps sensibly instead of dropping rows outright
    for col in config.DERIVED_FEATURES:
        if col in df.columns:
            df[col] = df.groupby("city")[col].transform(lambda s: s.bfill().ffill()).fillna(0)

    for col in config.WEATHER_FEATURES + config.POLLUTANT_FEATURES:
        if col in df.columns:
            df[col] = df.groupby("city")[col].transform(lambda s: s.ffill().bfill())

    df = df.dropna(subset=["aqi"]).reset_index(drop=True)
    return df
