"""
Feature engineering for the AQI predictor - multi-city version.

Takes the raw accumulated rows (from Hopsworks or a CSV export, now
containing multiple cities) and computes derived features: lags, rolling
stats, AQI/PM2.5 change rate, and cyclical time encodings.

CRITICAL: every derived feature is computed PER CITY (grouped), never
across the whole dataframe at once. Without this, a lag/rolling feature for
Karachi could accidentally pull in a value from Lahore's rows just because
they're adjacent after sorting by timestamp - a subtle but serious bug in
multi-city time series data.

Also gap-aware: lag features use merge_asof with a tolerance window rather
than a plain positional .shift(), so they don't silently return a stale
value across a multi-hour data gap.
"""

import numpy as np
import pandas as pd


def load_and_prepare(path_or_df):
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["city", "timestamp"]).drop_duplicates(
        subset=["city", "timestamp"]
    ).reset_index(drop=True)
    return df


def _lag_one_city(city_df, target_cols, lag_hours, tolerance_hours):
    city_df = city_df.sort_values("timestamp").reset_index(drop=True)
    result = city_df.copy()

    for col in target_cols:
        for lag in lag_hours:
            lookup = city_df[["timestamp", col]].copy()
            lookup["timestamp"] = lookup["timestamp"] + pd.Timedelta(hours=lag)
            lookup = lookup.rename(columns={col: f"{col}_lag_{lag}h"})

            result = pd.merge_asof(
                result.sort_values("timestamp"),
                lookup.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta(hours=tolerance_hours),
            )
    return result


def add_lag_features(df, target_cols=("pm2_5", "temp"), lag_hours=(1, 6, 24), tolerance_hours=2):
    pieces = []
    for city, city_df in df.groupby("city", sort=False):
        pieces.append(_lag_one_city(city_df, target_cols, lag_hours, tolerance_hours))
    return pd.concat(pieces, ignore_index=True).sort_values(["city", "timestamp"]).reset_index(drop=True)


def _rolling_one_city(city_df, target_cols, windows):
    city_df = city_df.sort_values("timestamp").set_index("timestamp")
    for col in target_cols:
        for window in windows:
            city_df[f"{col}_roll_mean_{window}"] = city_df[col].rolling(window).mean()
            city_df[f"{col}_roll_std_{window}"] = city_df[col].rolling(window).std()
    return city_df.reset_index()


def add_rolling_features(df, target_cols=("pm2_5", "temp"), windows=("6h", "24h")):
    pieces = []
    for city, city_df in df.groupby("city", sort=False):
        pieces.append(_rolling_one_city(city_df, target_cols, windows))
    return pd.concat(pieces, ignore_index=True).sort_values(["city", "timestamp"]).reset_index(drop=True)


def add_change_rate(df, col="pm2_5", lag_col="pm2_5_lag_6h", hours=6):
    df[f"{col}_change_rate_{hours}h"] = (df[col] - df[lag_col]) / hours
    return df


def add_cyclical_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_city_encoding(df):
    """One-hot encode city so a single model can learn city-specific patterns
    (e.g. Karachi's coastal wind behavior vs Lahore's inland stagnation)."""
    dummies = pd.get_dummies(df["city"], prefix="city")
    return pd.concat([df, dummies], axis=1)


def engineer_features(path_or_df):
    """Main entry point: raw multi-city rows in, fully engineered dataframe out."""
    df = load_and_prepare(path_or_df)

    # pm2_5 is the primary target across BOTH data sources (live OpenWeather +
    # historical Open-Meteo), since it's on a consistent concentration scale.
    # 'aqi' (OpenWeather's 1-5 scale) is kept as an auxiliary feature only for
    # live rows where it exists - see backfill_openmeteo.py's docstring.
    df = add_lag_features(df, target_cols=("pm2_5", "temp"), lag_hours=(1, 6, 24))
    df = add_rolling_features(df, target_cols=("pm2_5", "temp"), windows=("6h", "24h"))
    df = add_change_rate(df, col="pm2_5", lag_col="pm2_5_lag_6h", hours=6)
    df = add_cyclical_time_features(df)
    df = add_city_encoding(df)

    return df


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "aqi_data_export.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "aqi_features_engineered.csv"

    df = engineer_features(input_path)
    df.to_csv(output_path, index=False)

    print(f"Engineered {len(df)} rows, {len(df.columns)} columns -> {output_path}")
    print(df.groupby("city").size())
