"""
One-time (or periodic) backfill of historical AQI training data.

Pulls up to config.BACKFILL_YEARS of pollutant history from OpenAQ for
every configured Pakistani city, reshapes it into hourly wide-format rows
(one row per city/hour with all pollutant columns), runs it through the
same feature-engineering code path used by the live hourly pipeline, and
writes the result into the feature store.

Run:  python -m src.data.backfill_historical
"""
from __future__ import annotations
import argparse
import pandas as pd

from src import config
from src.data.fetch_openaq import fetch_city_historical
from src.features.feature_engineering import build_feature_table
from src.features.feature_store import get_feature_store
from src.utils.logging_utils import get_logger
from src.utils.paths import RAW_DIR

logger = get_logger(__name__)


def reshape_openaq_rows(rows: list[dict]) -> pd.DataFrame:
    """Long format (one row per pollutant reading) -> wide format
    (one row per city + hour, pollutants as columns)."""
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["date_utc"])
    df["timestamp"] = df["date_utc"].dt.floor("h")

    # normalize parameter names (OpenAQ uses "pm25", "pm10", "no2", "so2", "co", "o3")
    df["parameter"] = df["parameter"].astype(str).str.lower().str.replace(".", "", regex=False)

    pivot = (
        df.groupby(["city", "timestamp", "parameter"])["value"]
        .mean()
        .reset_index()
        .pivot_table(index=["city", "timestamp"], columns="parameter", values="value")
        .reset_index()
    )
    pivot.columns.name = None

    for pollutant in config.POLLUTANTS:
        if pollutant not in pivot.columns:
            pivot[pollutant] = pd.NA

    # backfill has no weather data from OpenAQ; fill neutral placeholders,
    # the hourly live pipeline will supply real weather going forward.
    for col in config.WEATHER_FEATURES:
        if col not in pivot.columns:
            pivot[col] = pd.NA

    return pivot


def run_backfill(years: int = config.BACKFILL_YEARS, cities=None) -> pd.DataFrame:
    cities = cities or config.CITIES
    all_rows: list[dict] = []
    for city in cities:
        try:
            rows = fetch_city_historical(city, years=years)
            all_rows.extend(rows)
        except Exception as exc:
            logger.error("Backfill failed for %s: %s", city.name, exc)

    raw_path = RAW_DIR / "openaq_backfill_raw.csv"
    if all_rows:
        pd.DataFrame(all_rows).to_csv(raw_path, index=False)
        logger.info("Saved raw OpenAQ backfill rows to %s", raw_path)

    wide_df = reshape_openaq_rows(all_rows)
    if wide_df.empty:
        logger.warning("Backfill produced no data (check OPENAQ_API_KEY / network / station coverage).")
        return pd.DataFrame()

    feature_df = build_feature_table(wide_df)
    store = get_feature_store()
    store.write_features(feature_df)
    logger.info("Backfill complete: %d feature rows written across %d cities.", len(feature_df), len(cities))
    return feature_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical AQI training data from OpenAQ")
    parser.add_argument("--years", type=int, default=config.BACKFILL_YEARS)
    parser.add_argument("--cities", nargs="*", default=None, help="Subset of city names, default = all")
    args = parser.parse_args()

    selected_cities = [config.get_city(c) for c in args.cities] if args.cities else None
    run_backfill(years=args.years, cities=selected_cities)
