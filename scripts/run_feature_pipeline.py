"""
Hourly feature pipeline entry point (invoked by GitHub Actions on a cron
schedule, or manually / locally for testing).

Fetches the latest OpenWeather weather + air-pollution snapshot for every
configured Pakistani city, engineers features, and writes them to the
feature store.

Run:  python scripts/run_feature_pipeline.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/run_feature_pipeline.py`

import pandas as pd

from src import config
from src.data.fetch_openweather import fetch_city_snapshot, OpenWeatherError
from src.features.feature_engineering import build_feature_table
from src.features.feature_store import get_feature_store
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run() -> pd.DataFrame:
    if not config.OPENWEATHER_API_KEY:
        logger.error("OPENWEATHER_API_KEY is not set — cannot fetch live data.")
        sys.exit(1)

    snapshots = []
    for city in config.CITIES:
        try:
            snapshot = fetch_city_snapshot(city)
            snapshots.append(snapshot)
            logger.info("Fetched snapshot for %s", city.name)
        except OpenWeatherError as exc:
            logger.error("Skipping %s: %s", city.name, exc)

    if not snapshots:
        logger.error("No snapshots fetched for any city — aborting.")
        sys.exit(1)

    raw_df = pd.DataFrame(snapshots)

    # Feature engineering needs history for lag/rolling features, so we
    # combine this run's snapshot with existing stored history first.
    store = get_feature_store()
    existing = store.read_features()
    combined_raw = pd.concat([existing, raw_df], ignore_index=True) if not existing.empty else raw_df

    feature_df = build_feature_table(combined_raw)
    # Only write back the newly-computed rows that match this run's timestamps
    new_rows = feature_df[feature_df["timestamp"].isin(pd.to_datetime(raw_df["timestamp"], utc=True))]
    store.write_features(new_rows)

    logger.info("Feature pipeline run complete: %d new rows written.", len(new_rows))
    return new_rows


if __name__ == "__main__":
    run()
