"""
Feature Store abstraction.

Two backends, chosen by src.config.USE_HOPSWORKS (defaults to True):

  1. HopsworksFeatureStore (DEFAULT) — writes feature rows to a real
     managed Hopsworks Feature Store (feature group with online + offline
     storage, versioning, and a query API), using the free serverless tier
     at app.hopsworks.ai. Requires HOPSWORKS_API_KEY. Every write ALSO
     lands in the local Parquet cache, and bulk reads always come from
     that cache (fast, no query-engine round trip needed for training).

  2. LocalFeatureStore (automatic fallback) — stores the feature table as
     partitioned Parquet files under data/features/, committed to the repo
     by GitHub Actions. Used automatically whenever Hopsworks isn't
     configured (no HOPSWORKS_API_KEY) or is temporarily unreachable, so
     the pipeline never hard-fails for lack of a Hopsworks account — set
     USE_HOPSWORKS=false to use this as the primary store instead.

Both backends expose the same three methods: write_features, read_features,
get_latest. The rest of the codebase only talks to `get_feature_store()`,
so switching backends never requires touching training/inference code.
"""
from __future__ import annotations
import abc
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config
from src.utils.paths import FEATURE_STORE_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

FEATURE_TABLE_PATH = FEATURE_STORE_DIR / "aqi_features.parquet"


class BaseFeatureStore(abc.ABC):
    @abc.abstractmethod
    def write_features(self, df: pd.DataFrame) -> None: ...

    @abc.abstractmethod
    def read_features(self, city: Optional[str] = None) -> pd.DataFrame: ...

    def get_latest(self, city: str) -> Optional[pd.Series]:
        df = self.read_features(city=city)
        if df.empty:
            return None
        df = df.sort_values("timestamp")
        return df.iloc[-1]


class LocalFeatureStore(BaseFeatureStore):
    """Parquet-file-backed feature store, versioned via git in CI."""

    def __init__(self, path: Path = FEATURE_TABLE_PATH):
        self.path = path

    def write_features(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.info("No new feature rows to write.")
            return

        if self.path.exists():
            existing = pd.read_parquet(self.path)
            combined = pd.concat([existing, df], ignore_index=True)
            # de-dupe on (city, timestamp) keeping the newest write
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
            combined = combined.sort_values("timestamp")
            combined = combined.drop_duplicates(subset=["city", "timestamp"], keep="last")
        else:
            combined = df

        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.path, index=False)
        logger.info("Wrote %d rows to local feature store (%d total).", len(df), len(combined))

    def read_features(self, city: Optional[str] = None) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self.path)
        if city:
            df = df[df["city"].str.lower() == city.lower()]
        return df.reset_index(drop=True)


class HopsworksFeatureStore(BaseFeatureStore):
    """
    Feature store backend for Hopsworks (hsfs / hopsworks-py client).

    Activates when USE_HOPSWORKS=true. Requires HOPSWORKS_API_KEY (and,
    for self-managed clusters, HOPSWORKS_HOST + HOPSWORKS_PROJECT_NAME —
    leave those blank to use the free managed serverless tier, which
    defaults to the "c.app.hopsworks.ai" host below).

    Every write also lands in the local Parquet cache first, so bulk
    training reads (which need full history, not point lookups) stay fast
    and the pipeline keeps working even if Hopsworks is briefly
    unreachable.
    """

    # The public hostname for Hopsworks' free managed serverless tier, as
    # opposed to app.hopsworks.ai (the web UI's hostname, not the API's).
    _SERVERLESS_HOST = "c.app.hopsworks.ai"

    def __init__(self):
        if not config.HOPSWORKS_API_KEY:
            raise ValueError("HOPSWORKS_API_KEY must be set when USE_HOPSWORKS=true.")

        import hopsworks  # imported lazily so it's optional

        # IMPORTANT: always pass an explicit, non-empty `host` — never
        # leave it to hopsworks.login()'s own env-var/default resolution.
        # In "external" execution contexts (GitHub Actions, plain scripts,
        # CI runners — anything that isn't an interactive Hopsworks-hosted
        # notebook), that internal resolution raises
        # "ExternalClientError: host cannot be of type NoneType" whenever
        # the HOPSWORKS_HOST environment variable is *present but empty*
        # rather than fully absent — which is exactly what GitHub Actions'
        # `env:` blocks produce when a repo variable was never configured
        # (the variable is still set, just to an empty string). Passing
        # host explicitly here sidesteps that ambiguity entirely.
        login_kwargs = {
            "host": config.HOPSWORKS_HOST or self._SERVERLESS_HOST,
            "api_key_value": config.HOPSWORKS_API_KEY,
        }
        if config.HOPSWORKS_PROJECT_NAME:
            login_kwargs["project"] = config.HOPSWORKS_PROJECT_NAME

        self._project = hopsworks.login(**login_kwargs)
        self._fs = self._project.get_feature_store()
        self._local_cache = LocalFeatureStore()
        self._fg = None
        logger.info(
            "Initialized Hopsworks Feature Store client (project=%s, feature_group=%s v%d)",
            self._project.name,
            config.HOPSWORKS_FEATURE_GROUP_NAME,
            config.HOPSWORKS_FEATURE_GROUP_VERSION,
        )

    def _get_or_create_feature_group(self, sample_df: pd.DataFrame):
        if self._fg is not None:
            return self._fg

        # Hopsworks needs a stable numeric/string event-time column and a
        # primary key. We derive a plain int64 unix-seconds column from
        # 'timestamp' since HSFS event_time works best on numeric/timestamp
        # dtypes, and use (city, timestamp) as the composite primary key.
        self._fg = self._fs.get_or_create_feature_group(
            name=config.HOPSWORKS_FEATURE_GROUP_NAME,
            version=config.HOPSWORKS_FEATURE_GROUP_VERSION,
            description="Pearls AQI Predictor — engineered hourly AQI features for Pakistani cities",
            primary_key=["city", "event_time_unix"],
            event_time="event_time_unix",
            online_enabled=True,
        )
        return self._fg

    @staticmethod
    def _with_event_time(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["event_time_unix"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
        # HSFS/Spark-Avro schema inference is happiest with plain strings for
        # timestamps rather than tz-aware pandas Timestamps.
        df["timestamp"] = df["timestamp"].astype(str)
        return df

    def write_features(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.info("No new feature rows to write.")
            return

        # Always cache locally first (cheap + fast + offline-safe).
        self._local_cache.write_features(df)

        try:
            hopsworks_df = self._with_event_time(df)
            fg = self._get_or_create_feature_group(hopsworks_df)
            fg.insert(hopsworks_df, write_options={"wait_for_job": False})
            logger.info("Inserted %d rows into Hopsworks feature group '%s'.", len(df), fg.name)
        except Exception as exc:  # pragma: no cover - network/Hopsworks dependent
            logger.error("Hopsworks write failed, local cache still has the data: %s", exc)

    def read_features(self, city: Optional[str] = None) -> pd.DataFrame:
        # Bulk training reads use the local Parquet cache (fast, no Spark
        # job / query engine round-trip needed). To read the authoritative
        # copy straight from Hopsworks instead, use read_from_hopsworks().
        return self._local_cache.read_features(city=city)

    def read_from_hopsworks(self, city: Optional[str] = None) -> pd.DataFrame:
        """Bypass the local cache and read directly from the Hopsworks
        offline feature store (useful for verifying a sync, or for a
        training pipeline running on a machine without the local cache)."""
        fg = self._get_or_create_feature_group(pd.DataFrame())
        query = fg.select_all()
        if city:
            query = query.filter(fg.city == city)
        df = query.read()
        if "event_time_unix" in df.columns:
            df = df.drop(columns=["event_time_unix"])
        return df


@lru_cache(maxsize=1)
def get_feature_store() -> BaseFeatureStore:
    """
    Returns the Hopsworks-backed feature store by default. If Hopsworks
    isn't configured (no HOPSWORKS_API_KEY) or is temporarily unreachable,
    this automatically falls back to the local Parquet feature store so
    the pipeline never hard-fails — set USE_HOPSWORKS=false to skip the
    Hopsworks attempt entirely and always use local storage.
    """
    if config.USE_HOPSWORKS:
        if not config.HOPSWORKS_API_KEY:
            logger.info(
                "USE_HOPSWORKS=true (default) but HOPSWORKS_API_KEY is not set — "
                "using the local Parquet feature store instead. Set HOPSWORKS_API_KEY "
                "to enable Hopsworks (see README -> 'Using Hopsworks')."
            )
        else:
            try:
                return HopsworksFeatureStore()
            except Exception as exc:
                logger.warning(
                    "Hopsworks client could not be initialized (%s). "
                    "Falling back to the local Parquet feature store.",
                    exc,
                )
    return LocalFeatureStore()
