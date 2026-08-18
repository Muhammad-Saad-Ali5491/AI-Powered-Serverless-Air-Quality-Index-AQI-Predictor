"""
Shared configuration for the AQI predictor - multi-city version.

Every pipeline stage (live fetch, backfill, feature engineering, training,
dashboard) imports CITIES from here, so adding/removing a city is a one-line
change instead of an edit in five different files.
"""

CITIES = {
    "Lahore":      {"lat": 31.5497, "lon": 74.3436},
    "Karachi":     {"lat": 24.8607, "lon": 67.0011},
    "Islamabad":   {"lat": 33.6844, "lon": 73.0479},
    "Faisalabad":  {"lat": 31.4504, "lon": 73.1350},
    "Rawalpindi":  {"lat": 33.5651, "lon": 73.0169},
    "Multan":      {"lat": 30.1575, "lon": 71.5249},
    "Peshawar":    {"lat": 34.0151, "lon": 71.5249},
}

# How many years of history to pull from Open-Meteo during backfill
BACKFILL_YEARS = 4

# Hopsworks feature group settings
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
