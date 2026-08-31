"""
US EPA Air Quality Index (AQI) calculation from raw pollutant concentrations.

OpenAQ and OpenWeather both return raw concentrations (ug/m3 or ppm), not a
ready-made AQI, so we compute the official EPA AQI ourselves using the
standard breakpoint tables. The overall AQI for a given time/place is the
MAX of the individual pollutant sub-indices, per EPA methodology.

Reference breakpoints: US EPA "Technical Assistance Document for the
Reporting of Daily Air Quality" (concentrations truncated per pollutant
before lookup, as EPA specifies).
"""
from __future__ import annotations
import math
from typing import Optional

import pandas as pd

# Each tuple: (C_low, C_high, I_low, I_high)
_PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

_PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

# NO2 in ppb
_NO2_BREAKPOINTS = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

# SO2 in ppb (1-hr for lower ranges per EPA table)
_SO2_BREAKPOINTS = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]

# CO in ppm (8-hr)
_CO_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

# O3 in ppb (8-hr, values above 200 use the 1-hr table in reality; simplified here)
_O3_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 70, 51, 100),
    (71, 85, 101, 150),
    (86, 105, 151, 200),
    (106, 200, 201, 300),
]

_BREAKPOINTS = {
    "pm25": _PM25_BREAKPOINTS,
    "pm10": _PM10_BREAKPOINTS,
    "no2": _NO2_BREAKPOINTS,
    "so2": _SO2_BREAKPOINTS,
    "co": _CO_BREAKPOINTS,
    "o3": _O3_BREAKPOINTS,
}

# Unit conversion helpers: OpenAQ/OpenWeather usually report ug/m3; EPA gas
# breakpoints are in ppb/ppm. Rough conversions at 25C, 1atm.
_UGM3_TO_PPB = {
    "no2": 1 / 1.88,
    "so2": 1 / 2.62,
    "o3": 1 / 2.00,
}
_UGM3_TO_PPM_CO = 1 / 1145.0


def _truncate(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def _sub_index(pollutant: str, concentration: Optional[float]) -> Optional[float]:
    # pd.isna() safely handles every "missing" sentinel we might encounter
    # here — None, float('nan'), numpy.nan, and pandas' own pd.NA/pd.NaT —
    # without raising. A plain `concentration != concentration` or
    # `math.isnan()` check is NOT enough: pd.NA in a numeric comparison
    # (e.g. `pd.NA < 0`) returns pd.NA itself, and evaluating that in a
    # boolean context (`if pd.NA:`) raises
    # "TypeError: boolean value of NA is ambiguous" rather than behaving
    # like a normal falsy/missing value.
    if pd.isna(concentration):
        return None
    if concentration < 0:
        return None

    conc = concentration
    if pollutant == "pm25":
        conc = _truncate(conc, 1)
    elif pollutant in ("pm10",):
        conc = _truncate(conc, 0)
    elif pollutant == "co":
        conc = _truncate(conc, 1)
    elif pollutant in ("no2", "so2", "o3"):
        conc = _truncate(conc, 0)

    table = _BREAKPOINTS[pollutant]
    for c_low, c_high, i_low, i_high in table:
        if c_low <= conc <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low)

    # Above the top of the table -> cap at 500 (hazardous, off the charts)
    if conc > table[-1][1]:
        return 500
    return None


def convert_units(pollutant: str, value_ugm3: float) -> float:
    """Convert an OpenAQ/OpenWeather ug/m3 reading into the unit EPA expects."""
    if pollutant == "co":
        return value_ugm3 * _UGM3_TO_PPM_CO
    if pollutant in _UGM3_TO_PPB:
        return value_ugm3 * _UGM3_TO_PPB[pollutant]
    return value_ugm3  # pm25 / pm10 already in ug/m3


def compute_aqi(pollutant_concentrations_ugm3: dict) -> Optional[int]:
    """
    Compute overall US EPA AQI from a dict of pollutant -> concentration
    (all inputs in ug/m3, the common unit returned by OpenAQ/OpenWeather).

    Returns the max sub-index across all available pollutants (EPA rule),
    or None if no pollutant data was available at all.
    """
    sub_indices = []
    for pollutant, raw_value in pollutant_concentrations_ugm3.items():
        if pollutant not in _BREAKPOINTS or raw_value is None:
            continue
        converted = convert_units(pollutant, raw_value)
        idx = _sub_index(pollutant, converted)
        if idx is not None:
            sub_indices.append(idx)

    if not sub_indices:
        return None
    return int(max(sub_indices))


def aqi_category(aqi: Optional[int]) -> str:
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"
