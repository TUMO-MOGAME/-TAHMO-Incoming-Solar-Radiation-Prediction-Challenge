"""Solar geometry features via pvlib.

For each station's (latitude, longitude, elevation), compute:
    - solar_zenith       apparent zenith angle in degrees (90 = horizon)
    - solar_elevation    90 - zenith; sun's angle above horizon
    - cos_zenith         cos(zenith); negative at night
    - clearsky_ghi       theoretical clear-sky Global Horizontal Irradiance (W/m^2)
    - clearsky_dni       Direct Normal Irradiance (W/m^2)
    - clearsky_dhi       Diffuse Horizontal Irradiance (W/m^2)
    - is_daylight        1 if zenith < 90 else 0

These features encode the physics that the LightGBM baseline is currently
approximating with `hour * temperature`. Because they depend only on
location + timestamp (not on year), they are year-invariant by
construction — useful for closing the year-overfit flag from exp_001.

Assumption: timestamps in the dataset are UTC (the TAHMO standard).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
from tqdm import tqdm

SOLAR_FEATURE_COLS = [
    "solar_zenith",
    "solar_elevation",
    "cos_zenith",
    "clearsky_ghi",
    "clearsky_dni",
    "clearsky_dhi",
    "is_daylight",
]


def add_solar_geometry_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    station_col: str = "station",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    elevation_col: str = "elevation",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Append solar geometry feature columns. Does not mutate `df`."""
    out = df.copy()
    for col in SOLAR_FEATURE_COLS:
        out[col] = np.nan

    groups = out.groupby(station_col, sort=False)
    iterator = tqdm(groups, desc="solar features", total=len(groups)) if show_progress else groups

    for station, group in iterator:
        lat = float(group[lat_col].iloc[0])
        lon = float(group[lon_col].iloc[0])
        elev = float(group[elevation_col].iloc[0])

        times = pd.DatetimeIndex(group[timestamp_col]).tz_localize("UTC")

        solpos = pvlib.solarposition.get_solarposition(times, lat, lon, altitude=elev)
        zenith = solpos["apparent_zenith"].to_numpy()
        elevation = solpos["apparent_elevation"].to_numpy()

        location = pvlib.location.Location(lat, lon, altitude=elev)
        clearsky = location.get_clearsky(times)

        idx = group.index
        out.loc[idx, "solar_zenith"] = zenith
        out.loc[idx, "solar_elevation"] = elevation
        out.loc[idx, "cos_zenith"] = np.cos(np.radians(zenith))
        out.loc[idx, "clearsky_ghi"] = clearsky["ghi"].to_numpy()
        out.loc[idx, "clearsky_dni"] = clearsky["dni"].to_numpy()
        out.loc[idx, "clearsky_dhi"] = clearsky["dhi"].to_numpy()

    out["is_daylight"] = (out["solar_zenith"] < 90).astype(np.int8)
    return out