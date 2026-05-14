"""Solar geometry features using pvlib.

Planned features:
    - solar_zenith, solar_azimuth (per timestamp, per station lat/lon)
    - day_length (sunrise to sunset duration)
    - clearsky_ghi (theoretical clear-sky global horizontal irradiance)
    - is_daylight (zero radiation indicator at night)

Implementation note: pvlib expects timezone-aware timestamps. We have UTC
implied from TAHMO; verify against per-station local-time radiation peaks
during EDA before committing to a timezone assumption.
"""
from __future__ import annotations

import pandas as pd


def add_solar_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: implement zenith/azimuth/clearsky/day_length via pvlib."""
    raise NotImplementedError(
        "Solar geometry features not yet implemented. See module docstring."
    )
