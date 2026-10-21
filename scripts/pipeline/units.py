"""
units.py — Single source of truth for unit handling and physical bounds
========================================================================
Every loader (scalar, gridded, station extraction) goes through these helpers so
AORC and NLDAS-3 are converted and cleaned identically.

Precipitation units are parsed from the NetCDF ``units`` attribute instead of
guessed from substrings: the previous rule ``'s' in units or 'kg' in units`` also
matched daily accumulations ("kg m-2"), multiplying them by 86400.
"""

from __future__ import annotations
import re
import numpy as np

# Per-pixel physical plausibility bounds (analysis units after conversion).
# Values outside are fill values or corrupt data and become NaN BEFORE any
# spatial averaging, for both products alike.
PHYS_BOUNDS: dict[str, tuple[float, float]] = {
    'precip':            (0.0,    500.0),    # mm/day, per pixel
    'temp':              (220.0,  330.0),    # K
    'temp_max':          (220.0,  340.0),    # K
    'temp_min':          (210.0,  330.0),    # K
    'shortwave':         (0.0,    500.0),    # W/m², daily mean
    'longwave':          (100.0,  550.0),    # W/m²
    'pressure':          (50000., 110000.),  # Pa
    'specific_humidity': (0.0,    0.040),    # kg/kg
    'u_wind':            (-50.0,  50.0),     # m/s
    'v_wind':            (-50.0,  50.0),     # m/s
    'wind_10m':          (0.0,    50.0),     # m/s
    'wind_2m':           (0.0,    40.0),     # m/s
}

# FAO-56 eq. 47 with z = 10 m: u2 = u10 * 4.87 / ln(67.8*10 - 5.42) ≈ 0.748 * u10
WIND_10M_TO_2M = 4.87 / np.log(67.8 * 10 - 5.42)


def precip_factor_to_mm_day(units: str) -> float:
    """
    Multiplicative factor converting a precipitation field to mm/day.

      'kg m-2 s-1', 'kg/m2/s', 'mm s-1', 'mm/s'      -> 86400  (rate per second)
      'mm/hr', 'mm h-1', 'kg m-2 h-1'                -> 24     (rate per hour)
      'kg m-2', 'kg/m^2', 'mm', 'kg/m^2/day', 'mm/d' -> 1      (daily accumulation)

    Unknown strings raise ValueError so a units problem stops the run instead of
    silently producing numbers 86400x off.
    """
    u = (units or '').strip().lower().replace('**', '^')
    u = re.sub(r'\s+', ' ', u)
    per_second = re.search(r'(s-1|s\^-1|/s\b|/sec|per second)', u)
    per_hour   = re.search(r'(h-1|hr-1|h\^-1|/h\b|/hr\b|/hour|per hour)', u)
    per_day    = re.search(r'(d-1|day-1|/d\b|/day|per day)', u)
    if per_second:
        return 86400.0
    if per_hour:
        return 24.0
    if per_day:
        return 1.0
    if re.fullmatch(r'(kg ?(m-2|m\^-2|/m2|/m\^2|m\^\{-2\})|mm)', u):
        return 1.0   # accumulation over the file's day
    raise ValueError(f'Unrecognised precipitation units: {units!r}')


def clean(arr: np.ndarray, var: str) -> np.ndarray:
    """Set non-finite and out-of-bounds values to NaN (per pixel)."""
    lo, hi = PHYS_BOUNDS.get(var, (-np.inf, np.inf))
    a = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(a) & (a >= lo) & (a <= hi), a, np.nan)


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-pixel wind speed. Average speeds, never speeds of averaged vectors."""
    return np.sqrt(np.asarray(u, float) ** 2 + np.asarray(v, float) ** 2)
