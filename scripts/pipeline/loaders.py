"""
LEGACY MODULE — kept so old notebooks keep importing. ``load_period`` now delegates
to pipeline.gridpass (common footprint, strict daily pairing, per-pixel wind); the
per-product domain means it used to return mixed footprints (NLDAS-3 Gulf/ocean
cells vs AORC land-only) and are not used anywhere in the analysis pipeline.
loaders.py — Data loading for all 8 forcing variables
======================================================
Reads AORC and NLDAS processed NetCDF files from HYDRO_DATA_DIR,
applies unit conversions, computes wind speed, and returns clean DataFrames.

All time series are daily spatial-means over the given domain.
"""

import os, glob, re, warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')

# ── Variable mappings ──────────────────────────────────────────────────────────
AORC_VARS = {
    'precip':             ('precip',             'sum'),   # mm/day already
    'temp':               ('temp',               'mean'),  # K
    'shortwave':          ('shortwave',          'mean'),  # W/m²
    'longwave':           ('longwave',           'mean'),  # W/m²
    'pressure':           ('pressure',           'mean'),  # Pa
    'specific_humidity':  ('specific_humidity',  'mean'),  # kg/kg
    'u_wind':             ('u_wind',             'mean'),  # m/s @ 10m
    'v_wind':             ('v_wind',             'mean'),  # m/s @ 10m
}

NLDAS_VARS = {
    'precip':             ('Rainf',   'sum'),    # kg/m²/s → ×86400 for mm/day
    'temp':               ('Tair',    'mean'),   # K
    'shortwave':          ('SWdown',  'mean'),   # W/m²
    'longwave':           ('LWdown',  'mean'),   # W/m²
    'pressure':           ('PSurf',   'mean'),   # Pa
    'specific_humidity':  ('Qair',    'mean'),   # kg/kg
    'u_wind':             ('Wind_E',  'mean'),   # m/s @ 10m
    'v_wind':             ('Wind_N',  'mean'),   # m/s @ 10m
}


def _date_from_aorc(fp: str) -> pd.Timestamp | None:
    m = re.search(r'aorc_(\d{8})_south_states', os.path.basename(fp))
    return pd.to_datetime(m.group(1), format='%Y%m%d') if m else None


def _date_from_nldas(fp: str) -> pd.Timestamp | None:
    m = re.search(r'(\d{8})_southern', os.path.basename(fp))
    return pd.to_datetime(m.group(1), format='%Y%m%d') if m else None


def _wind_speed(u: float, v: float) -> float:
    """Resultant wind speed at 10 m."""
    return float(np.sqrt(u**2 + v**2))


def _wind_2m(ws10: float) -> float:
    """FAO-56 log-law correction: wind at 10 m → 2 m."""
    return float(ws10 * (4.87 / np.log(67.8 * 10 - 5.42)))


def _load_one_nc(fp: str, var_map: dict, is_nldas: bool) -> dict:
    """
    Open one NetCDF file, extract spatial means for available variables,
    apply unit conversions, return dict.

    Robustness
    ----------
    * Handles (lat,lon), (1,lat,lon), (time,lat,lon) arrays correctly.
      Precip: nansum over the time axis (daily accumulated total).
      All others: nanmean over the time axis (average state).
    * Uses np.nanmean over the spatial grid so NaN masked pixels
      (ocean, out-of-domain) are not included in the area mean — exactly
      what load_spatial_grid does in phase 7.
    * Unit conversion applied per variable, even for AORC, based on the
      'units' attribute in the file (not just the is_nldas flag).
    """
    row = {}
    try:
        with xr.open_dataset(fp) as ds:
            for std_name, (raw_name, agg) in var_map.items():
                if raw_name not in ds:
                    continue

                da  = ds[raw_name]
                arr = da.values.astype(float)
                units = str(da.attrs.get('units', ''))

                # Collapse time dimension if present
                if arr.ndim == 3:
                    if std_name == 'precip' and agg == 'sum':
                        arr_2d = np.nansum(arr, axis=0)   # accumulate time steps
                    else:
                        arr_2d = np.nanmean(arr, axis=0)  # average time steps
                else:
                    arr_2d = arr.squeeze()
                # Reject known corrupted geometries from bad clipping
                if arr_2d.shape == (1071, 3161):
                    return {}

                # Spatial nanmean — NaN masked region not included
                val = float(np.nanmean(arr_2d))
                if not np.isfinite(val):
                    continue

                # Unit conversions
                if std_name == 'precip':
                    u = units.lower()
                    if '/s' in u or ('kg' in u and 's' in u):
                        val *= 86400.0   # kg/m²/s → mm/day
                    elif 'mm/hr' in u or 'mm hr' in u:
                        val *= 24.0     # mm/hr → mm/day

                # Physical plausibility gate — reject fill-value leakage
                lo, hi = _PHYS_BOUNDS.get(std_name, (-1e15, 1e15))
                if not (lo <= val <= hi):
                    continue   # treated as missing day for this variable

                row[std_name] = val
    except Exception:
        pass   # silence individual file errors; caller handles missing days

    # Compute derived wind speed at 2 m
    if 'u_wind' in row and 'v_wind' in row:
        ws10 = _wind_speed(row['u_wind'], row['v_wind'])
        row['wind_10m'] = ws10
        row['wind_2m']  = _wind_2m(ws10)

    return row


# Per-variable physical plausibility bounds (domain-mean daily values).
# Values outside these ranges are treated as fill/corrupt and set to NaN.
_PHYS_BOUNDS: dict[str, tuple[float, float]] = {
    'precip':            (0.0,    200.0),   # mm/day   (domain-mean; extremes are ~50)
    'temp':              (240.0,  330.0),   # K        (~-33 to +57 C)
    'shortwave':         (0.0,    500.0),   # W/m²
    'longwave':          (150.0,  550.0),   # W/m²
    'pressure':          (80000., 110000.), # Pa       (sea-level ±30%)
    'specific_humidity': (0.0,    0.040),   # kg/kg
    'u_wind':            (-30.0,  30.0),    # m/s
    'v_wind':            (-30.0,  30.0),    # m/s
    'wind_10m':          (0.0,    30.0),    # m/s
    'wind_2m':           (0.0,    25.0),    # m/s
}



def load_period(
    aorc_dir: str,
    nldas_dir: str,
    start: str,
    end: str,
    aorc_pattern: str = 'aorc_*_south_states.nc',
    nldas_pattern: str = '*_southern.nc',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Daily domain-mean series for both products ON THE COMMON FOOTPRINT
    (delegates to pipeline.gridpass; columns <var>_aorc / <var>_nldas as before).
    """
    import warnings
    from pipeline import gridpass as GP
    warnings.warn('loaders.load_period is legacy; use pipeline.gridpass.run_gridpass',
                  DeprecationWarning, stacklevel=2)
    pairs, _, _ = GP.discover_pairs(aorc_dir, nldas_dir, start, end, aorc_pattern, nldas_pattern)
    if not pairs:
        return pd.DataFrame()
    dates = sorted(pairs)
    sample = [pairs[dates[i]] for i in np.linspace(0, len(dates) - 1, min(8, len(dates))).astype(int)]
    grid = GP.CommonGrid(sample, domains={'southern': None})
    res = GP.run_gridpass(pairs, grid, map_vars=[], log=print if verbose else (lambda *_: None))
    return res['domain_dfs']['southern']


def load_spatial_grid(fp: str, var: str, is_nldas: bool = False) -> tuple:
    """
    Full grid (lats ascending, lons, data_2d) for one variable from one file, in
    ANALYSIS units (precip mm/day from the units attribute; physical bounds applied).
    Returns (None, None, None) if the variable is absent.
    """
    from pipeline.gridpass import read_product_day, AORC_RAW, NLDAS_RAW
    f = read_product_day(fp, NLDAS_RAW if is_nldas else AORC_RAW, [var])
    if var not in f:
        return None, None, None
    return f['_lat'], f['_lon'], f[var]
