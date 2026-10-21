"""
pipeline/ghcn.py -- GHCN-Daily helper utilities
=================================================
Performance-critical design: all grid extraction is FILE-CENTRIC, not
station-centric. Opening each NetCDF file once and evaluating ALL station
points via vectorized interpolation reduces file I/O by N_stations times.

Old (wrong): for each station -> open all 365 files  = 365 * N_stations opens
New (correct): for each file  -> extract all stations = 365 opens total
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

SOUTHERN_STATES = {
    'AL', 'AR', 'FL', 'GA', 'KY', 'LA',
    'MS', 'MO', 'NC', 'OK', 'SC', 'TN', 'TX',
}

# ══════════════════════════════════════════════════════════════════════════════
# Inventory helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_inventory(ghcn_dir: str) -> pd.DataFrame:
    inv_path = os.path.join(ghcn_dir, 'inventory.csv')
    if not os.path.exists(inv_path):
        raise FileNotFoundError(
            f'GHCN inventory not found at {inv_path}.\n'
            'Run scripts/retrieval/get_ghcn_retrieval.py first.'
        )
    return pd.read_csv(inv_path, dtype={'station_id': str, 'state': str})


def filter_inventory(
    inv: pd.DataFrame,
    state: str | None = None,
    min_years: int = 10,
) -> pd.DataFrame:
    if state is not None:
        inv = inv[inv['state'].str.upper() == state.upper()]
    if 'n_years' in inv.columns:
        inv = inv[inv['n_years'] >= min_years]
    return inv.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Per-station GHCN monthly data (reads local parquet)
# ══════════════════════════════════════════════════════════════════════════════
def load_station_monthly(
    station_id: str,
    ghcn_dir: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Load monthly-aggregated observations for one station from local parquet.
    Returns DataFrame with columns: prcp_mm, tmax_c, tmin_c, tmean_c, wind_ms
    """
    pq = os.path.join(ghcn_dir, 'daily', f'{station_id}.parquet')
    if not os.path.exists(pq):
        return pd.DataFrame()

    df = pd.read_parquet(pq)
    df.index = pd.to_datetime(df.index)
    df = df.loc[start:end]
    if df.empty:
        return pd.DataFrame()

    cols = {}
    if 'PRCP' in df.columns:
        cols['prcp_mm'] = df['PRCP'].resample('ME').sum(min_count=15)
    if 'TMAX' in df.columns:
        cols['tmax_c']  = df['TMAX'].resample('ME').mean()
    if 'TMIN' in df.columns:
        cols['tmin_c']  = df['TMIN'].resample('ME').mean()
    if 'tmax_c' in cols and 'tmin_c' in cols:
        cols['tmean_c'] = (cols['tmax_c'] + cols['tmin_c']) / 2
    if 'AWND' in df.columns:
        cols['wind_ms'] = df['AWND'].resample('ME').mean()

    return pd.DataFrame(cols).dropna(how='all') if cols else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# BATCH grid extraction — FILE-CENTRIC (the correct, fast approach)
# ══════════════════════════════════════════════════════════════════════════════
def build_gridded_timeseries(
    file_list: list[tuple],          # (date, filepath) from EXT.discover_*
    var_name: str,                   # standardized name: 'precip', 'temp', ...
    is_nldas: bool,
    station_ids: list[str],
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Open each file ONCE and extract values at ALL station locations.

    Returns DataFrame: index=daily dates, columns=station_ids
    Values are in analysis units: precip=mm/day, temp=degC, etc.

    This is the key performance fix:
        Old: N_stations × N_files opens  (363,540 for LA × 1yr)
        New: N_files opens only           (730 for 1yr AORC+NLDAS)
    """
    import xarray as xr
    from pipeline.loaders import AORC_VARS, NLDAS_VARS

    var_map  = NLDAS_VARS if is_nldas else AORC_VARS
    if var_name not in var_map:
        return pd.DataFrame()

    raw_name, _ = var_map[var_name]
    t_start, t_end = pd.to_datetime(start), pd.to_datetime(end)

    # Station points array for vectorized interpolation
    station_pts = np.column_stack([station_lats, station_lons])  # (N, 2)

    rows = {}  # date -> np.array of length N_stations

    for date, fp in file_list:
        if not (t_start <= date <= t_end):
            continue
        try:
            with xr.open_dataset(fp) as ds:
                if raw_name not in ds:
                    continue

                lat_dim = next((c for c in ds.coords
                                if c.lower() in ('lat', 'latitude', 'y')), None)
                lon_dim = next((c for c in ds.coords
                                if c.lower() in ('lon', 'longitude', 'x')), None)
                if lat_dim is None or lon_dim is None:
                    continue

                lats = ds[lat_dim].values.astype(float)
                lons = ds[lon_dim].values.astype(float)
                arr  = ds[raw_name].values.squeeze()
                if arr.ndim == 3:
                    arr = np.nanmean(arr, axis=0)
                arr = arr.astype(float)

                # Ensure ascending lats for interpolator
                if lats[0] > lats[-1]:
                    lats = lats[::-1]
                    arr  = arr[::-1, :]

                # Unit conversions
                if var_name == 'precip':      # both products, from the units attribute
                    from pipeline.units import precip_factor_to_mm_day
                    arr = arr * precip_factor_to_mm_day(str(ds[raw_name].attrs.get('units', '')))
                if var_name == 'temp':
                    if np.nanmean(arr) > 200:  # Kelvin to Celsius
                        arr = arr - 273.15

                # One interpolator per file, evaluate at ALL stations at once
                interp = RegularGridInterpolator(
                    (lats, lons), arr,
                    method='linear', bounds_error=False, fill_value=np.nan,
                )
                vals = interp(station_pts)   # shape: (N_stations,)
                rows[date] = vals

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    # Build wide DataFrame: rows=dates, cols=station_ids
    df = pd.DataFrame.from_dict(rows, orient='index', columns=station_ids)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# Validation metrics
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(obs: pd.Series, sim: pd.Series, label: str = 'model') -> dict:
    """Compute standard validation metrics on common non-NaN monthly dates."""
    both = pd.DataFrame({'obs': obs, 'sim': sim}).dropna()
    if len(both) < 6:
        return {'n': len(both), 'label': label}

    o, s = both['obs'].values, both['sim'].values
    n    = len(o)

    bias = float(np.mean(s - o))
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))
    mae  = float(np.mean(np.abs(s - o)))

    r  = float(np.corrcoef(o, s)[0, 1]) if np.std(o) > 1e-9 and np.std(s) > 1e-9 else np.nan
    r2 = r ** 2 if np.isfinite(r) else np.nan

    ss_res = np.sum((o - s) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    nse = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    alpha = (np.std(s) / np.std(o)) if np.std(o) > 1e-9 else np.nan
    beta  = (np.mean(s) / np.mean(o)) if abs(np.mean(o)) > 1e-9 else np.nan
    kge   = (float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))
             if all(np.isfinite(v) for v in [r, alpha, beta]) else np.nan)

    return dict(
        label=label, n=n,
        bias=round(bias, 4), rmse=round(rmse, 4), mae=round(mae, 4),
        r=round(r, 4) if np.isfinite(r) else np.nan,
        r2=round(r2, 4) if np.isfinite(r2) else np.nan,
        nse=round(nse, 4) if np.isfinite(nse) else np.nan,
        kge=round(kge, 4) if np.isfinite(kge) else np.nan,
        obs_mean=round(float(np.mean(o)), 4),
        sim_mean=round(float(np.mean(s)), 4),
        obs_std=round(float(np.std(o)), 4),
        sim_std=round(float(np.std(s)), 4),
    )
