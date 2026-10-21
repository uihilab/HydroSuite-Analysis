"""
AORC 20-Year Download + Southern States Clip Pipeline
=====================================================
Downloads 2001–2020 AORC V1.1 1-km data from the NOAA public S3 bucket
and saves both the Southern-US raw Zarr and the Southern-states-clipped NetCDF
under HYDRO_DATA_DIR.

Coverage: TX, LA, MS, AL, FL, GA, SC, AR, TN, OK, and surrounding states.

Usage
-----
    python download_aorc_20yr_external.py                  # full 2001–2020
    python download_aorc_20yr_external.py 2010 2015        # specific years
    python download_aorc_20yr_external.py 2010 2010 --skip-raw   # clip only

Outputs
-------
  RAW  : $HYDRO_DATA_DIR/aorc/raw/aorc_YYYYMMDD_south.zarr
  PROC : $HYDRO_DATA_DIR/aorc/processed/aorc_YYYYMMDD_south_states.nc
"""

import os
import sys
import argparse
import glob
import time
import traceback
from datetime import datetime, timedelta

import s3fs
import xarray as xr
import pandas as pd
import numpy as np

# ── Output paths ────────────────────────────────────────────────────────
EXT_BASE      = os.path.join(os.environ.get('HYDRO_DATA_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))), 'aorc')
RAW_DIR       = os.path.join(EXT_BASE, 'raw')
PROCESSED_DIR = os.path.join(EXT_BASE, 'processed')

# ── Spatial domains ─────────────────────────────────────────────────────────────
# Southern US bounding box (matches existing raw files)
BBOX_SOUTH = dict(lon_min=-109.0, lon_max=-79.0, lat_min=25.0, lat_max=38.0)

# Southern US states bounding box
# Covers: TX, LA, MS, AL, FL, GA, SC, TN, AR, OK (+ Gulf coastal margin)
BBOX_SOUTH_STATES = dict(lon_min=-106.5, lon_max=-75.5, lat_min=24.5, lat_max=37.5)

# ── AORC S3 variable mapping ─────────────────────────────────────────────────
# key = AORC Zarr variable name, value = (output name, aggregation)
VARS_MAP = {
    'APCP_surface'         : ('precip',            'sum'),   # kg/m²/day
    'TMP_2maboveground'    : ('temp',              'mean'),  # K
    'SPFH_2maboveground'   : ('specific_humidity', 'mean'),  # kg/kg
    'DLWRF_surface'        : ('longwave',          'mean'),  # W/m²
    'DSWRF_surface'        : ('shortwave',         'mean'),  # W/m²
    'PRES_surface'         : ('pressure',          'mean'),  # Pa
    'UGRD_10maboveground'  : ('u_wind',            'mean'),  # m/s
    'VGRD_10maboveground'  : ('v_wind',            'mean'),  # m/s
}

# Daily extremes of hourly 2-m temperature, needed for Hargreaves-Samani and for the
# FAO-56 saturation vapour pressure (es = mean of es(Tmax), es(Tmin)). Days are
# 00–23 UTC; check that NLDAS-3 daily files use the same window.
EXTRA_AGGS = [('TMP_2maboveground', 'temp_max', 'max'),   # K
              ('TMP_2maboveground', 'temp_min', 'min')]   # K


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 – Download one day from S3 → raw Zarr
# ═══════════════════════════════════════════════════════════════════════════════
def download_day(ds_south_year, day: datetime, skip_existing=True) -> str | None:
    """
    Slice one calendar day from a pre-opened Southern-US annual xr.Dataset,
    aggregate hourly → daily, and save to a Zarr store.

    Returns the output path if successfully written, else None.
    """
    day_str  = day.strftime('%Y%m%d')
    outpath  = os.path.join(RAW_DIR, f'aorc_{day_str}_south.zarr')

    if skip_existing and os.path.exists(outpath):
        print(f'  [RAW SKIP] {day_str} (already exists)')
        return outpath

    print(f'  [RAW] Downloading {day_str} ...', flush=True)
    t0 = time.time()

    try:
        t_start = day
        t_end   = day + timedelta(hours=23)
        ds_day  = ds_south_year.sel(time=slice(t_start, t_end))

        if ds_day.time.size == 0:
            print(f'  [RAW WARN] No hourly data found for {day_str}')
            return None

        # Load coordinates into memory
        lats = ds_day.latitude.values
        lons = ds_day.longitude.values

        ds_out = xr.Dataset(coords={
            'time'     : [pd.to_datetime(day_str, format='%Y%m%d')],
            'latitude' : lats,
            'longitude': lons,
        })

        for zarr_var, (out_name, method) in VARS_MAP.items():
            if zarr_var not in ds_day:
                print(f'    [WARN] {zarr_var} not in dataset, skipping')
                continue
            da = ds_day[zarr_var].load()
            agg = da.sum(dim='time', keep_attrs=True) if method == 'sum' \
                else da.mean(dim='time', keep_attrs=True)
            agg.attrs = da.attrs
            if method == 'sum' and 'precip' in out_name:
                agg.attrs['units'] = 'kg/m^2/day'
            ds_out[out_name] = (('time', 'latitude', 'longitude'), [agg.values])
            ds_out[out_name].attrs = agg.attrs

        for zarr_var, out_name, method in EXTRA_AGGS:
            if zarr_var not in ds_day:
                continue
            da = ds_day[zarr_var].load()
            agg = getattr(da, method)(dim='time', keep_attrs=True)
            ds_out[out_name] = (('time', 'latitude', 'longitude'), [agg.values])
            ds_out[out_name].attrs = dict(da.attrs, cell_methods=f'time: {method} (hourly, 00-23 UTC)')

        ds_out.to_zarr(outpath, mode='w')
        elapsed = time.time() - t0
        print(f'  [RAW OK]  {day_str} -> {outpath}  ({elapsed:.1f}s)')
        return outpath

    except Exception:
        print(f'  [RAW ERR] {day_str}')
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 – Clip raw Zarr → Louisiana NetCDF
# ═══════════════════════════════════════════════════════════════════════════════
def clip_day(zarr_path: str, skip_existing=True) -> str | None:
    """
    Read a daily Southern-US Zarr, clip to Southern states bounding box,
    and save as a compressed NetCDF.

    Returns output path if successful.
    """
    day_str = os.path.basename(zarr_path)        # aorc_YYYYMMDD_south.zarr
    day_str = day_str.split('_')[1]              # YYYYMMDD
    outpath = os.path.join(PROCESSED_DIR, f'aorc_{day_str}_south_states.nc')

    if skip_existing and os.path.exists(outpath):
        print(f'  [PROC SKIP] {day_str} (already exists)')
        return outpath

    print(f'  [PROC] Clipping {day_str} to Southern states...', flush=True)
    t0 = time.time()

    try:
        ds = xr.open_zarr(zarr_path)
        ds_la = ds.sel(
            latitude =slice(BBOX_SOUTH_STATES['lat_min'], BBOX_SOUTH_STATES['lat_max']),
            longitude=slice(BBOX_SOUTH_STATES['lon_min'], BBOX_SOUTH_STATES['lon_max']),
        )
        # Compressed NetCDF4
        encoding = {v: {'zlib': True, 'complevel': 4} for v in ds_la.data_vars}
        ds_la.to_netcdf(outpath, encoding=encoding)
        elapsed = time.time() - t0
        print(f'  [PROC OK]  {day_str} -> {outpath}  ({elapsed:.1f}s)')
        return outpath

    except Exception:
        print(f'  [PROC ERR] {day_str}')
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════
def run(year_start: int, year_end: int,
        skip_raw: bool = False,
        skip_clip: bool = False):

    os.makedirs(RAW_DIR,       exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  AORC 1-km Pipeline  |  {year_start}–{year_end}')
    print(f'  RAW  -> {RAW_DIR}')
    print(f'  PROC -> {PROCESSED_DIR}')
    print(f'{"="*60}\n')

    fs = s3fs.S3FileSystem(anon=True)

    for year in range(year_start, year_end + 1):
        print(f'\n{"─"*50}')
        print(f'  YEAR {year}')
        print(f'{"─"*50}')

        # ── Connect to S3 annual Zarr (lazy, no data downloaded yet) ──────────
        if not skip_raw:
            url = f's3://noaa-nws-aorc-v1-1-1km/{year}.zarr'
            print(f'  Connecting to {url} ...')
            try:
                ds_year = xr.open_dataset(url, engine='zarr', backend_kwargs={'storage_options': {'anon': True}})
                # Slice to Southern US once to limit coordinate loading
                ds_south = ds_year.sel(
                    latitude =slice(BBOX_SOUTH['lat_min'], BBOX_SOUTH['lat_max']),
                    longitude=slice(BBOX_SOUTH['lon_min'], BBOX_SOUTH['lon_max']),
                )
                print(f'  Connected. Dims: {dict(ds_south.dims)}')
            except Exception:
                print(f'  [ERR] Cannot connect to S3 for {year}:')
                traceback.print_exc()
                print('  Skipping raw download for this year.')
                ds_south = None
        else:
            ds_south = None

        # ── Iterate over days ──────────────────────────────────────────────────
        start = datetime(year, 1, 1)
        end   = datetime(year, 12, 31)
        days  = [start + timedelta(days=i) for i in range((end - start).days + 1)]

        raw_ok = clip_ok = raw_skip = clip_skip = raw_err = clip_err = 0

        for day in days:
            zarr_path = os.path.join(RAW_DIR, f'aorc_{day.strftime("%Y%m%d")}_south.zarr')

            # Step 1 – Download raw
            if not skip_raw and ds_south is not None:
                result = download_day(ds_south, day)
                if result:
                    if 'SKIP' in repr(result):
                        raw_skip += 1
                    else:
                        raw_ok += 1
                else:
                    raw_err += 1
                raw_zarr = zarr_path  # path regardless of download result
            else:
                raw_zarr = zarr_path

            # Step 2 – Clip to Louisiana
            if not skip_clip and os.path.exists(raw_zarr):
                result = clip_day(raw_zarr)
                if result:
                    clip_ok += 1
                else:
                    clip_err += 1
            elif skip_clip:
                pass  # user asked to only download
            else:
                clip_err += 1
                print(f'  [PROC SKIP] Raw not found: {raw_zarr}')

        print(f'\n  Year {year} summary:')
        if not skip_raw:
            print(f'    RAW   : {raw_ok} downloaded | {raw_err} errors')
        if not skip_clip:
            print(f'    PROC  : {clip_ok} clipped    | {clip_err} errors')


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download AORC 1-km data (2001-2020) to HYDRO_DATA_DIR.')
    parser.add_argument('year_start', nargs='?', type=int, default=2001,
                        help='Start year (default: 2001)')
    parser.add_argument('year_end',   nargs='?', type=int, default=2020,
                        help='End year   (default: 2020)')
    parser.add_argument('--skip-raw',  action='store_true',
                        help='Skip S3 download; only run the clip step on existing Zarr files')
    parser.add_argument('--skip-clip', action='store_true',
                        help='Skip Louisiana clip; only run the S3 download step')
    args = parser.parse_args()

    if args.year_start > args.year_end:
        print('ERROR: year_start must be <= year_end'); sys.exit(1)

    t_total = time.time()
    run(args.year_start, args.year_end,
        skip_raw=args.skip_raw,
        skip_clip=args.skip_clip)
    elapsed = time.time() - t_total
    hours, rem = divmod(int(elapsed), 3600)
    mins,  secs = divmod(rem, 60)
    print(f'\nTotal wall time: {hours}h {mins}m {secs}s')
