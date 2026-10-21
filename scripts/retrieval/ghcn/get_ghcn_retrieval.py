"""
get_ghcn_retrieval.py -- GHCN-Daily HTTPS Downloader
======================================================
Downloads GHCN-Daily station data for the 13 Southern US states (2001-2020)
from NOAA's CDO HTTPS access endpoint. No API key required.

Notes
-----
  - HTTPS access (no FTP idle-disconnect timeout)
  - The state filter applies to all stations in the inventory
  - Per-station retry on transient HTTP errors

HTTPS data endpoint
-------------------
  Inventory : https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
  Daily data: https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/<STATION_ID>.csv

CSV format (access endpoint, already pivoted wide, one row per day):
  STATION, DATE, LATITUDE, LONGITUDE, ELEVATION, NAME,
  PRCP, PRCP_ATTRIBUTES, TMAX, TMAX_ATTRIBUTES, TMIN, TMIN_ATTRIBUTES,
  AWND, AWND_ATTRIBUTES, ...
  Units: PRCP in tenths of mm; TMAX, TMIN in tenths of deg C; AWND in tenths of m/s

Output (under HYDRO_DATA_DIR)
-----------------------------
  $HYDRO_DATA_DIR/ghcn/
    inventory.csv              -- filtered stations
    daily/<station_id>.parquet -- daily data (already unit-converted)

Usage
-----
  python scripts/retrieval/get_ghcn_retrieval.py              # all Southern US
  python scripts/retrieval/get_ghcn_retrieval.py --dry-run    # inventory only
  python scripts/retrieval/get_ghcn_retrieval.py --state LA   # one state
  python scripts/retrieval/get_ghcn_retrieval.py --resume     # skip existing
"""

import os, sys, argparse, io, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request
import urllib.error

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
_EXT_ROOT = os.environ.get('HYDRO_DATA_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
GHCN_DIR  = os.path.join(_EXT_ROOT, 'ghcn')
DAILY_DIR = os.path.join(GHCN_DIR,  'daily')

# ── Domain config ─────────────────────────────────────────────────────────────
SOUTHERN_STATES = {
    'AL', 'AR', 'FL', 'GA', 'KY', 'LA',
    'MS', 'MO', 'NC', 'OK', 'SC', 'TN', 'TX',
}

# Broad bounding box (state filter is the primary constraint)
BBOX = dict(lat_min=24.5, lat_max=38.5, lon_min=-110.5, lon_max=-78.5)

ELEMENTS   = ['PRCP', 'TMAX', 'TMIN', 'AWND']
YEAR_START = 2001
YEAR_END   = 2020
MIN_YEARS  = 10

# ── NOAA HTTPS endpoints ──────────────────────────────────────────────────────
INVT_URL  = 'https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt'
DATA_BASE = 'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access'

HEADERS   = {'User-Agent': 'HydroSuiteAnalysis/1.0 (research project)'}
TIMEOUT   = 60    # seconds per request
MAX_RETRY = 3


def log(msg, level='INFO'):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] [{level}] {msg}', flush=True)


def _get(url: str, retries: int = MAX_RETRY) -> bytes | None:
    """HTTP GET with retry on transient errors. Returns raw bytes or None."""
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # station has no data file
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Inventory
# ══════════════════════════════════════════════════════════════════════════════
def download_inventory(state_filter: str | None = None) -> pd.DataFrame:
    """
    Download and parse GHCN station inventory.
    Filters to Southern US states (state code + bounding box).
    Pre-filters using ghcnd-inventory.txt to only include stations that
    have PRCP data overlapping with 2001-2020.
    """
    log('Downloading station inventory...')
    raw = _get(INVT_URL)
    if raw is None:
        raise RuntimeError(f'Could not download inventory from {INVT_URL}')

    rows = []
    for line in raw.decode('latin-1').splitlines():
        if len(line) < 50:
            continue
        try:
            sid   = line[0:11].strip()
            lat   = float(line[12:20].strip())
            lon   = float(line[21:30].strip())
            elev  = float(line[31:37].strip()) if line[31:37].strip() not in ('', '-999.9') else np.nan
            state = line[38:40].strip().upper()
            name  = line[41:71].strip()
        except (ValueError, IndexError):
            continue

        # KEY FIX: state check applies to ALL stations
        if state not in SOUTHERN_STATES:
            continue
        if not (BBOX['lat_min'] <= lat <= BBOX['lat_max']):
            continue
        if not (BBOX['lon_min'] <= lon <= BBOX['lon_max']):
            continue

        rows.append(dict(station_id=sid, lat=lat, lon=lon, elev=elev,
                         state=state, name=name))

    inv = pd.DataFrame(rows)
    log(f'  Candidate stations (all historical): {len(inv)}')

    # ── Pre-filter using element inventory ────────────────────────────────────
    # ghcnd-inventory.txt format: STATION LAT LON ELEMENT FIRSTYEAR LASTYEAR
    inv_elem_url = 'https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt'
    log('  Downloading element inventory for year-range pre-filter...')
    raw_elem = _get(inv_elem_url)
    if raw_elem:
        active_ids = set()
        for line in raw_elem.decode('latin-1').splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            sid_e, elem, first_yr, last_yr = parts[0], parts[3], int(parts[4]), int(parts[5])
            # Require at least MIN_YEARS of overlap with 2001-2020
            overlap = min(last_yr, YEAR_END) - max(first_yr, YEAR_START) + 1
            if elem in ('PRCP', 'TMAX', 'TMIN') and overlap >= MIN_YEARS:
                active_ids.add(sid_e)
        before = len(inv)
        inv = inv[inv['station_id'].isin(active_ids)].reset_index(drop=True)
        log(f'  After year-range pre-filter ({YEAR_START}-{YEAR_END}): {len(inv)} (was {before})')
    else:
        log('  Could not download element inventory -- skipping pre-filter', 'WARN')

    if state_filter:
        inv = inv[inv['state'] == state_filter.upper()]

    log(f'  Final inventory: {len(inv)} stations'
        + (f' ({state_filter.upper()})' if state_filter else ' (Southern US)'))
    return inv.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Daily data download & parsing (HTTPS CSV format)
# ══════════════════════════════════════════════════════════════════════════════
def _parse_access_csv(raw_bytes: bytes) -> pd.DataFrame:
    """
    Parse NOAA CDO 'access' format CSV.

    Format: wide CSV, one row per day. Key columns:
        DATE  -- YYYY-MM-DD
        PRCP  -- tenths of mm  (NaN if missing)
        TMAX  -- tenths of degC
        TMIN  -- tenths of degC
        AWND  -- tenths of m/s

    Returns DataFrame indexed by date, units converted to:
        PRCP mm, TMAX degC, TMIN degC, AWND m/s
    """
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    except Exception:
        return pd.DataFrame()

    # Normalise column names (CDO sometimes uses lowercase)
    df.columns = [c.strip().upper() for c in df.columns]

    if 'DATE' not in df.columns:
        return pd.DataFrame()

    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])
    df = df.set_index('DATE').sort_index()

    # Filter to years of interest
    df = df[(df.index.year >= YEAR_START) & (df.index.year <= YEAR_END)]
    if df.empty:
        return pd.DataFrame()

    # Keep only the columns we use
    out_cols = {}
    for elem in ELEMENTS:
        if elem in df.columns:
            col = pd.to_numeric(df[elem], errors='coerce')
            # Convert tenths to standard units
            out_cols[elem] = col / 10.0

    if not out_cols:
        return pd.DataFrame()

    result = pd.DataFrame(out_cols, index=df.index)
    result.index.name = 'date'
    return result


def download_station(
    station_id: str,
    out_dir: str,
    resume: bool = True,
) -> dict:
    """
    Download, parse, and cache daily data for one station.

    Returns metadata dict: {n_years, n_prcp, status}
    """
    pq_path = os.path.join(out_dir, f'{station_id}.parquet')

    if resume and os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
        return {
            'n_years': int(df.index.year.nunique()) if not df.empty else 0,
            'n_prcp' : int(df['PRCP'].notna().sum()) if 'PRCP' in df.columns else 0,
            'status' : 'cached',
        }

    url  = f'{DATA_BASE}/{station_id}.csv'
    raw  = _get(url)

    if raw is None:
        return {'n_years': 0, 'n_prcp': 0, 'status': 'not_found'}

    daily = _parse_access_csv(raw)
    if daily.empty:
        return {'n_years': 0, 'n_prcp': 0, 'status': 'empty'}

    daily.to_parquet(pq_path, compression='snappy')
    return {
        'n_years': int(daily.index.year.nunique()),
        'n_prcp' : int(daily['PRCP'].notna().sum()) if 'PRCP' in daily.columns else 0,
        'status' : 'downloaded',
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    os.makedirs(GHCN_DIR,  exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)

    log(f'GHCN-Daily retrieval (HTTPS) -- Southern US {YEAR_START}-{YEAR_END}')
    log(f'Output: {GHCN_DIR}')

    # ── Inventory ────────────────────────────────────────────────────────────
    inv = download_inventory(state_filter=args.state)
    inv_path = os.path.join(GHCN_DIR, 'inventory.csv')

    if args.dry_run:
        inv['n_years'] = 0; inv['n_prcp'] = 0
        inv.to_csv(inv_path, index=False)
        log(f'[DRY-RUN] Inventory saved: {inv_path}  ({len(inv)} stations)')
        log('[DRY-RUN] No daily files downloaded.')
        return

    # ── Daily download ───────────────────────────────────────────────────────
    log(f'Downloading daily data for {len(inv)} stations '
        f'(resume={args.resume}) via HTTPS...')
    t0 = time.time()

    meta_rows = []
    n_downloaded = n_cached = n_empty = 0

    for i, row in inv.iterrows():
        sid  = row['station_id']
        meta = download_station(sid, DAILY_DIR, resume=args.resume)
        meta_rows.append(meta)

        status = meta.get('status', '')
        if status == 'downloaded':  n_downloaded += 1
        elif status == 'cached':    n_cached += 1
        else:                       n_empty  += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            log(f'  {i+1}/{len(inv)} stations | {elapsed/60:.1f} min | '
                f'downloaded={n_downloaded} cached={n_cached} empty={n_empty} | '
                f'last: {sid} ({status})')

    # ── Finalize inventory ───────────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_rows, index=inv.index)
    inv = inv.join(meta_df)

    inv_full = inv[inv['n_years'] >= MIN_YEARS].copy()
    log(f'Stations with >= {MIN_YEARS} yr data: {len(inv_full)} / {len(inv)}')

    inv_full.to_csv(inv_path, index=False)
    # Also save full pre-filter inventory for reference
    inv.to_csv(os.path.join(GHCN_DIR, 'inventory_all.csv'), index=False)

    elapsed = time.time() - t0
    h, r = divmod(int(elapsed), 3600); m, s = divmod(r, 60)
    log(f'Done. Total time: {h}h {m}m {s}s')
    log(f'Stations in inventory.csv:    {len(inv_full)}')
    log(f'Downloaded: {n_downloaded}  Cached: {n_cached}  Empty/404: {n_empty}')
    log(f'Daily parquets: {DAILY_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GHCN-Daily HTTPS downloader for Southern US (2001-2020)')
    parser.add_argument('--dry-run',  action='store_true',
                        help='Download inventory only, no daily files')
    parser.add_argument('--state',    type=str, default=None,
                        help='Restrict to one state (e.g. LA, TX)')
    parser.add_argument('--resume',   action='store_true', default=True,
                        help='Skip stations already downloaded (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Re-download all stations even if parquet exists')
    args = parser.parse_args()
    main(args)
