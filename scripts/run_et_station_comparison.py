"""
run_et_station_comparison.py — ET₀ at GHCN-Daily stations (HS and PM)
=====================================================================
Replaces the removed run_et0_validation_pipeline.py (commit 11a7b35). Produces the
station tables, with these choices:

  1. Hargreaves-Samani now includes the 0.408 MJ m⁻² d⁻¹ → mm d⁻¹ conversion
     (pipeline.et_calc.hs_et0). The old code multiplied by Ra in MJ m⁻² d⁻¹ and
     therefore inflated every HS value — station and gridded — by 1/0.408 = 2.45×.
  2. Gridded HS uses the product's own daily Tmax/Tmin when the files carry them
     (or sub-daily steps to derive them). Otherwise Tmean_grid ± DTR/2 with the
     station's CALENDAR-MONTH median DTR (was one all-year median), and the
     approach is recorded per row in ``hs_tminmax_source``.
  3. PM-ET₀ is computed AT THE STATIONS from each product's forcing, so the
     HS-vs-PM "method gap" in et07 compares the same locations. The old et07 put
     station-located HS next to a domain-mean PM over the whole South.
  4. Monthly values use only days valid in BOTH the station and the product
     (≥ MIN_DAYS per month), so a station month with gaps is not compared with a
     complete gridded month. Monthly totals = mean daily rate × days in month.
  5. A station is used only if its nearest grid cell lies within 1.5 cells and is
     valid (coastal stations falling on water cells are dropped and counted).

Outputs  (results/et0_validation/<label>/tables/):
  et0_station_metrics.csv     per station × product: bias/rmse/r/nse/kge [mm/month]
  et0_skill_scores.csv        Southern US and Louisiana summaries
  et07_method_comparison.csv  12 monthly rows, mm/day (same columns as before + n_stations)
  station_selection.json      how many stations were kept / dropped and why

Usage:
  py -3.11 scripts/run_et_station_comparison.py --full
  py -3.11 scripts/run_et_station_comparison.py --full --rebuild-cache
"""

from __future__ import annotations
import os, sys, json, argparse, time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_config import EXT

# Log lines contain non-ASCII (ET₀, °C, −); keep a redirected Windows console from crashing.
for _stream in (sys.stdout, sys.stderr):
    try: _stream.reconfigure(errors='replace')
    except (AttributeError, ValueError): pass
from pipeline.ghcn import load_inventory, filter_inventory, compute_metrics
from pipeline.et_calc import hs_et0, pm_et0
from pipeline.units import wind_speed, WIND_10M_TO_2M
from pipeline import gridpass as GP

MIN_DAYS = 20            # paired days required for a station-month
MIN_MONTHS = 24          # paired months required for station metrics
STATION_VARS = ['temp', 'temp_max', 'temp_min', 'shortwave', 'longwave', 'pressure',
                'specific_humidity', 'u_wind', 'v_wind']


def log(msg, level='INFO'):
    print(f'[{datetime.now():%H:%M:%S}] [{level}] {msg}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Station observations
# ══════════════════════════════════════════════════════════════════════════════
def load_station_daily(inv: pd.DataFrame, start: str, end: str) -> dict[str, pd.DataFrame]:
    """{station_id: daily DataFrame[TMAX, TMIN] in °C, TMAX ≥ TMIN}."""
    out = {}
    for sid in inv['station_id']:
        pq = os.path.join(EXT.GHCN_DIR, 'daily', f'{sid}.parquet')
        if not os.path.exists(pq):
            continue
        df = pd.read_parquet(pq)
        df.index = pd.to_datetime(df.index)
        df = df.loc[start:end]
        if 'TMAX' not in df or 'TMIN' not in df:
            continue
        df = df[['TMAX', 'TMIN']].dropna()
        df = df[(df['TMAX'] >= df['TMIN']) & df['TMAX'].between(-40, 55) & df['TMIN'].between(-45, 45)]
        if len(df) >= 365:
            out[sid] = df
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Gridded values at station pixels (one file-centric pass per product)
# ══════════════════════════════════════════════════════════════════════════════
def extract_at_stations(files: list[tuple], raw_map: dict, lats, lons, sids,
                        progress_every: int = 730) -> dict[str, pd.DataFrame]:
    """
    Nearest-cell daily series for every station: {var: DataFrame(dates × stations)}.
    Stations farther than 1.5 cells from any cell centre are returned as all-NaN.
    """
    if not files:
        return {}
    # nearest cell per station, recomputed for each distinct file grid (some processed
    # files were clipped to a different extent)
    cache = {}

    def station_index(glat, glon):
        key = (len(glat), round(float(glat[0]), 5), round(float(glat[-1]), 5),
               len(glon), round(float(glon[0]), 5), round(float(glon[-1]), 5))
        if key not in cache:
            dlat, dlon = abs(np.median(np.diff(glat))), abs(np.median(np.diff(glon)))
            ii = GP._nearest_index(glat, np.asarray(lats, float), 1.5 * dlat)
            jj = GP._nearest_index(glon, np.asarray(lons, float), 1.5 * dlon)
            cache[key] = (ii, jj, (ii >= 0) & (jj >= 0))
        return cache[key]
    rows = {v: [] for v in STATION_VARS}
    dates = []
    t0 = time.time()
    bad = []
    for k, (d, fp) in enumerate(files):
        try:
            f = GP.read_product_day(fp, raw_map, STATION_VARS)
        except Exception as e:                   # fresh re-reads, then rebuild, else NaN
            f = None
            for attempt in range(1, 4):
                time.sleep(attempt)
                try:
                    f = GP.read_product_day(fp, raw_map, STATION_VARS, fresh=True); break
                except Exception:
                    pass
            if f is None:
                from pipeline.repair import verify, rebuild_aorc, rebuild_nldas
                fix = rebuild_aorc if raw_map is GP.AORC_RAW else rebuild_nldas
                try:
                    ok = bool(verify(fp)) and fix(fp, EXT.DATA_ROOT)
                    f = GP.read_product_day(fp, raw_map, STATION_VARS, fresh=True) if ok else None
                except Exception:
                    f = None
            if f is None:
                log(f'    {pd.Timestamp(d).date()} unreadable: {e}', 'WARN'); bad.append(str(e)); f = {}
            else:
                log(f'    {pd.Timestamp(d).date()} rebuilt from raw')
        if f:
            ii, jj, inside = station_index(f['_lat'], f['_lon'])
        for v in STATION_VARS:
            vals = np.full(len(sids), np.nan, np.float32)
            if v in f:
                vals[inside] = f[v][ii[inside], jj[inside]]
            rows[v].append(vals)
        dates.append(pd.Timestamp(d))
        if progress_every and (k + 1) % progress_every == 0:
            log(f'    {k + 1}/{len(files)} files ({time.time() - t0:.0f}s)')
    idx = pd.DatetimeIndex(dates)
    if len(cache) > 1:
        log(f'    {len(cache)} distinct file grids encountered (stations mapped per grid)', 'WARN')
    if bad:
        log(f'    {len(bad)} unreadable file(s) treated as missing days', 'WARN')
    return {v: pd.DataFrame(np.vstack(rows[v]), index=idx, columns=sids)
            for v in STATION_VARS if np.isfinite(np.vstack(rows[v])).any()}


def _cache_path(label: str, product: str) -> str:
    d = os.path.join(EXT.results_dir(f'et0_validation/{label}'), 'cache')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'station_extract_{product.lower()}.parquet')


def load_or_extract(product: str, files, raw_map, lats, lons, sids, label, rebuild):
    path = _cache_path(label, product)
    if os.path.exists(path) and not rebuild:
        long = pd.read_parquet(path)
        log(f'  {product}: station extract loaded from cache')
        return {v: g.drop(columns='var') for v, g in long.groupby('var')}
    log(f'  extracting {product} at {len(sids)} stations from {len(files)} files…')
    ext = extract_at_stations(files, raw_map, lats, lons, sids)
    pd.concat([df.assign(var=v) for v, df in ext.items()]).to_parquet(path)
    return ext


# ══════════════════════════════════════════════════════════════════════════════
# ET₀ series per station
# ══════════════════════════════════════════════════════════════════════════════
def _monthly_dtr(obs: pd.DataFrame) -> pd.Series:
    """Median diurnal range per calendar month (index 1..12), °C."""
    dtr = (obs['TMAX'] - obs['TMIN']).clip(lower=0)
    m = dtr.groupby(dtr.index.month).median()
    return m.reindex(range(1, 13)).fillna(float(dtr.median()))


def gridded_station_et0(ext: dict, sid: str, lat: float, dtr_month: pd.Series):
    """Daily gridded HS and PM ET₀ [mm/day] at one station, plus the Tmax/Tmin source."""
    T = ext['temp'][sid].astype(float) - 273.15
    idx = T.index
    doy = idx.dayofyear.values
    have_mm = ('temp_max' in ext and 'temp_min' in ext
               and ext['temp_max'][sid].notna().mean() > 0.9)
    if have_mm:
        tx = ext['temp_max'][sid].astype(float) - 273.15
        tn = ext['temp_min'][sid].astype(float) - 273.15
        src = 'gridded Tmax/Tmin'
    else:
        half = dtr_month.reindex(idx.month).values / 2.0
        tx, tn = T + half, T - half
        src = 'gridded Tmean ± station monthly DTR/2'
    hs = pd.Series(hs_et0(((tx + tn) / 2).values, tx.values, tn.values, lat, doy), index=idx)

    u2 = wind_speed(ext['u_wind'][sid].values, ext['v_wind'][sid].values) * WIND_10M_TO_2M
    pm = pm_et0(ext['temp'][sid].values.astype(float), ext['shortwave'][sid].values,
                ext['longwave'][sid].values, ext['pressure'][sid].values,
                ext['specific_humidity'][sid].values, u2, doy, lat,
                Tmax_K=(tx.values + 273.15) if have_mm else None,
                Tmin_K=(tn.values + 273.15) if have_mm else None)
    return hs, pd.Series(np.asarray(pm, float), index=idx), src


def monthly_paired(obs_daily: pd.Series, sim_daily: pd.Series) -> pd.DataFrame:
    """Monthly mean daily rate [mm/day] over days valid in both; months with < MIN_DAYS dropped."""
    both = pd.DataFrame({'obs': obs_daily, 'sim': sim_daily}).dropna()
    g = both.resample('ME')
    m = g.mean()[g.size() >= MIN_DAYS]
    return m.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run(start: str, end: str, label: str, state: str | None = None, rebuild: bool = False):
    out = f'et0_validation/{label}'
    inv = filter_inventory(load_inventory(EXT.GHCN_DIR), state=state, min_years=5)
    obs = load_station_daily(inv, start, end)
    inv = inv[inv['station_id'].isin(obs)].reset_index(drop=True)
    log(f'Stations with ≥1 yr of valid TMAX/TMIN: {len(inv)}')
    sids, lats, lons = inv['station_id'].tolist(), inv['lat'].values, inv['lon'].values

    # station HS (observed)
    obs_hs = {}
    for sid, lat in zip(sids, lats):
        o = obs[sid]
        obs_hs[sid] = pd.Series(hs_et0(((o['TMAX'] + o['TMIN']) / 2).values, o['TMAX'].values,
                                       o['TMIN'].values, float(lat), o.index.dayofyear.values),
                                index=o.index)

    pairs, _, _ = GP.discover_pairs(EXT.AORC_DIR, EXT.NLDAS_DIR, start, end,
                                    EXT.AORC_PATTERN, EXT.NLDAS_PATTERN)
    dates = sorted(pairs)
    product_files = {'AORC': ([(d, pairs[d][0]) for d in dates], GP.AORC_RAW),
                     'NLDAS': ([(d, pairs[d][1]) for d in dates], GP.NLDAS_RAW)}

    # Extract both products first: a station is used only if its cell is valid in
    # BOTH (NLDAS-3 carries values over Gulf/water cells where AORC is NaN), so the
    # AORC and NLDAS-3 statistics describe the same stations.
    exts = {}
    for product, (files, raw) in product_files.items():
        exts[product] = load_or_extract(product, files, raw, lats, lons, sids, label, rebuild)
        if not exts[product]:
            log(f'  {product}: no files', 'ERROR'); return

    def _valid(ext, sid):
        return sid in ext['temp'] and ext['temp'][sid].notna().mean() >= 0.5

    per_product = {p: {} for p in exts}          # product -> sid -> (row, hs, pm)
    n_invalid = {p: 0 for p in exts}
    for sid, lat, (_, srow) in zip(sids, lats, inv.iterrows()):
        dtr_m = _monthly_dtr(obs[sid])
        for product, ext in exts.items():
            if not _valid(ext, sid):
                n_invalid[product] += 1; continue
            hs, pm, src = gridded_station_et0(ext, sid, float(lat), dtr_m)
            mon = monthly_paired(obs_hs[sid], hs)
            if len(mon) < MIN_MONTHS:
                continue
            mon_mm = mon * mon.index.days_in_month.values[:, None]       # mm/month
            m = compute_metrics(mon_mm['obs'], mon_mm['sim'], label=product)
            row = dict(station_id=sid, lat=float(lat), lon=float(srow['lon']),
                       state=str(srow.get('state', '')), name=str(srow.get('name', '')),
                       elev=float(srow.get('elev', np.nan)), product=product,
                       dt_clim=round(float(dtr_m.mean()), 2), hs_tminmax_source=src, **m)
            per_product[product][sid] = (row, hs, pm)

    common_sids = sorted(set.intersection(*(set(d) for d in per_product.values())))
    metric_rows = [per_product[p][sid][0] for p in per_product for sid in common_sids]
    daily_store = {(p, k): {sid: per_product[p][sid][1 if k == 'hs' else 2] for sid in common_sids}
                   for p in per_product for k in ('hs', 'pm')}
    selection = dict(stations_with_ghcn_tmax_tmin=len(sids),
                     cell_invalid_or_off_grid={p: n_invalid[p] for p in exts},
                     usable_per_product={p: len(d) for p, d in per_product.items()},
                     used_common_to_both=len(common_sids),
                     min_days_per_month=MIN_DAYS, min_months=MIN_MONTHS)
    log(f'  station selection: {selection}')

    if not metric_rows:
        log('No station metrics produced', 'ERROR'); return
    met = pd.DataFrame(metric_rows)
    met.to_csv(EXT.tbl_path(out, 'et0_station_metrics.csv'), index=False)
    with open(EXT.tbl_path(out, 'station_selection.json'), 'w') as fh:
        json.dump(selection, fh, indent=1)

    # summary table (units: mm/month for bias/RMSE)
    rows = []
    for region, sf in [('Southern US', None), ('Louisiana', 'LA')]:
        d_r = met if sf is None else met[met['state'].str.upper() == sf]
        for product in ('AORC', 'NLDAS'):
            dp = d_r[d_r['product'] == product]
            if dp.empty: continue
            rows.append(dict(region=region, product=product, n_stations=len(dp),
                             bias_mean=round(dp['bias'].mean(), 2), bias_std=round(dp['bias'].std(), 2),
                             rmse_mean=round(dp['rmse'].mean(), 2), r_mean=round(dp['r'].mean(), 3),
                             kge_mean=round(dp['kge'].mean(), 3), nse_mean=round(dp['nse'].mean(), 3),
                             obs_mean_mm_month=round(dp['obs_mean'].mean(), 1),
                             dt_clim_mean=round(dp['dt_clim'].mean(), 1),
                             hs_tminmax_source=dp['hs_tminmax_source'].mode().iat[0]))
    pd.DataFrame(rows).to_csv(EXT.tbl_path(out, 'et0_skill_scores.csv'), index=False)

    # et07: monthly climatology [mm/day] over stations valid in all series, same locations
    common = common_sids
    log(f'  et07 uses {len(common)} stations common to GHCN, AORC and NLDAS-3')

    def clim(series_by_sid):
        mons = []
        for sid in common:
            s = series_by_sid[sid]
            g = s.dropna().resample('ME')
            mons.append(g.mean()[g.size() >= MIN_DAYS])
        allm = pd.concat(mons, axis=1).mean(axis=1)
        return allm.groupby(allm.index.month).mean()

    ser = {'ghcn_hs': clim(obs_hs),
           'aorc_hs': clim(daily_store[('AORC', 'hs')]), 'nldas_hs': clim(daily_store[('NLDAS', 'hs')]),
           'aorc_pm': clim(daily_store[('AORC', 'pm')]), 'nldas_pm': clim(daily_store[('NLDAS', 'pm')])}
    ml = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    t = pd.DataFrame({k: v.reindex(range(1, 13)).values for k, v in ser.items()})
    t.insert(0, 'month_label', ml); t.insert(0, 'month', range(1, 13))
    t['aorc_method_gap'] = t['aorc_pm'] - t['aorc_hs']
    t['nldas_method_gap'] = t['nldas_pm'] - t['nldas_hs']
    t['forcing_gap_hs'] = t['aorc_hs'] - t['nldas_hs']
    t['forcing_gap_pm'] = t['aorc_pm'] - t['nldas_pm']
    t['n_stations'] = len(common)
    t.round(4).to_csv(EXT.tbl_path(f'comparisons/{label}', 'et07_method_comparison.csv'), index=False)
    log('Done. Regenerate figures with scripts/regen_et0_figures.py and regen_et07_figures.py ')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--full', action='store_true', help='2001-01-01 .. 2020-12-31')
    ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--label')
    ap.add_argument('--state', default=None, help='restrict to one state (e.g. LA)')
    ap.add_argument('--rebuild-cache', action='store_true')
    a = ap.parse_args()
    if a.full or not (a.start and a.end):
        s, e, lbl = EXT.PERIOD_START, EXT.PERIOD_END, 'full_2001_2020'
    else:
        s, e, lbl = a.start, a.end, a.label or f'{a.start[:4]}_{a.end[:4]}'
    run(s, e, lbl + (f'_{a.state.lower()}' if a.state else ''), a.state, a.rebuild_cache)
