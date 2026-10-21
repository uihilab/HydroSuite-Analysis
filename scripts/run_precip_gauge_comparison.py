"""
run_precip_gauge_comparison.py — Precipitation intermittency against GHCN-Daily gauges
=====================================================================================
Tests which gridded product reproduces the rain-day structure seen at gauges. The
gridded comparison found that NLDAS-3 rains on more days with lower intensity than
AORC; this script asks which of the two is closer to the gauges.

For every GHCN-Daily station with precipitation (COOP, CoCoRaHS, first-order,
other networks) whose grid cell is valid in both products:
  * one pass over the daily files reads only precipitation at the gauge cells
    (nearest cell, per distinct file grid), cached;
  * on days valid in the gauge AND both products (identical days for all three),
    wet-day fraction (≥ WET mm), wet-day mean intensity, P90 wet-day intensity,
    drizzle fraction (0 < P < WET), dry fraction (P = 0), mean / max wet and dry
    spell lengths (a missing day ends a spell), and mean annual total;
  * daily correlation at lag 0 and with the gauge shifted one day earlier, because
    many gauges (CoCoRaHS, most COOP) report a 24-h total ending in the morning of
    the observation date, while the gridded days run 00–24 UTC.

Interpretation notes: a gauge is a point and a grid cell
averages ~1 km², so cells are expected to be wet slightly more often than a single
gauge; the day-boundary mismatch affects daily correlation, not the frequency and
spell statistics. Results are reported by network so the first-order stations
(midnight-to-midnight days) can be checked separately.

Outputs (results/precip_gauges/<label>/):
  tables/gauge_station_metrics.csv   one row per station × source (GAUGE, AORC, NLDAS)
  tables/gauge_summary.csv           medians by network and for all stations
  tables/gauge_closeness.csv         share of stations where each product is closer
  tables/station_selection.json
  figures/gauge_intermittency.png

    py -3.11 scripts/run_precip_gauge_comparison.py --full
"""

from __future__ import annotations
import os, sys, json, argparse, time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_config import EXT
from pipeline import gridpass as GP

for _stream in (sys.stdout, sys.stderr):
    try: _stream.reconfigure(errors='replace')
    except (AttributeError, ValueError): pass

WET = EXT.WET_THRESHOLD_MM          # 1.0 mm, as in the gridded intermittency analysis
MIN_PAIRED_DAYS = 1000
NETWORKS = {'US1': 'CoCoRaHS', 'USC': 'COOP', 'USW': 'First-order (ASOS/WBAN)', 'USR': 'Other'}


def log(msg, level='INFO'):
    print(f'[{datetime.now():%H:%M:%S}] [{level}] {msg}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
def load_gauges(inv: pd.DataFrame, start: str, end: str) -> dict[str, pd.Series]:
    out = {}
    for sid in inv['station_id']:
        fp = os.path.join(EXT.GHCN_DIR, 'daily', f'{sid}.parquet')
        if not os.path.exists(fp):
            continue
        df = pd.read_parquet(fp)
        if 'PRCP' not in df:
            continue
        s = df['PRCP']; s.index = pd.to_datetime(s.index)
        s = s.loc[start:end]
        s = s[(s >= 0) & (s < 1000)]
        if s.notna().sum() >= MIN_PAIRED_DAYS:
            out[sid] = s.astype(float)
    return out


def extract_precip(files, raw_map, lats, lons, sids, progress_every=730) -> pd.DataFrame:
    """Daily precipitation at the nearest cell of each gauge (dates × stations)."""
    cache, rows, dates = {}, [], []
    t0 = time.time()
    for k, (d, fp) in enumerate(files):
        try:
            f = GP.read_product_day(fp, raw_map, ['precip'])
        except Exception:
            f = None
            for attempt in range(1, 4):
                time.sleep(attempt)
                try:
                    f = GP.read_product_day(fp, raw_map, ['precip'], fresh=True); break
                except Exception:
                    pass
        vals = np.full(len(sids), np.nan, np.float32)
        if f and 'precip' in f:
            glat, glon = f['_lat'], f['_lon']
            key = (len(glat), round(float(glat[0]), 5), round(float(glat[-1]), 5),
                   len(glon), round(float(glon[0]), 5), round(float(glon[-1]), 5))
            if key not in cache:
                ii = GP._nearest_index(glat, np.asarray(lats, float), 1.5 * abs(np.median(np.diff(glat))))
                jj = GP._nearest_index(glon, np.asarray(lons, float), 1.5 * abs(np.median(np.diff(glon))))
                cache[key] = (ii, jj, (ii >= 0) & (jj >= 0))
            ii, jj, inside = cache[key]
            vals[inside] = f['precip'][ii[inside], jj[inside]]
        rows.append(vals); dates.append(pd.Timestamp(d))
        if progress_every and (k + 1) % progress_every == 0:
            log(f'    {k + 1}/{len(files)} files ({time.time() - t0:.0f}s)')
    if len(cache) > 1:
        log(f'    {len(cache)} distinct file grids (gauges mapped per grid)')
    return pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(dates), columns=sids)


def load_or_extract(product, files, raw_map, lats, lons, sids, out_dir, rebuild):
    path = os.path.join(out_dir, 'cache', f'precip_extract_{product.lower()}.parquet')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not rebuild:
        df = pd.read_parquet(path)
        if set(sids) <= set(df.columns):
            log(f'  {product}: precipitation extract loaded from cache'); return df[sids]
    log(f'  extracting {product} precipitation at {len(sids)} gauges from {len(files)} files…')
    df = extract_precip(files, raw_map, lats, lons, sids)
    df.to_parquet(path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
def spell_lengths(wet: np.ndarray, valid: np.ndarray, target: bool) -> np.ndarray:
    """Lengths of runs where wet == target, broken by invalid days."""
    runs, c = [], 0
    for w, ok in zip(wet, valid):
        if ok and w == target:
            c += 1
        else:
            if c: runs.append(c)
            c = 0
    if c: runs.append(c)
    return np.asarray(runs)


def metrics(p: np.ndarray, valid: np.ndarray) -> dict:
    x = p[valid]
    wet = p >= WET
    w = x[x >= WET]
    ws, ds = spell_lengths(wet, valid, True), spell_lengths(wet, valid, False)
    return dict(wet_day_frac=float((x >= WET).mean()), dry_day_frac_zero=float((x == 0).mean()),
                drizzle_frac=float(((x > 0) & (x < WET)).mean()),
                wet_mean_mm=float(w.mean()) if w.size else np.nan,
                p90_wet_mm=float(np.percentile(w, 90)) if w.size >= 10 else np.nan,
                wet_spell_mean=float(ws.mean()) if ws.size else np.nan, wet_spell_max=int(ws.max()) if ws.size else 0,
                dry_spell_mean=float(ds.mean()) if ds.size else np.nan, dry_spell_max=int(ds.max()) if ds.size else 0,
                annual_total_mm=float(x.mean() * 365.25))


def daily_r(g, s):
    m = np.isfinite(g) & np.isfinite(s)
    return float(np.corrcoef(g[m], s[m])[0, 1]) if m.sum() > 100 and g[m].std() > 0 and s[m].std() > 0 else np.nan


# ══════════════════════════════════════════════════════════════════════════════
def run(start, end, label, rebuild=False):
    out = os.path.join(EXT.RESULTS_ROOT, 'precip_gauges', label)
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True); os.makedirs(os.path.join(out, 'figures'), exist_ok=True)

    inv = pd.read_csv(os.path.join(EXT.GHCN_DIR, 'inventory.csv'), dtype={'station_id': str})
    gauges = load_gauges(inv, start, end)
    inv = inv[inv['station_id'].isin(gauges)].reset_index(drop=True)
    log(f'Gauges with ≥{MIN_PAIRED_DAYS} valid precipitation days: {len(inv)}')
    sids, lats, lons = inv['station_id'].tolist(), inv['lat'].values, inv['lon'].values

    pairs, _, _ = GP.discover_pairs(EXT.AORC_DIR, EXT.NLDAS_DIR, start, end, EXT.AORC_PATTERN, EXT.NLDAS_PATTERN)
    dates = sorted(pairs)
    grid = {p: load_or_extract(p, [(d, pairs[d][k]) for d in dates], raw, lats, lons, sids, out, rebuild)
            for p, k, raw in (('AORC', 0, GP.AORC_RAW), ('NLDAS', 1, GP.NLDAS_RAW))}

    idx = pd.DatetimeIndex(dates)
    rows, sel = [], dict(gauges_with_data=len(sids), cell_invalid=0, too_few_paired_days=0, used=0)
    for sid, (_, srow) in zip(sids, inv.iterrows()):
        g = gauges[sid].reindex(idx).values
        a = grid['AORC'][sid].values.astype(float); n = grid['NLDAS'][sid].values.astype(float)
        if np.isfinite(a).mean() < 0.5 or np.isfinite(n).mean() < 0.5:
            sel['cell_invalid'] += 1; continue
        valid = np.isfinite(g) & np.isfinite(a) & np.isfinite(n)
        if valid.sum() < MIN_PAIRED_DAYS:
            sel['too_few_paired_days'] += 1; continue
        sel['used'] += 1
        g_prev = np.r_[g[1:], np.nan]        # gauge total reported on day t+1 covers mostly day t
        base = dict(station_id=sid, network=NETWORKS.get(sid[:3], 'Other'), state=srow['state'],
                    lat=float(srow['lat']), lon=float(srow['lon']), n_days=int(valid.sum()))
        for src, p in (('GAUGE', g), ('AORC', a), ('NLDAS', n)):
            rec = dict(base, source=src, **metrics(np.where(valid, p, np.nan), valid))
            if src != 'GAUGE':
                rec['daily_r_lag0'] = daily_r(np.where(valid, g, np.nan), np.where(valid, p, np.nan))
                rec['daily_r_gauge_next_day'] = daily_r(g_prev, np.where(valid, p, np.nan))
                gm = pd.Series(np.where(valid, g, np.nan), idx).resample('ME').mean()
                pm = pd.Series(np.where(valid, p, np.nan), idx).resample('ME').mean()
                rec['monthly_r'] = float(gm.corr(pm))
            rows.append(rec)
    log(f'  station selection: {sel}')
    m = pd.DataFrame(rows)
    m.round(4).to_csv(os.path.join(out, 'tables', 'gauge_station_metrics.csv'), index=False)
    with open(os.path.join(out, 'tables', 'station_selection.json'), 'w') as fh:
        json.dump(sel, fh, indent=1)

    keys = ['wet_day_frac', 'dry_day_frac_zero', 'drizzle_frac', 'wet_mean_mm', 'p90_wet_mm', 'wet_spell_mean',
            'wet_spell_max', 'dry_spell_mean', 'dry_spell_max', 'annual_total_mm', 'daily_r_lag0',
            'daily_r_gauge_next_day', 'monthly_r']
    summ = []
    for net, sub in [('All', m)] + list(m.groupby('network')):
        for src in ('GAUGE', 'AORC', 'NLDAS'):
            s = sub[sub['source'] == src]
            summ.append(dict(network=net, source=src, n_stations=s['station_id'].nunique(),
                             **{k: round(float(s[k].median()), 4) for k in keys if k in s and s[k].notna().any()}))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out, 'tables', 'gauge_summary.csv'), index=False)

    # which product is closer to the gauge, station by station
    w = m.pivot_table(index='station_id', columns='source', values=['wet_day_frac', 'wet_mean_mm', 'p90_wet_mm', 'dry_day_frac_zero', 'annual_total_mm'])
    net = m.drop_duplicates('station_id').set_index('station_id')['network']
    close = []
    for k in ['wet_day_frac', 'wet_mean_mm', 'p90_wet_mm', 'dry_day_frac_zero', 'annual_total_mm']:
        ea = (w[(k, 'AORC')] - w[(k, 'GAUGE')]).abs(); en = (w[(k, 'NLDAS')] - w[(k, 'GAUGE')]).abs()
        for nm, mask in [('All', slice(None))] + [(x, net[net == x].index) for x in sorted(net.unique())]:
            a_, n_ = ea.loc[mask], en.loc[mask]
            close.append(dict(metric=k, network=nm, n=int(a_.notna().sum()),
                              aorc_closer=round(float((a_ < n_).mean()), 3),
                              median_bias_aorc=round(float((w[(k, 'AORC')] - w[(k, 'GAUGE')]).loc[mask].median()), 4),
                              median_bias_nldas=round(float((w[(k, 'NLDAS')] - w[(k, 'GAUGE')]).loc[mask].median()), 4)))
    pd.DataFrame(close).to_csv(os.path.join(out, 'tables', 'gauge_closeness.csv'), index=False)

    _figure(m, w, os.path.join(out, 'figures', 'gauge_intermittency.png'))
    log('Done.')
    print(summ[summ['network'] == 'All'].to_string(index=False))


def _figure(m, w, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pipeline.fig_style import apply_publication_style, CA, CN
    apply_publication_style(font_size=14, axes_linewidth=2.0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, k, lbl in [(axes[0], 'wet_day_frac', 'Wet-day fraction (≥1 mm)'),
                       (axes[1], 'wet_mean_mm', 'Wet-day intensity (mm day$^{-1}$)'),
                       (axes[2], 'dry_day_frac_zero', 'Fraction of days with zero rain')]:
        g = w[(k, 'GAUGE')]
        for src, c in (('AORC', CA), ('NLDAS', CN)):
            ax.scatter(g, w[(k, src)], s=6, alpha=0.35, color=c, label='NLDAS-3' if src == 'NLDAS' else src, edgecolors='none')
        lo, hi = np.nanpercentile(np.r_[g, w[(k, 'AORC')], w[(k, 'NLDAS')]], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f'Gauge: {lbl}'); ax.set_ylabel(f'Gridded: {lbl}')
    axes[0].legend(frameon=False, markerscale=3)
    plt.tight_layout(); fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--label')
    ap.add_argument('--rebuild-cache', action='store_true')
    a = ap.parse_args()
    if a.full or not (a.start and a.end):
        s, e, lbl = EXT.PERIOD_START, EXT.PERIOD_END, 'full_2001_2020'
    else:
        s, e, lbl = a.start, a.end, a.label or f'{a.start}_{a.end}'
    run(s, e, lbl, a.rebuild_cache)
