"""
diagnose_inputs.py — Check the processed AORC / NLDAS-3 files BEFORE re-running.
================================================================================
Run this first, once the processed files are in place. It reads a handful of days spread
over the record and reports, for each product:

  * variables, dimensions, number of time steps per file, lat order, grid spacing
  * the units attribute of precipitation and the mm/day factor it implies
    (fails loudly on a units string the pipeline does not recognise)
  * whether Tmax/Tmin are available (explicit variables or sub-daily steps)
  * valid-pixel counts and the bounding box of valid data
  * how many NLDAS-3 cells carry values where AORC is NaN (Gulf / ocean cells)
  * domain means on each product's OWN valid pixels vs on the COMMON footprint —
    the difference is the footprint-mixing error the old loaders had
  * paired / unpaired dates over the study period

    py -3.11 scripts/diagnose_inputs.py                 # 8 sample days
    py -3.11 scripts/diagnose_inputs.py --days 24 --out results/diagnostics.json
"""

from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_config import EXT

# Log lines contain non-ASCII (ET₀, °C, −); keep a redirected Windows console from crashing.
for _stream in (sys.stdout, sys.stderr):
    try: _stream.reconfigure(errors='replace')
    except (AttributeError, ValueError): pass
from pipeline import gridpass as GP
from pipeline.units import precip_factor_to_mm_day

VARS = ['precip', 'temp', 'temp_max', 'temp_min', 'shortwave', 'longwave', 'pressure',
        'specific_humidity', 'u_wind', 'v_wind']


def file_report(fp: str, raw_map: dict) -> dict:
    rep = {'file': os.path.basename(fp)}
    with xr.open_dataset(fp) as ds:
        lat_n, lon_n = GP._coord_names(ds)
        lats, lons = ds[lat_n].values, ds[lon_n].values
        rep.update(dims={k: int(v) for k, v in ds.sizes.items()},
                   data_vars=sorted(ds.data_vars),
                   lat_order='descending' if lats[0] > lats[-1] else 'ascending',
                   dlat=float(abs(np.median(np.diff(lats)))), dlon=float(abs(np.median(np.diff(lons)))),
                   lat_range=[float(lats.min()), float(lats.max())],
                   lon_range=[float(lons.min()), float(lons.max())],
                   time_steps=int(ds.sizes.get('time', 1)))
        units = {}
        for v in VARS:
            n = GP._find(ds, raw_map.get(v, [v]))
            if n:
                units[v] = f"{n} [{ds[n].attrs.get('units', '?')}]"
        rep['variables'] = units
        pn = GP._find(ds, raw_map['precip'])
        if pn:
            u = str(ds[pn].attrs.get('units', ''))
            try:
                rep['precip_factor_to_mm_day'] = precip_factor_to_mm_day(u)
            except ValueError as e:
                rep['precip_factor_to_mm_day'] = f'ERROR: {e}'
    f = GP.read_product_day(fp, raw_map, VARS)
    rep['tmax_tmin_available'] = 'temp_max' in f and 'temp_min' in f
    valid = {}
    for v in VARS:
        if v in f:
            a = f[v]; ok = np.isfinite(a)
            valid[v] = dict(n_valid=int(ok.sum()), n_cells=int(a.size),
                            mean=float(np.nanmean(a)) if ok.any() else None,
                            p01=float(np.nanpercentile(a, 1)) if ok.any() else None,
                            p99=float(np.nanpercentile(a, 99)) if ok.any() else None)
    rep['valid'] = valid
    t = f.get('temp')
    if t is not None and np.isfinite(t).any():
        ok = np.isfinite(t)
        la, lo = f['_lat'], f['_lon']
        rep['valid_bbox'] = [float(lo[ok.any(0)].min()), float(la[ok.any(1)].min()),
                             float(lo[ok.any(0)].max()), float(la[ok.any(1)].max())]
    return rep


def footprint_report(pair: tuple[str, str]) -> dict:
    """Own-mask vs common-footprint domain means for one day."""
    grid = GP.CommonGrid([pair], bbox=EXT.SOUTHERN_BBOX, domains={'southern': None},
                         landmask_geojson=EXT.LANDMASK_GEOJSON)
    vars_ = ['precip', 'temp', 'shortwave', 'specific_humidity']
    a = GP.read_product_day(pair[0], GP.AORC_RAW, vars_)
    n = GP.read_product_day(pair[1], GP.NLDAS_RAW, vars_)
    out = {'grid': grid.summary()}
    # NLDAS cells with values where AORC (sampled to the NLDAS grid) is NaN
    t_a = grid.aorc_on_grid(a['temp'], a['_lat'], a['_lon']); t_n = grid.nldas_on_grid(n['temp'], n['_lat'], n['_lon'])
    out['nldas_valid_where_aorc_nan'] = int((np.isfinite(t_n) & ~np.isfinite(t_a)).sum())
    out['aorc_valid_where_nldas_nan'] = int((np.isfinite(t_a) & ~np.isfinite(t_n)).sum())
    rows = {}
    fp = grid.footprint
    for v in vars_:
        if v not in a or v not in n:
            continue
        ga, gn = grid.aorc_on_grid(a[v], a['_lat'], a['_lon']), grid.nldas_on_grid(n[v], n['_lat'], n['_lon'])
        both = fp & np.isfinite(ga) & np.isfinite(gn)
        rows[v] = dict(aorc_own=float(np.nanmean(a[v])), nldas_own=float(np.nanmean(n[v])),
                       aorc_common=float(ga[both].mean()) if both.any() else None,
                       nldas_common=float(gn[both].mean()) if both.any() else None,
                       n_common=int(both.sum()))
        if both.any():
            rows[v]['diff_own'] = rows[v]['nldas_own'] - rows[v]['aorc_own']
            rows[v]['diff_common'] = rows[v]['nldas_common'] - rows[v]['aorc_common']
    out['domain_means'] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=8)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    pairs, only_a, only_n = GP.discover_pairs(EXT.AORC_DIR, EXT.NLDAS_DIR, EXT.PERIOD_START,
                                              EXT.PERIOD_END, EXT.AORC_PATTERN, EXT.NLDAS_PATTERN)
    expected = pd.date_range(EXT.PERIOD_START, EXT.PERIOD_END, freq='D')
    rep = dict(aorc_dir=EXT.AORC_DIR, nldas_dir=EXT.NLDAS_DIR,
               period=[EXT.PERIOD_START, EXT.PERIOD_END], expected_days=len(expected),
               paired_days=len(pairs), aorc_only=len(only_a), nldas_only=len(only_n),
               missing_in_both=len(set(expected) - set(pairs) - set(only_a) - set(only_n)))
    print(json.dumps({k: v for k, v in rep.items()}, indent=1))
    if not pairs:
        print('No paired files — check HYDRO_DATA_DIR and the file patterns.'); return

    dates = sorted(pairs)
    pick = [dates[i] for i in np.linspace(0, len(dates) - 1, min(a.days, len(dates))).astype(int)]
    rep['samples'] = []
    for d in pick:
        fa, fn = pairs[d]
        try:
            s = dict(date=str(d.date()), aorc=file_report(fa, GP.AORC_RAW),
                     nldas=file_report(fn, GP.NLDAS_RAW), footprint=footprint_report((fa, fn)))
        except Exception as e:
            print(f'\n── {d.date()} ─ UNREADABLE: {e}')
            rep.setdefault('unreadable', []).append(dict(date=str(d.date()), error=str(e)))
            continue
        rep['samples'].append(s)
        dm = s['footprint']['domain_means']
        print(f"\n── {d.date()} ─ steps A/N: {s['aorc']['time_steps']}/{s['nldas']['time_steps']} | "
              f"precip factor A/N: {s['aorc'].get('precip_factor_to_mm_day')}/"
              f"{s['nldas'].get('precip_factor_to_mm_day')} | Tmax/Tmin A/N: "
              f"{s['aorc']['tmax_tmin_available']}/{s['nldas']['tmax_tmin_available']}")
        print(f"   NLDAS-valid & AORC-NaN cells: {s['footprint']['nldas_valid_where_aorc_nan']}")
        for v, r in dm.items():
            if 'diff_own' in r:
                print(f"   {v:18s} NLDAS−AORC  own masks {r['diff_own']:+.4g}   common {r['diff_common']:+.4g}")

    if not rep['samples']:
        print('No readable sample days.'); return
    first = rep['samples'][0]
    print('\nAORC variables :', first['aorc']['variables'])
    print('NLDAS variables:', first['nldas']['variables'])
    errs = [s['date'] for s in rep['samples']
            for p in ('aorc', 'nldas') if str(s[p].get('precip_factor_to_mm_day', '')).startswith('ERROR')]
    if errs:
        print(f'\n!! Unrecognised precipitation units on {errs} — fix pipeline/units.py before running.')
    out = a.out or os.path.join(EXT.RESULTS_ROOT, 'diagnose_inputs.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(f'\nFull report: {out}')


if __name__ == '__main__':
    main()
