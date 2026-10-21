"""
survey_grids.py — List the distinct lat/lon grids among the processed daily files
================================================================================
Reads only the coordinate vectors of every AORC and NLDAS-3 processed file and writes
results/grid_survey.csv (one row per file) plus a summary of each distinct grid with
its file count and date range. Processed files that were clipped with a different
geometry or resolution show up as extra grids.

    py -3.11 scripts/survey_grids.py
"""

import os, sys, time
import numpy as np
import pandas as pd
import netCDF4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_config import EXT
from pipeline import gridpass as GP


def coords(fp):
    with netCDF4.Dataset(fp) as ds:
        names = list(ds.variables)
        la = next(c for c in names if c.lower() in ('lat', 'latitude', 'y'))
        lo = next(c for c in names if c.lower() in ('lon', 'longitude', 'x'))
        lat = np.asarray(ds.variables[la][:], float); lon = np.asarray(ds.variables[lo][:], float)
    return lat, lon


def main():
    pairs, only_a, only_n = GP.discover_pairs(EXT.AORC_DIR, EXT.NLDAS_DIR, EXT.PERIOD_START, EXT.PERIOD_END,
                                              EXT.AORC_PATTERN, EXT.NLDAS_PATTERN)
    rows, t0 = [], time.time()
    for k, (d, (af, nf)) in enumerate(sorted(pairs.items())):
        for prod, fp in (('aorc', af), ('nldas', nf)):
            try:
                lat, lon = coords(fp)
                rows.append(dict(date=d.date(), product=prod, file=os.path.basename(fp), nlat=len(lat), nlon=len(lon),
                                 lat_min=round(lat.min(), 4), lat_max=round(lat.max(), 4),
                                 lon_min=round(lon.min(), 4), lon_max=round(lon.max(), 4),
                                 dlat=round(float(abs(np.median(np.diff(lat)))), 5),
                                 lat_desc=bool(lat[0] > lat[-1]), error=''))
            except Exception as e:
                rows.append(dict(date=d.date(), product=prod, file=os.path.basename(fp), error=str(e)))
        if (k + 1) % 1000 == 0:
            print(f'{k + 1}/{len(pairs)} days  {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(EXT.RESULTS_ROOT, 'grid_survey.csv')
    df.to_csv(out, index=False)
    key = ['product', 'nlat', 'nlon', 'lat_min', 'lat_max', 'lon_min', 'lon_max', 'dlat', 'lat_desc']
    ok = df[df['error'] == '']
    summ = (ok.groupby(key).agg(n_files=('date', 'size'), first=('date', 'min'), last=('date', 'max'))
              .reset_index().sort_values(['product', 'n_files'], ascending=[True, False]))
    pd.set_option('display.width', 250)
    print('\nDistinct grids:\n' + summ.to_string(index=False))
    if (df['error'] != '').any():
        print(f"\n{(df['error'] != '').sum()} file(s) could not be read for coordinates")
    print(f'\nPer-file table: {out}')


if __name__ == '__main__':
    main()
