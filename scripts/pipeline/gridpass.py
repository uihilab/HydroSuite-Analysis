"""
gridpass.py — Common-footprint, single-pass gridded analysis
=============================================================
Reads AORC and NLDAS-3 daily files onto ONE common grid with ONE validity mask, then
accumulates everything the pipeline needs in a single pass over the record:

  * domain-mean daily series per domain (southern box, Louisiana box, basins)
  * per-pixel PM-ET₀ for both products (per-pixel latitude for Ra) and the per-pixel
    one-at-a-time (OAT) substitution, averaged to the same domains
  * full-period pixel maps (mean, bias, RMSE, Pearson r) for selected variables
  * per-pixel precipitation intermittency (wet-day fraction, spell lengths, intensity)
  * full daily series at a seeded random sample of pixels, for pixel-scale extremes
    (GEV / POT) and per-pixel P90 intensity

Why this replaces the per-product domain means
----------------------------------------------
The earlier loaders averaged each product over whatever pixels it happened to hold:
AORC was cut to one box (-106.5..-75.5, 24.5..37.5) and NLDAS-3 to another
(-106.6..-75, 25.8..36.5), pixel cleaning was applied to AORC before averaging but to
NLDAS-3 only after, and no land/sea mask was shared. Domain-mean differences then mix
product differences with footprint differences. Here a pixel enters any statistic only
if BOTH products are valid there.

Grid alignment
--------------
The common grid is the NLDAS-3 0.01° grid (clipped to the configured bounding box and
to AORC's extent). AORC (~0.0083°) is sampled by nearest neighbour onto it; points
whose nearest AORC cell is more than 0.75 cell away are invalid. Both products are
~1 km, so this is coordinate alignment, not resampling across scales.

Wind
----
Wind speed is computed per pixel from (u, v) before any averaging. The earlier code
took the speed of the domain-averaged vector, √(ū² + v̄²), which underestimates mean
speed. Note the daily files themselves hold daily-mean u and v, so per-pixel daily
speed still slightly underestimates the mean of hourly speeds; that limitation is in
the preprocessing and applies to both products equally.
"""

from __future__ import annotations
import os
import json
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath

from pipeline.units import precip_factor_to_mm_day, clean, wind_speed, WIND_10M_TO_2M
from pipeline.et_calc import pm_et0, SWAP_VARS

# Raw variable names per product. Tmax/Tmin are optional: used if present in files,
# or derived from sub-daily time steps if a file holds more than one.
AORC_RAW = {
    'precip': ['precip', 'APCP_surface'],
    'temp': ['temp', 'TMP_2maboveground'],
    'temp_max': ['temp_max', 'tmax'],
    'temp_min': ['temp_min', 'tmin'],
    'shortwave': ['shortwave', 'DSWRF_surface'],
    'longwave': ['longwave', 'DLWRF_surface'],
    'pressure': ['pressure', 'PRES_surface'],
    'specific_humidity': ['specific_humidity', 'SPFH_2maboveground'],
    'u_wind': ['u_wind', 'UGRD_10maboveground'],
    'v_wind': ['v_wind', 'VGRD_10maboveground'],
}
NLDAS_RAW = {
    'precip': ['Rainf', 'precip'],
    'temp': ['Tair', 'temp'],
    'temp_max': ['Tair_max', 'Tmax', 'temp_max'],
    'temp_min': ['Tair_min', 'Tmin', 'temp_min'],
    'shortwave': ['SWdown', 'shortwave'],
    'longwave': ['LWdown', 'longwave'],
    'pressure': ['PSurf', 'pressure'],
    'specific_humidity': ['Qair', 'specific_humidity'],
    'u_wind': ['Wind_E', 'u_wind'],
    'v_wind': ['Wind_N', 'v_wind'],
}
BASE_VARS = ['precip', 'temp', 'shortwave', 'longwave', 'pressure', 'specific_humidity',
             'u_wind', 'v_wind']
MAP_VARS_DEFAULT = ['precip', 'temp', 'shortwave']
WET_THRESHOLD_MM = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# File discovery
# ══════════════════════════════════════════════════════════════════════════════
def _date_from(fp: str, regex: str):
    m = re.search(regex, os.path.basename(fp))
    return pd.to_datetime(m.group(1), format='%Y%m%d') if m else None


def discover_pairs(aorc_dir, nldas_dir, start, end,
                   aorc_pattern='aorc_*_south_states.nc', nldas_pattern='*_southern.nc'):
    """Dates present in BOTH products within [start, end] → {date: (aorc_fp, nldas_fp)}."""
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    a = {d: fp for fp in glob.glob(os.path.join(aorc_dir, aorc_pattern))
         if (d := _date_from(fp, r'aorc_(\d{8})_south_states')) is not None and s <= d <= e}
    n = {d: fp for fp in glob.glob(os.path.join(nldas_dir, nldas_pattern))
         if (d := _date_from(fp, r'(\d{8})_southern')) is not None and s <= d <= e}
    common = sorted(set(a) & set(n))
    return {d: (a[d], n[d]) for d in common}, sorted(set(a) - set(n)), sorted(set(n) - set(a))


# ══════════════════════════════════════════════════════════════════════════════
# Reading one product-day
# ══════════════════════════════════════════════════════════════════════════════
def _coord_names(ds):
    lat = next((c for c in ds.coords if c.lower() in ('lat', 'latitude', 'y')), None)
    lon = next((c for c in ds.coords if c.lower() in ('lon', 'longitude', 'x')), None)
    if lat is None or lon is None:
        raise ValueError(f'No lat/lon coordinates in dataset (coords: {list(ds.coords)})')
    return lat, lon


def _find(ds, names):
    return next((n for n in names if n in ds.data_vars), None)


def _to_2d(arr: np.ndarray, how: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        return a
    a = a.reshape((-1,) + a.shape[-2:])
    if a.shape[0] == 1:
        return a[0]
    with np.errstate(all='ignore'):
        return {'sum': np.nansum, 'mean': np.nanmean, 'max': np.nanmax, 'min': np.nanmin}[how](a, axis=0)


class _NC:
    """
    Minimal read-only view of a NetCDF file through netCDF4 directly. xarray's
    netCDF4 backend queries the compression filters of every variable on open, which
    costs ~0.15 s per variable with netCDF4 1.7 (plugin probing) — about 3 s per
    product-day, i.e. hours over 7,305 days. Direct reads take ~0.02 s per file.
    """

    def __init__(self, fp, fresh: bool = False):
        import netCDF4
        self.fp = fp
        if fresh:                         # bytes straight from the device (see pipeline.freshread)
            from pipeline.freshread import read_unbuffered
            self._ds = netCDF4.Dataset(os.path.basename(fp), memory=read_unbuffered(fp))
        else:
            self._ds = netCDF4.Dataset(fp)
        self._ds.set_auto_mask(True)
        self.data_vars = {k for k, v in self._ds.variables.items() if v.ndim >= 2}
        names = list(self._ds.variables)
        self.lat = next((c for c in names if c.lower() in ('lat', 'latitude', 'y')), None)
        self.lon = next((c for c in names if c.lower() in ('lon', 'longitude', 'x')), None)
        if self.lat is None or self.lon is None:
            raise ValueError(f'No lat/lon coordinates in {fp} (variables: {names})')

    def values(self, name):
        try:
            a = self._ds.variables[name][:]
        except Exception as e:          # e.g. a corrupt HDF5 chunk in one variable
            raise OSError(f'{os.path.basename(self.fp)}: variable {name!r}: {e}') from e
        return np.ma.filled(np.ma.asarray(a).astype(float), np.nan)

    def units(self, name):
        return str(getattr(self._ds.variables[name], 'units', ''))

    def close(self):
        self._ds.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def read_product_day(fp: str, raw_map: dict, vars_: list[str], fresh: bool = False) -> dict:
    """
    Return {var: 2-D array in analysis units, lat ascending}, plus '_lat'/'_lon'.
    Precip → mm/day from the units attribute; files holding several time steps are
    aggregated (sum for accumulations, mean × factor for rates); Tmax/Tmin come from
    explicit variables or, failing that, from max/min over sub-daily steps.
    """
    out = {}
    with _NC(fp, fresh=fresh) as ds:
        lats = ds.values(ds.lat).ravel()
        lons = ds.values(ds.lon).ravel()
        flip = lats[0] > lats[-1]
        out['_lat'] = lats[::-1] if flip else lats
        out['_lon'] = lons
        for v in vars_:
            name = _find(ds, raw_map.get(v, [v]))
            if name is None:
                # derive Tmax/Tmin from sub-daily temperature if available
                if v in ('temp_max', 'temp_min') and (tn := _find(ds, raw_map['temp'])):
                    arr = ds.values(tn)
                    if arr.ndim == 3 and arr.shape[0] > 1:
                        a2 = _to_2d(arr, 'max' if v == 'temp_max' else 'min')
                        out[v] = clean(a2[::-1] if flip else a2, v).astype(np.float32)
                continue
            arr = ds.values(name)
            if v == 'precip':
                f = precip_factor_to_mm_day(ds.units(name))
                a2 = _to_2d(arr, 'sum') if f == 1.0 else _to_2d(arr, 'mean') * f
            elif v == 'temp_max':
                a2 = _to_2d(arr, 'max')
            elif v == 'temp_min':
                a2 = _to_2d(arr, 'min')
            else:
                a2 = _to_2d(arr, 'mean')
            out[v] = clean(a2[::-1] if flip else a2, v).astype(np.float32)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Common grid, footprint and domain masks
# ══════════════════════════════════════════════════════════════════════════════
def read_pair_robust(af: str, nf: str, vars_: list[str], retries: int = 3):
    """
    Read one day of both products. On failure, re-read the bytes straight from the
    device (bypassing the OS cache) up to ``retries`` times.
    Returns (aorc, nldas, n_fresh_attempts_used); raises the last error if all fail.
    """
    import time
    try:
        return read_product_day(af, AORC_RAW, vars_), read_product_day(nf, NLDAS_RAW, vars_), 0
    except Exception as e:
        err = e
    for attempt in range(1, retries + 1):
        time.sleep(attempt)
        try:
            return (read_product_day(af, AORC_RAW, vars_, fresh=True),
                    read_product_day(nf, NLDAS_RAW, vars_, fresh=True), attempt)
        except Exception as e:
            err = e
    raise err


def _nearest_index(src: np.ndarray, tgt: np.ndarray, max_dist: float):
    """Nearest source index for each target coordinate (src ascending); -1 if too far."""
    i = np.clip(np.searchsorted(src, tgt), 1, len(src) - 1)
    left, right = src[i - 1], src[i]
    i = np.where(np.abs(tgt - left) <= np.abs(right - tgt), i - 1, i)
    i[np.abs(src[i] - tgt) > max_dist] = -1
    return i


def polygon_mask(lats: np.ndarray, lons: np.ndarray, geojson_path: str) -> np.ndarray:
    """Vectorized point-in-polygon of grid-cell centres (all polygons and holes)."""
    with open(geojson_path) as fh:
        gj = json.load(fh)
    feats = gj.get('features', [gj])
    lon_g, lat_g = np.meshgrid(lons, lats)
    pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for ft in feats:
        geom = ft.get('geometry', ft)
        polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
        for poly in polys:
            ring_in = MplPath(np.asarray(poly[0])[:, :2]).contains_points(pts)
            for hole in poly[1:]:
                ring_in &= ~MplPath(np.asarray(hole)[:, :2]).contains_points(pts)
            inside |= ring_in
    return inside.reshape(lon_g.shape)


class CommonGrid:
    """
    NLDAS-3 grid (clipped to ``bbox``) with an AORC nearest-neighbour index and a
    footprint where BOTH products are valid.
    """

    def __init__(self, sample_pairs: list[tuple[str, str]],
                 bbox: tuple[float, float, float, float] | None = None,
                 domains: dict[str, str | None] | None = None,
                 landmask_geojson: str | None = None,
                 footprint_vars=('temp', 'precip', 'shortwave')):
        a0 = read_product_day(sample_pairs[0][0], AORC_RAW, list(footprint_vars))
        n0 = read_product_day(sample_pairs[0][1], NLDAS_RAW, list(footprint_vars))
        alat, alon, nlat, nlon = a0['_lat'], a0['_lon'], n0['_lat'], n0['_lon']

        lon_min = max(nlon.min(), alon.min()); lon_max = min(nlon.max(), alon.max())
        lat_min = max(nlat.min(), alat.min()); lat_max = min(nlat.max(), alat.max())
        if bbox is not None:
            lon_min, lat_min = max(lon_min, bbox[0]), max(lat_min, bbox[1])
            lon_max, lat_max = min(lon_max, bbox[2]), min(lat_max, bbox[3])
        self.ni = np.where((nlat >= lat_min) & (nlat <= lat_max))[0]
        self.nj = np.where((nlon >= lon_min) & (nlon <= lon_max))[0]
        self.lats, self.lons = nlat[self.ni], nlon[self.nj]

        dlat = np.median(np.diff(alat)); dlon = np.median(np.diff(alon))
        self.ai = _nearest_index(alat, self.lats, 0.75 * abs(dlat))
        self.aj = _nearest_index(alon, self.lons, 0.75 * abs(dlon))
        self.shape = (len(self.lats), len(self.lons))
        # Files are mapped by THEIR OWN coordinates (some processed files were clipped to a
        # different extent); index arrays are cached per distinct coordinate vector.
        self._tol = dict(a_lat=0.75 * abs(dlat), a_lon=0.75 * abs(dlon),
                         n_lat=0.5 * abs(np.median(np.diff(nlat))), n_lon=0.5 * abs(np.median(np.diff(nlon))))
        self._idx_cache = {}
        self.grid_counts = {}
        for key, src, tgt, idx in (('a_lat', alat, self.lats, self.ai), ('a_lon', alon, self.lons, self.aj),
                                   ('n_lat', nlat, self.lats, self.ni), ('n_lon', nlon, self.lons, self.nj)):
            self._idx_cache[(key,) + self._sig(src)] = idx

        # Footprint: valid in BOTH products for every footprint variable on every sample day
        fp = np.ones(self.shape, dtype=bool)
        fp &= (self.ai >= 0)[:, None] & (self.aj >= 0)[None, :]
        for af, nf in sample_pairs:
            a = read_product_day(af, AORC_RAW, list(footprint_vars))
            n = read_product_day(nf, NLDAS_RAW, list(footprint_vars))
            for v in footprint_vars:
                if v not in a or v not in n:
                    raise ValueError(f'Footprint variable {v!r} missing in {af} or {nf}')
                fp &= (np.isfinite(self.aorc_on_grid(a[v], a['_lat'], a['_lon']))
                       & np.isfinite(self.nldas_on_grid(n[v], n['_lat'], n['_lon'])))
        if landmask_geojson:
            fp &= polygon_mask(self.lats, self.lons, landmask_geojson)
        self.footprint = fp

        # Domain masks (always intersected with the footprint). A configured domain whose
        # boundary file is missing raises — never a silent fall-back to the full grid.
        self.domains: dict[str, np.ndarray] = {}
        for name, path in (domains or {'southern': None}).items():
            if path is None:
                m = np.ones(self.shape, dtype=bool)
            else:
                if not os.path.exists(path):
                    raise FileNotFoundError(f'Boundary for domain {name!r} not found: {path}')
                m = polygon_mask(self.lats, self.lons, path)
            m &= self.footprint
            if not m.any():
                raise ValueError(f'Domain {name!r} has no valid common-footprint pixels')
            self.domains[name] = m
        self.lat2d = np.broadcast_to(self.lats[:, None], self.shape)

    # ── mapping helpers ──
    @staticmethod
    def _sig(v):
        v = np.asarray(v, float)
        return (len(v), round(float(v[0]), 5), round(float(v[-1]), 5))

    def _index(self, key, src, tgt):
        k = (key,) + self._sig(src)
        if k not in self._idx_cache:
            self._idx_cache[k] = _nearest_index(np.asarray(src, float), tgt, self._tol[key])
        return self._idx_cache[k]

    def _take(self, arr2d, ii, jj):
        out = np.asarray(arr2d)[np.ix_(np.clip(ii, 0, None), np.clip(jj, 0, None))].astype(np.float32)
        out[(ii < 0)[:, None] | (jj < 0)[None, :]] = np.nan
        return out

    def _count(self, prod, lat, lon):
        k = f'{prod}:{len(lat)}x{len(lon)} lat {float(lat[0]):.3f}..{float(lat[-1]):.3f} ' \
            f'lon {float(lon[0]):.3f}..{float(lon[-1]):.3f}'
        self.grid_counts[k] = self.grid_counts.get(k, 0) + 1

    def aorc_on_grid(self, arr2d, lat=None, lon=None):
        if lat is None:
            return self._take(arr2d, self.ai, self.aj)
        return self._take(arr2d, self._index('a_lat', lat, self.lats), self._index('a_lon', lon, self.lons))

    def nldas_on_grid(self, arr2d, lat=None, lon=None):
        if lat is None:
            return self._take(arr2d, self.ni, self.nj)
        return self._take(arr2d, self._index('n_lat', lat, self.lats), self._index('n_lon', lon, self.lons))

    def summary(self) -> dict:
        return dict(n_lat=self.shape[0], n_lon=self.shape[1],
                    lat_range=(float(self.lats.min()), float(self.lats.max())),
                    lon_range=(float(self.lons.min()), float(self.lons.max())),
                    footprint_pixels=int(self.footprint.sum()),
                    footprint_fraction=round(float(self.footprint.mean()), 4),
                    domains={k: int(v.sum()) for k, v in self.domains.items()})


# ══════════════════════════════════════════════════════════════════════════════
# Accumulators
# ══════════════════════════════════════════════════════════════════════════════
class _Moments:
    """Per-pixel streaming moments for mean / bias / RMSE / Pearson r maps (float64)."""

    def __init__(self, shape, offset=0.0):
        self.off = offset       # subtracted before squaring (temperature in K) for precision
        self.n = np.zeros(shape, np.int32)
        self.sa = np.zeros(shape); self.sb = np.zeros(shape)
        self.saa = np.zeros(shape); self.sbb = np.zeros(shape); self.sab = np.zeros(shape)

    def add(self, a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        a = np.where(ok, a - self.off, 0.0); b = np.where(ok, b - self.off, 0.0)
        self.n += ok; self.sa += a; self.sb += b
        self.saa += a * a; self.sbb += b * b; self.sab += a * b

    def maps(self, min_n=30):
        with np.errstate(all='ignore'):
            n = np.where(self.n >= min_n, self.n, np.nan)
            ma, mb = self.sa / n, self.sb / n
            va, vb = self.saa / n - ma ** 2, self.sbb / n - mb ** 2
            cov = self.sab / n - ma * mb
            mse = (self.saa - 2 * self.sab + self.sbb) / n
            return dict(mean_aorc=ma + self.off, mean_nldas=mb + self.off,
                        bias=mb - ma, rmse=np.sqrt(np.maximum(mse, 0)),
                        r=cov / np.sqrt(va * vb), n=self.n)


class _Intermittency:
    """Per-pixel wet/dry statistics for one product, streamed day by day."""

    def __init__(self, shape, thr=WET_THRESHOLD_MM):
        self.thr = thr
        self.ndays = np.zeros(shape, np.int32); self.nwet = np.zeros(shape, np.int32)
        self.wetsum = np.zeros(shape)
        self.run_w = np.zeros(shape, np.int32); self.run_d = np.zeros(shape, np.int32)
        self.nsp_w = np.zeros(shape, np.int32); self.sum_w = np.zeros(shape, np.int64)
        self.nsp_d = np.zeros(shape, np.int32); self.sum_d = np.zeros(shape, np.int64)
        self.max_w = np.zeros(shape, np.int32); self.max_d = np.zeros(shape, np.int32)

    def _close(self, run, mask, nsp, ssum, smax):
        done = mask & (run > 0)
        nsp += done; ssum += np.where(done, run, 0)
        np.maximum(smax, np.where(done, run, 0), out=smax)
        run[done] = 0

    def add(self, p):
        ok = np.isfinite(p)
        wet = ok & (p >= self.thr); dry = ok & ~wet
        self.ndays += ok; self.nwet += wet; self.wetsum += np.where(wet, p, 0)
        self._close(self.run_d, wet | ~ok, self.nsp_d, self.sum_d, self.max_d)   # a dry run ends
        self._close(self.run_w, dry | ~ok, self.nsp_w, self.sum_w, self.max_w)   # a wet run ends
        self.run_w += wet; self.run_d += dry

    def maps(self):
        self._close(self.run_w, np.ones_like(self.run_w, bool), self.nsp_w, self.sum_w, self.max_w)
        self._close(self.run_d, np.ones_like(self.run_d, bool), self.nsp_d, self.sum_d, self.max_d)
        with np.errstate(all='ignore'):
            return dict(wet_day_frac=self.nwet / self.ndays,
                        wet_mean_mm=self.wetsum / self.nwet,
                        wet_spell_mean=self.sum_w / self.nsp_w,
                        dry_spell_mean=self.sum_d / self.nsp_d,
                        wet_spell_max=self.max_w.astype(float),
                        dry_spell_max=self.max_d.astype(float),
                        n_days=self.ndays)


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def _et0_fields(f: dict, lat2d, doy, rnl_method):
    """Per-pixel PM-ET₀ from one product's fields (dict of 2-D arrays)."""
    return pm_et0(T_K=f['temp'], Rs=f['shortwave'], Rl=f['longwave'], P_Pa=f['pressure'],
                  q=f['specific_humidity'], u2=f['wind_2m'], doy=doy, lat_deg=lat2d,
                  Tmax_K=f.get('temp_max'), Tmin_K=f.get('temp_min'), rnl_method=rnl_method)


def run_gridpass(pairs: dict, grid: CommonGrid, *,
                 map_vars=MAP_VARS_DEFAULT, map_stride: int = 2, et_stride: int = 2,
                 n_sample_pixels: int = 3000, seed: int = 42,
                 wet_threshold: float = WET_THRESHOLD_MM, rnl_method: str = 'forcing',
                 progress_every: int = 365, log=print,
                 repair=None, checkpoint_path: str | None = None,
                 checkpoint_every: int = 180) -> dict:
    """
    One pass over all paired days. Returns
      domain_dfs : {domain: DataFrame of daily domain means (common footprint)}
      maps       : {var: {stat: 2-D}} on the grid subsampled by ``map_stride``
      intermittency : {'aorc'|'nldas': {stat: 2-D}} (same subsampled grid)
      samples    : DataFrame (date × pixel) per product for sampled pixels + their coords
      map_lats, map_lons, grid_summary

    repair(date, aorc_fp, nldas_fp) -> dict : called when a day cannot be read; it may
        rebuild the files from raw (pipeline.repair.repair_pair); the day is retried once.
    checkpoint_path : accumulator state is pickled there every ``checkpoint_every`` days,
        and a checkpoint with identical settings is resumed, so an interrupted run continues.
    """
    sl = (slice(None, None, map_stride), slice(None, None, map_stride))
    esl = (slice(None, None, et_stride), slice(None, None, et_stride))
    mshape = grid.footprint[sl].shape
    offsets = {'temp': 290.0, 'pressure': 100000.0, 'longwave': 350.0}
    moments = {v: _Moments(mshape, offsets.get(v, 0.0)) for v in map_vars}
    inter = {'aorc': _Intermittency(mshape, wet_threshold), 'nldas': _Intermittency(mshape, wet_threshold)}

    rng = np.random.default_rng(seed)
    cand = np.argwhere(grid.footprint)
    pick = cand[rng.choice(len(cand), size=min(n_sample_pixels, len(cand)), replace=False)]
    samp_a, samp_n = {}, {}

    et_lat = grid.lat2d[esl]
    et_domains = {k: m[esl] for k, m in grid.domains.items()}
    read_vars = BASE_VARS + ['temp_max', 'temp_min']
    rows = {k: [] for k in grid.domains}
    skipped = []                     # unreadable days → grid_summary.json['skipped_days']
    repaired = []                    # days rebuilt from raw → grid_summary.json['repaired_days']
    recovered = []                   # read OK on a fresh (uncached) re-read → ['recovered_days']

    items = sorted(pairs.items())
    meta = dict(n_pairs=len(items), first=str(items[0][0].date()), last=str(items[-1][0].date()),
                shape=list(grid.shape), footprint=int(grid.footprint.sum()),
                domains={k: int(v.sum()) for k, v in grid.domains.items()},
                map_vars=list(map_vars), map_stride=map_stride, et_stride=et_stride,
                n_sample_pixels=n_sample_pixels, seed=seed, wet_threshold=wet_threshold,
                rnl_method=rnl_method, version=3)
    start_k = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        import pickle
        with open(checkpoint_path, 'rb') as fh:
            ck = pickle.load(fh)
        if ck.get('meta') == meta:
            moments, inter, rows = ck['moments'], ck['inter'], ck['rows']
            samp_a, samp_n = ck['samp_a'], ck['samp_n']
            skipped, repaired = ck['skipped'], ck['repaired']
            recovered = ck.get('recovered', [])
            grid.grid_counts = ck.get('grid_counts', {})
            start_k = ck['next_k']
            log(f'  [gridpass] resuming from checkpoint at day {start_k}/{len(items)}')
        else:
            log('  [gridpass] checkpoint was made with different settings; starting over')

    def _save_checkpoint(next_k):
        import pickle
        tmp = checkpoint_path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(dict(meta=meta, next_k=next_k, moments=moments, inter=inter, rows=rows,
                             samp_a=samp_a, samp_n=samp_n, skipped=skipped, repaired=repaired,
                             recovered=recovered, grid_counts=grid.grid_counts),
                        fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, checkpoint_path)

    for k in range(start_k, len(items)):
        date, (af, nf) = items[k]
        if checkpoint_path and k > start_k and k % checkpoint_every == 0:
            _save_checkpoint(k)
        try:
            a_raw, n_raw, used = read_pair_robust(af, nf, read_vars)
            if used:
                recovered.append(dict(date=str(date.date()), fresh_attempts=used))
        except Exception as e:
            a_raw, err = None, e
            if repair is not None:
                rec = repair(date, af, nf)
                if rec and all(r['rebuilt'] for r in rec.values()):
                    try:
                        a_raw = read_product_day(af, AORC_RAW, read_vars, fresh=True)
                        n_raw = read_product_day(nf, NLDAS_RAW, read_vars, fresh=True)
                        repaired.append(dict(date=str(date.date()), **rec))
                    except Exception as e2:
                        a_raw, err = None, e2
            if a_raw is None:
                log(f'  [gridpass] {date.date()} skipped: {err}')
                skipped.append(dict(date=str(date.date()), aorc_file=af, nldas_file=nf, error=str(err)))
                continue
        fp = grid.footprint
        A, N = {}, {}
        grid._count('aorc', a_raw['_lat'], a_raw['_lon']); grid._count('nldas', n_raw['_lat'], n_raw['_lon'])
        for v in read_vars:
            if v in a_raw and v in n_raw:
                a = grid.aorc_on_grid(a_raw[v], a_raw['_lat'], a_raw['_lon'])
                n = grid.nldas_on_grid(n_raw[v], n_raw['_lat'], n_raw['_lon'])
                # strict daily pairing: a pixel counts only where BOTH are valid today
                both = fp & np.isfinite(a) & np.isfinite(n)
                A[v] = np.where(both, a, np.nan).astype(np.float32)
                N[v] = np.where(both, n, np.nan).astype(np.float32)
        for F in (A, N):
            if 'u_wind' in F and 'v_wind' in F:
                F['wind_10m'] = wind_speed(F['u_wind'], F['v_wind'])
                F['wind_2m'] = F['wind_10m'] * WIND_10M_TO_2M

        # per-pixel ET₀ and OAT (AORC baseline, one NLDAS variable swapped in)
        et = {}
        if all(v in A and v in N for v in SWAP_VARS):
            doy = date.dayofyear
            As = {v: A[v][esl] for v in A}; Ns = {v: N[v][esl] for v in N}
            et['et0_aorc'] = _et0_fields(As, et_lat, doy, rnl_method)
            et['et0_nldas'] = _et0_fields(Ns, et_lat, doy, rnl_method)
            for v in SWAP_VARS:
                sw = dict(As); sw[v] = Ns[v]
                if v == 'temp':   # swap the temperature trio together
                    for t in ('temp_max', 'temp_min'):
                        if t in Ns: sw[t] = Ns[t]
                et[f'et0_swap_{v}'] = _et0_fields(sw, et_lat, doy, rnl_method)

        for dom, m in grid.domains.items():
            row = {'date': date}
            for v in A:
                with np.errstate(all='ignore'):
                    row[f'{v}_aorc'] = float(np.nanmean(A[v][m])) if np.isfinite(A[v][m]).any() else np.nan
                    row[f'{v}_nldas'] = float(np.nanmean(N[v][m])) if np.isfinite(N[v][m]).any() else np.nan
            em = et_domains[dom]
            for c, fld in et.items():
                vals = fld[em]
                row[c] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row['n_valid_pixels'] = int(np.isfinite(A['temp'][m] + N['temp'][m]).sum()) if 'temp' in A else 0
            rows[dom].append(row)

        for v in map_vars:
            if v in A:
                moments[v].add(A[v][sl], N[v][sl])
        if 'precip' in A:
            inter['aorc'].add(A['precip'][sl]); inter['nldas'].add(N['precip'][sl])
            samp_a[date] = A['precip'][pick[:, 0], pick[:, 1]]
            samp_n[date] = N['precip'][pick[:, 0], pick[:, 1]]

        if progress_every and (k + 1) % progress_every == 0:
            log(f'  [gridpass] {k + 1}/{len(pairs)} days | recovered on re-read {len(recovered)} | '
                f'repaired {len(repaired)} | skipped {len(skipped)}')

    domain_dfs = {}
    for dom, r in rows.items():
        df = pd.DataFrame(r)
        domain_dfs[dom] = df.set_index('date').sort_index() if not df.empty else df

    cols = [f'px{i}' for i in range(len(pick))]
    samples = dict(
        aorc=pd.DataFrame.from_dict(samp_a, orient='index', columns=cols).sort_index(),
        nldas=pd.DataFrame.from_dict(samp_n, orient='index', columns=cols).sort_index(),
        coords=pd.DataFrame({'pixel': cols, 'lat': grid.lats[pick[:, 0]], 'lon': grid.lons[pick[:, 1]]}),
    )
    return dict(domain_dfs=domain_dfs,
                maps={v: m.maps() for v, m in moments.items()},
                intermittency={k: i.maps() for k, i in inter.items()},
                samples=samples,
                map_lats=grid.lats[::map_stride], map_lons=grid.lons[::map_stride],
                grid_summary=dict(grid.summary(), n_days_used=len(pairs) - len(skipped),
                                  skipped_days=skipped, repaired_days=repaired,
                                  recovered_days=recovered, file_grids=grid.grid_counts))


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════
def save_gridpass(res: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for dom, df in res['domain_dfs'].items():
        if not df.empty:
            df.to_parquet(os.path.join(out_dir, f'domain_{dom}.parquet'))
    arrays = {'map_lats': res['map_lats'], 'map_lons': res['map_lons']}
    for v, st in res['maps'].items():
        for s, a in st.items():
            arrays[f'map__{v}__{s}'] = a
    for p, st in res['intermittency'].items():
        for s, a in st.items():
            arrays[f'int__{p}__{s}'] = a
    np.savez_compressed(os.path.join(out_dir, 'spatial.npz'), **arrays)
    for k in ('aorc', 'nldas', 'coords'):
        res['samples'][k].to_parquet(os.path.join(out_dir, f'samples_{k}.parquet'))
    with open(os.path.join(out_dir, 'grid_summary.json'), 'w') as fh:
        json.dump(res['grid_summary'], fh, indent=2, default=str)


def load_gridpass(out_dir: str) -> dict | None:
    if not os.path.exists(os.path.join(out_dir, 'spatial.npz')):
        return None
    z = np.load(os.path.join(out_dir, 'spatial.npz'))
    maps, inter = {}, {}
    for key in z.files:
        if key.startswith('map__'):
            _, v, s = key.split('__'); maps.setdefault(v, {})[s] = z[key]
        elif key.startswith('int__'):
            _, p, s = key.split('__'); inter.setdefault(p, {})[s] = z[key]
    domain_dfs = {}
    for fp in glob.glob(os.path.join(out_dir, 'domain_*.parquet')):
        domain_dfs[os.path.basename(fp)[7:-8]] = pd.read_parquet(fp)
    samples = {k: pd.read_parquet(os.path.join(out_dir, f'samples_{k}.parquet'))
               for k in ('aorc', 'nldas', 'coords')}
    with open(os.path.join(out_dir, 'grid_summary.json')) as fh:
        summary = json.load(fh)
    return dict(domain_dfs=domain_dfs, maps=maps, intermittency=inter, samples=samples,
                map_lats=z['map_lats'], map_lons=z['map_lons'], grid_summary=summary)
