"""
LEGACY MODULE — load_period_by_domain now delegates to pipeline.gridpass.

domains.py — Spatial Domain Masking
=====================================
Clips gridded data to GeoJSON polygon masks so the same analysis
phases can run at multiple spatial scales:
  - Southern US  (full domain — no clipping needed)
  - Louisiana    (state boundary GeoJSON)
  - Basin        (HUC8/HUC10 GeoJSON — Amite, Calcasieu, Upper Ouachita)

Uses a fast numpy ray-casting PIP test so shapely is optional.
If shapely is available it provides exact results; otherwise the
numpy fallback is used (accurate to ~pixel-width at boundaries).
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

try:
    from shapely.geometry import shape, Point
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# ── GeoJSON paths (resolved relative to project root) ─────────────────────────
_HERE  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BOUND = os.path.join(_HERE, 'data', 'raw', 'basins')     # was data/raw/boundaries (absent)

DOMAIN_GEOJSON = {
    'southern'  : None,                                              # full domain
    'louisiana' : os.path.join(_BOUND, 'Louisiana_State.geojson'),   # NOTE: a box, not the state outline
    'amite'     : os.path.join(_BOUND, 'Amite_HUC8.geojson'),
    'calcasieu' : os.path.join(_BOUND, 'Calcasieu_HUC8.geojson'),
    'ouachita'  : os.path.join(_BOUND, 'UpperOuachita_HUC8.geojson'),
}


# Physical plausibility bounds — mirrors loaders._PHYS_BOUNDS
# Values outside these ranges are set to NaN before spatial averaging
_PHYS_BOUNDS = {
    'precip':            (0.0,    200.0),
    'temp':              (240.0,  330.0),
    'shortwave':         (0.0,    500.0),
    'longwave':          (150.0,  550.0),
    'pressure':          (80000., 110000.),
    'specific_humidity': (0.0,    0.040),
    'u_wind':            (-30.0,  30.0),
    'v_wind':            (-30.0,  30.0),
    'wind_10m':          (0.0,    30.0),
    'wind_2m':           (0.0,    25.0),
}


def _load_polygons(geojson_path: str) -> list:
    """Return list of coordinate rings from a GeoJSON file."""
    with open(geojson_path) as fh:
        gj = json.load(fh)
    rings = []
    feats = gj.get('features', [gj])
    for feat in feats:
        geom = feat.get('geometry', feat)
        coords = geom.get('coordinates', [])
        gtype  = geom.get('type', '')
        if gtype == 'Polygon':
            rings.append(coords[0])
        elif gtype == 'MultiPolygon':
            for part in coords:
                rings.append(part[0])
    return rings


def _pip_numpy(lon_grid: np.ndarray, lat_grid: np.ndarray,
               rings: list) -> np.ndarray:
    """
    Ray-casting point-in-polygon test (numpy, no shapely).
    Returns boolean mask (True = inside any ring).
    """
    mask = np.zeros(lon_grid.shape, dtype=bool)
    for ring in rings:
        vx = np.array([v[0] for v in ring])
        vy = np.array([v[1] for v in ring])
        n  = len(vx)
        inside = np.zeros(lon_grid.shape, dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = vx[i], vy[i]
            xj, yj = vx[j], vy[j]
            cond = ((vy[i] > lat_grid) != (vy[j] > lat_grid)) & \
                   (lon_grid < (xj - xi) * (lat_grid - yi) / (yj - yi + 1e-12) + xi)
            inside ^= cond
            j = i
        mask |= inside
    return mask


def _pip_shapely(lon_grid: np.ndarray, lat_grid: np.ndarray,
                 rings: list) -> np.ndarray:
    """Shapely-based point-in-polygon (slower but exact)."""
    from shapely.geometry import MultiPolygon, Polygon
    polys = [Polygon(ring) for ring in rings]
    union = polys[0] if len(polys) == 1 else MultiPolygon(polys).convex_hull
    mask  = np.array([
        union.contains(Point(lon_grid.flat[k], lat_grid.flat[k]))
        for k in range(lon_grid.size)
    ]).reshape(lon_grid.shape)
    return mask


def build_mask(
    lats_1d: np.ndarray,
    lons_1d: np.ndarray,
    domain: str,
) -> np.ndarray:
    """
    Build a boolean 2-D mask (lat × lon) for a named domain.

    Parameters
    ----------
    lats_1d, lons_1d : 1-D coordinate arrays
    domain           : one of 'southern', 'louisiana', 'amite', 'calcasieu', 'ouachita'

    Returns
    -------
    Boolean array (len(lats_1d) × len(lons_1d)), True = in the domain.
    """
    if domain == 'southern' or DOMAIN_GEOJSON.get(domain) is None:
        return np.ones((len(lats_1d), len(lons_1d)), dtype=bool)

    gj_path = DOMAIN_GEOJSON.get(domain)
    if not gj_path or not os.path.exists(gj_path):
        raise FileNotFoundError(
            f"GeoJSON for domain '{domain}' not found at: {gj_path}\n"
            f"Expected at: {_BOUND}/"
        )

    rings = _load_polygons(gj_path)
    lon_g, lat_g = np.meshgrid(lons_1d, lats_1d)

    if HAS_SHAPELY:
        return _pip_shapely(lon_g, lat_g, rings)
    return _pip_numpy(lon_g, lat_g, rings)


def masked_spatial_mean(
    data_2d: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return spatial mean of data_2d over the masked cells."""
    vals = data_2d[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if vals.size > 0 else np.nan


# ── Domain-aware batch loader ─────────────────────────────────────────────────
def load_period_by_domain(
    aorc_dir: str,
    nldas_dir: str,
    start: str,
    end: str,
    domains: list[str] | None = None,
    aorc_pattern: str = 'aorc_*_south_states.nc',
    nldas_pattern: str = '*_southern.nc',
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    {domain: daily DataFrame} on the common footprint (delegates to pipeline.gridpass).

    The previous implementation (a) multiplied NLDAS-3 precipitation by 86400 whenever
    the units string contained "kg" or "s" — including accumulations — (b) applied
    physical bounds to the NLDAS domain MEAN after that conversion, silently dropping
    most days (scale_sensitivity.csv had n = 29), and (c) replaced a missing GeoJSON
    with the full grid. A missing boundary now raises FileNotFoundError.
    """
    import warnings
    from pipeline import gridpass as GP
    warnings.warn('domains.load_period_by_domain is legacy; use pipeline.gridpass',
                  DeprecationWarning, stacklevel=2)
    domains = domains or list(DOMAIN_GEOJSON)
    dom_paths = {}
    for d in domains:
        path = DOMAIN_GEOJSON.get(d)
        if d != 'southern' and (not path or not os.path.exists(path)):
            raise FileNotFoundError(f"GeoJSON for domain '{d}' not found at: {path}")
        dom_paths[d] = None if d == 'southern' else path
    pairs, _, _ = GP.discover_pairs(aorc_dir, nldas_dir, start, end, aorc_pattern, nldas_pattern)
    if not pairs:
        return {d: pd.DataFrame() for d in domains}
    dates = sorted(pairs)
    sample = [pairs[dates[i]] for i in np.linspace(0, len(dates) - 1, min(8, len(dates))).astype(int)]
    grid = GP.CommonGrid(sample, domains=dom_paths)
    res = GP.run_gridpass(pairs, grid, map_vars=[], log=print if verbose else (lambda *_: None))
    return res['domain_dfs']
