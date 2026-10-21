"""
Path and study configuration
============================
Single source of truth for all paths and study periods when running
the analysis pipeline.

Usage in scripts:
    from ext_config import EXT

Data location: set HYDRO_DATA_DIR to the directory that holds aorc/, nldas/, ghcn/ and
data/processed/ (defaults to the repository root). Results are written to
HYDRO_DATA_DIR/results.
"""
import os, re, glob
import pandas as pd

# ── Data Root ─────────────────────────────────────────────────────────
# Defaults to the project root directory, but can be overridden by environment variable.
_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_EXT_ROOT = os.environ.get('HYDRO_DATA_DIR', _DEFAULT_ROOT)

class ExtConfig:
    DATA_ROOT = _EXT_ROOT                      # holds aorc/raw and nldas/ (used to rebuild corrupt files)
    # ── Data sources ────────────────────────────────────────────────────────────
    AORC_DIR  = os.path.join(_EXT_ROOT, 'aorc', 'processed')        # aorc_YYYYMMDD_south_states.nc
    NLDAS_DIR = os.path.join(_EXT_ROOT, 'data', 'processed', 'nldas_southern')  # YYYYMMDD_southern.nc

    AORC_PATTERN  = 'aorc_*_south_states.nc'
    NLDAS_PATTERN = '*_southern.nc'

    # ── Results output ──────────────────────────────────────────────────────────
    RESULTS_ROOT = os.path.join(_EXT_ROOT, 'results')

    # ── Study period (the paired record is 2001-01-01 .. 2020-12-31, 7,305 days) ─
    PERIOD_START = '2001-01-01'
    PERIOD_END   = '2020-12-31'

    # ── Common analysis domain ──────────────────────────────────────────────────
    # Single definition used by every phase. It is the intersection of the two
    # products' clip boxes (AORC: -106.5..-75.5, 24.5..37.5; NLDAS-3 GeoJSON:
    # -106.6..-75.0, 25.8..36.5); gridpass further restricts every statistic to
    # pixels where BOTH products are valid (common footprint).
    SOUTHERN_BBOX = (-106.5, 25.8, -75.5, 36.5)   # (lon_min, lat_min, lon_max, lat_max)

    # Sub-domains: name → GeoJSON path (None = whole common footprint). A listed
    # domain whose file is missing stops the run (no silent full-grid fallback).
    # Louisiana: the state outline (Louisiana_Outline.geojson, written by
    # scripts/processing/utils/get_louisiana_boundary.py) is used when present.
    # Otherwise the run falls back to Louisiana_State.geojson, which is a BOX
    # (-94.1..-88.75, 28.85..33.05) including Gulf water and parts of TX/MS/AR —
    # the log and grid_summary.json say which one was used.
    # HUC8 files: scripts/retrieval/get_huc8_boundaries.py.
    _BASINS = os.path.join(_DEFAULT_ROOT, 'data', 'raw', 'basins')
    LA_OUTLINE = os.path.join(_BASINS, 'Louisiana_Outline.geojson')
    LA_BOX     = os.path.join(_BASINS, 'Louisiana_State.geojson')
    DOMAINS = {
        'southern':  None,
        'louisiana': LA_OUTLINE if os.path.exists(LA_OUTLINE) else LA_BOX,
    }
    OPTIONAL_DOMAINS = {   # included automatically when the file exists
        'amite':     os.path.join(_BASINS, 'Amite_HUC8.geojson'),
        'calcasieu': os.path.join(_BASINS, 'Calcasieu_HUC8.geojson'),
        'ouachita':  os.path.join(_BASINS, 'UpperOuachita_HUC8.geojson'),
    }
    # Optional land polygon (e.g. US states) to exclude water where both products
    # happen to have values; None = rely on the joint-validity footprint only.
    LANDMASK_GEOJSON = os.environ.get('HYDRO_LANDMASK', None)

    # ── Gridded pass settings ───────────────────────────────────────────────────
    FOOTPRINT_SAMPLE_DAYS = 8      # days spread across the record used to build the footprint
    MAP_STRIDE = 2                 # pixel maps on every 2nd cell (0.02°) to bound memory
    ET_STRIDE  = 2                 # per-pixel ET₀ on every 2nd cell
    N_SAMPLE_PIXELS = 3000         # pixels whose full daily series are kept (extremes, P90)
    WET_THRESHOLD_MM = 1.0         # wet-day threshold [mm/day]

    # ── GHCN-Daily (station ET₀ comparison) ─────────────────────────────────────
    # Expected layout: GHCN_DIR/inventory.csv and GHCN_DIR/daily/<station_id>.parquet
    GHCN_DIR = os.environ.get('HYDRO_GHCN_DIR', os.path.join(_EXT_ROOT, 'ghcn'))

    def active_domains(self) -> dict:
        d = dict(self.DOMAINS)
        d.update({k: p for k, p in self.OPTIONAL_DOMAINS.items() if os.path.exists(p)})
        return d

    # ── Variable names in processed files ──────────────────────────────────────
    AORC_PRECIP_VAR  = 'precip'          # mm/day (already aggregated)
    NLDAS_PRECIP_VAR = 'Rainf'           # kg/m²/s — multiply by 86400 for mm/day
    NLDAS_TEMP_VAR   = 'Tair'            # K
    AORC_TEMP_VAR    = 'temp'            # K

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def results_dir(self, period: str) -> str:
        """Return (and create) the results directory for a given period label."""
        if not os.path.exists(_EXT_ROOT):
            raise FileNotFoundError(
                f'Data directory not found: {_EXT_ROOT}\n'
                'Set HYDRO_DATA_DIR to the directory that holds the data.'
            )
        d = os.path.join(self.RESULTS_ROOT, period)
        os.makedirs(os.path.join(d, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(d, 'tables'),  exist_ok=True)
        return d

    def fig_path(self, period: str, name: str) -> str:
        return os.path.join(self.results_dir(period), 'figures', name)

    def tbl_path(self, period: str, name: str) -> str:
        return os.path.join(self.results_dir(period), 'tables', name)

    def _date_from_aorc(self, fp):
        m = re.search(r'aorc_(\d{8})_south_states', os.path.basename(fp))
        return pd.to_datetime(m.group(1), format='%Y%m%d') if m else None

    def _date_from_nldas(self, fp):
        m = re.search(r'(\d{8})_southern', os.path.basename(fp))
        return pd.to_datetime(m.group(1), format='%Y%m%d') if m else None

    def discover_aorc(self, start=None, end=None):
        """Return list of (date, path) tuples for AORC processed files in range."""
        files = sorted(glob.glob(os.path.join(self.AORC_DIR, self.AORC_PATTERN)))
        result = []
        s = pd.to_datetime(start) if start else pd.Timestamp.min
        e = pd.to_datetime(end)   if end   else pd.Timestamp.max
        for fp in files:
            d = self._date_from_aorc(fp)
            if d and s <= d <= e:
                result.append((d, fp))
        return result

    def discover_nldas(self, start=None, end=None):
        """Return list of (date, path) tuples for NLDAS processed files in range."""
        files = sorted(glob.glob(os.path.join(self.NLDAS_DIR, self.NLDAS_PATTERN)))
        result = []
        s = pd.to_datetime(start) if start else pd.Timestamp.min
        e = pd.to_datetime(end)   if end   else pd.Timestamp.max
        for fp in files:
            d = self._date_from_nldas(fp)
            if d and s <= d <= e:
                result.append((d, fp))
        return result

# Singleton — import and use EXT directly
EXT = ExtConfig()
