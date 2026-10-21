"""
regrid.py — Spatial Regridding Utilities
=========================================
Regrids AORC (1km) fields onto the NLDAS-3 grid (0.01° / ~1 km) using bilinear
interpolation so grid-cell-level comparisons are on equal footing.

NLDAS-3 vs NLDAS-2
-------------------
The data used here is NLDAS-3 (NASA GES DISC, NLDAS_FOR0010_D), which runs at
0.01° (~1 km) — the same resolution as AORC. This means:
  * Domain-mean analyses (Phases 1-6, 8-9) are fully resolution-agnostic.
  * Spatial map phases (Phase 7) compare two 1km grids — coordinate alignment
    via bilinear interpolation is still useful for minor grid offsets.
  * The old downscaling pipeline (NLDAS-2 premise) is invalid and archived.

Design notes
------------
* We do NOT regrid for domain-mean time series — spatial averaging already
  collapses both grids to the same scalar; regridding there is unnecessary.
* Regridding is applied ONLY when producing spatial maps or pixel-level stats.
* Method: scipy RegularGridInterpolator (bilinear) — robust on Windows, no
  compiled dependencies.
* All output figures and tables clearly label "AORC vs NLDAS-3 (0.01°)"
  so the resolution context is transparent.
"""

from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ── NLDAS-3 target grid ────────────────────────────────────────────────────────
# NLDAS-3 Southern US domain: 0.01° spacing (~1 km).
# NOTE: No module-level constant arrays — build_target_grid() reads actual
# file coordinates or uses 0.01° defaults.


def build_target_grid(
    lat_min: float = 25.0,
    lat_max: float = 38.0,
    lon_min: float = -110.0,
    lon_max: float = -79.0,
    dlat: float = 0.01,
    dlon: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats_1d, lons_1d) for the NLDAS-3 target grid (0.01° / ~1 km)."""
    lats = np.arange(lat_min + dlat / 2, lat_max, dlat)
    lons = np.arange(lon_min + dlon / 2, lon_max, dlon)
    return lats, lons


def bilinear_regrid(
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    src_data: np.ndarray,
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
    method: str = 'linear',
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Bilinear interpolation from source grid to target grid.

    Parameters
    ----------
    src_lats, src_lons : 1-D arrays of source grid coordinates (must be monotonic)
    src_data           : 2-D array (lat × lon) on source grid
    tgt_lats, tgt_lons : 1-D arrays of target grid coordinates
    method             : 'linear' (bilinear) or 'nearest'
    fill_value         : value used for out-of-bounds points

    Returns
    -------
    2-D array (len(tgt_lats) × len(tgt_lons)) on target grid
    """
    # Ensure monotonically increasing lats
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        src_data = src_data[::-1, :]

    interp = RegularGridInterpolator(
        (src_lats, src_lons),
        src_data,
        method=method,
        bounds_error=False,
        fill_value=fill_value,
    )
    tgt_lon_2d, tgt_lat_2d = np.meshgrid(tgt_lons, tgt_lats)
    pts = np.column_stack([tgt_lat_2d.ravel(), tgt_lon_2d.ravel()])
    return interp(pts).reshape(len(tgt_lats), len(tgt_lons))


def regrid_aorc_to_nldas(
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    src_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: aligns an AORC field to the NLDAS-3 0.01° grid.

    Since NLDAS-3 and AORC are both ~1 km resolution, this performs a
    coordinate-alignment interpolation rather than resolution downsampling.

    Returns
    -------
    (nldas3_lats, nldas3_lons, regridded_data)
    """
    tgt_lats, tgt_lons = build_target_grid()
    regridded = bilinear_regrid(src_lats, src_lons, src_data, tgt_lats, tgt_lons)
    return tgt_lats, tgt_lons, regridded


def spatial_bias_map(
    ref_lats: np.ndarray,
    ref_lons: np.ndarray,
    ref_stack: np.ndarray,     # shape (T, lat, lon)
    eval_stack: np.ndarray,    # shape (T, lat, lon) — already on ref grid
) -> dict[str, np.ndarray]:
    """
    Compute pixel-level bias metrics from two 3-D stacks (time × lat × lon).
    Both stacks must be on the same grid.

    Returns dict with 2-D arrays: bias, rmse, r, pbias.
    """
    n = min(ref_stack.shape[0], eval_stack.shape[0])
    r_s = ref_stack[:n]
    e_s = eval_stack[:n]

    bias  = np.nanmean(e_s - r_s, axis=0)
    rmse  = np.sqrt(np.nanmean((e_s - r_s)**2, axis=0))
    pbias = np.where(
        np.abs(np.nanmean(r_s, axis=0)) > 0,
        bias / np.abs(np.nanmean(r_s, axis=0)) * 100,
        np.nan,
    )

    # Pixel-wise Pearson r
    r_map = np.full(bias.shape, np.nan)
    for i in range(bias.shape[0]):
        for j in range(bias.shape[1]):
            rv = r_s[:, i, j]
            ev = e_s[:, i, j]
            mask = np.isfinite(rv) & np.isfinite(ev)
            if mask.sum() > 3:
                r_map[i, j] = float(np.corrcoef(rv[mask], ev[mask])[0, 1])

    return dict(bias=bias, rmse=rmse, r=r_map, pbias=pbias,
                lats=ref_lats, lons=ref_lons)


def regrid_nldas_to_aorc(
    nldas_lats: np.ndarray,
    nldas_lons: np.ndarray,
    nldas_data: np.ndarray,
    aorc_lats: np.ndarray,
    aorc_lons: np.ndarray,
) -> np.ndarray:
    """
    Bilinear alignment of a NLDAS-3 0.01° field onto the AORC 1km grid.

    Since NLDAS-3 and AORC are both ~1 km, this performs coordinate alignment
    rather than spatial downscaling. The downscaling pipeline that used this
    function under NLDAS-2 assumptions has been archived.

    Parameters
    ----------
    nldas_lats, nldas_lons : 1-D coordinate arrays for the NLDAS-3 field
    nldas_data             : 2-D array (lat × lon) on the NLDAS-3 grid
    aorc_lats, aorc_lons   : 1-D coordinate arrays for the AORC 1km target grid

    Returns
    -------
    2-D array (len(aorc_lats) × len(aorc_lons)) — NLDAS-3 aligned to AORC grid
    """
    return bilinear_regrid(nldas_lats, nldas_lons, nldas_data, aorc_lats, aorc_lons)


def monthly_delta_correction(
    aorc_clim: np.ndarray,    # (12, lat_nldas, lon_nldas) — monthly clim on NLDAS grid
    nldas_clim: np.ndarray,   # (12, lat_nldas, lon_nldas) — monthly clim on NLDAS grid
    nldas_lats: np.ndarray,
    nldas_lons: np.ndarray,
    aorc_lats: np.ndarray,
    aorc_lons: np.ndarray,
    mode: str = 'multiplicative',
) -> np.ndarray:
    """
    Compute and spatially disaggregate a monthly climatological correction factor.

    Steps
    -----
    1. At the coarse NLDAS grid: compute ratio (multiplicative) or difference
       (additive) between monthly AORC and NLDAS climatologies.
    2. Bilinearly interpolate the 12-month correction field to the 1km AORC grid.
    3. Return the correction field — caller applies it to disaggregated NLDAS.

    Parameters
    ----------
    aorc_clim, nldas_clim : (12 × nlat_nldas × nlon_nldas) climatological arrays
                            — monthly means for each calendar month
    mode : 'multiplicative' — correction = aorc / nldas (suitable for precip,
                              shortwave, longwave, pressure)
           'additive'       — correction = aorc - nldas (suitable for temp,
                              specific_humidity, wind)

    Returns
    -------
    (12, len(aorc_lats), len(aorc_lons)) — correction factors on the 1km grid
    """
    corrections = np.full((12, len(aorc_lats), len(aorc_lons)), np.nan)

    for m in range(12):
        ac = aorc_clim[m].astype(float)
        nc = nldas_clim[m].astype(float)

        if mode == 'multiplicative':
            # Avoid division by zero or near-zero
            corr_coarse = np.where(np.abs(nc) > 1e-6, ac / nc, 1.0)
            # Clip extreme ratios (outlier pixels at domain edges)
            corr_coarse = np.clip(corr_coarse, 0.1, 10.0)
        else:  # additive
            corr_coarse = ac - nc

        # Disaggregate correction to 1km grid
        corrections[m] = bilinear_regrid(
            nldas_lats, nldas_lons, corr_coarse, aorc_lats, aorc_lons)

    return corrections
