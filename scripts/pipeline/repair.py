"""
repair.py — Rebuild damaged processed daily files from the raw archive
=====================================================================
Last resort after fresh re-reads (pipeline.freshread) have failed. Most "NetCDF: HDF
error" reads on unreliable storage are transient: the same file reads cleanly later.
A file is rebuilt only if a fresh, uncached read of it still fails. The rebuild
follows the original preprocessing exactly, is verified by a fresh read of every
variable, and the damaged file is moved to ``<dir>/_corrupt/`` (nothing is deleted).

  AORC   raw  aorc/raw/aorc_YYYYMMDD_south.zarr
         → box selection 24.5–37.5°N, 106.5–75.5°W, zlib level 4
           (identical to retrieval/aorc/download_aorc_20yr_external.clip_day)
  NLDAS  raw  nldas/NLDAS_FOR0010_D.AYYYYMMDD.030.beta.nc
         → processing/nldas/clip_nldas_batch.process_file (unchanged)
"""

from __future__ import annotations
import os, glob, shutil, importlib.util
import numpy as np
import pandas as pd

AORC_BOX = dict(lat_min=24.5, lat_max=37.5, lon_min=-106.5, lon_max=-75.5)
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_nldas_mod = None
_nldas_geoms = None


def verify(fp: str, fresh: bool = True) -> list[str]:
    """
    Read every ≥2-D variable; return a list of 'var: error' (empty = readable).
    ``fresh`` reads the bytes straight from the device, so a damaged cached read is not
    mistaken for a damaged file (see pipeline.freshread).
    """
    import netCDF4
    bad = []
    try:
        if fresh:
            from pipeline.freshread import read_unbuffered
            ds_ctx = netCDF4.Dataset(os.path.basename(fp), memory=read_unbuffered(fp))
        else:
            ds_ctx = netCDF4.Dataset(fp)
        with ds_ctx as ds:
            for k, v in ds.variables.items():
                if v.ndim < 2:
                    continue
                try:
                    v[:]
                except Exception as e:
                    bad.append(f'{k}: {e}')
    except Exception as e:
        bad.append(f'open: {e}')
    return bad


def _quarantine(fp: str) -> str:
    qdir = os.path.join(os.path.dirname(fp), '_corrupt')
    os.makedirs(qdir, exist_ok=True)
    dst = os.path.join(qdir, os.path.basename(fp))
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(fp, dst)
    return dst


def rebuild_aorc(processed_fp: str, data_root: str) -> bool:
    import xarray as xr
    day = os.path.basename(processed_fp).split('_')[1]           # aorc_YYYYMMDD_south_states.nc
    raw = os.path.join(data_root, 'aorc', 'raw', f'aorc_{day}_south.zarr')
    if not os.path.exists(raw):
        return False
    tmp = processed_fp + '.rebuild.tmp'
    with xr.open_zarr(raw) as ds:
        sub = ds.sel(latitude=slice(AORC_BOX['lat_min'], AORC_BOX['lat_max']),
                     longitude=slice(AORC_BOX['lon_min'], AORC_BOX['lon_max']))
        enc = {v: {'zlib': True, 'complevel': 4} for v in sub.data_vars}
        sub.load().to_netcdf(tmp, encoding=enc)
    if verify(tmp):
        os.remove(tmp)
        return False
    _quarantine(processed_fp)
    os.replace(tmp, processed_fp)
    return True


def _load_nldas_clipper():
    global _nldas_mod, _nldas_geoms
    if _nldas_mod is None:
        path = os.path.join(_SCRIPTS, 'processing', 'nldas', 'clip_nldas_batch.py')
        spec = importlib.util.spec_from_file_location('clip_nldas_batch', path)
        _nldas_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_nldas_mod)                         # paths read HYDRO_DATA_DIR
        _nldas_geoms = _nldas_mod.load_geometries()
    return _nldas_mod, _nldas_geoms


def rebuild_nldas(processed_fp: str, data_root: str) -> bool:
    day = os.path.basename(processed_fp).split('_')[0]           # YYYYMMDD_southern.nc
    raws = glob.glob(os.path.join(data_root, 'nldas', f'NLDAS_FOR0010_D.A{day}.*.nc'))
    if not raws:
        return False
    mod, geoms = _load_nldas_clipper()
    if os.path.abspath(mod.PROCESSED_SOUTH) != os.path.abspath(os.path.dirname(processed_fp)):
        raise RuntimeError(f'NLDAS clipper writes to {mod.PROCESSED_SOUTH}, expected '
                           f'{os.path.dirname(processed_fp)} — set HYDRO_DATA_DIR')
    q = _quarantine(processed_fp)
    ok = mod.process_file(raws[0], geoms) and os.path.exists(processed_fp) and not verify(processed_fp)
    if not ok:                                   # put the original back if the rebuild failed
        if os.path.exists(processed_fp):
            os.remove(processed_fp)
        shutil.move(q, processed_fp)
    return ok


def repair_pair(aorc_fp: str, nldas_fp: str, data_root: str, log=print) -> dict:
    """Verify both files of a day; rebuild whichever is corrupt. Returns a status record."""
    rec = {}
    for name, fp, fn in (('aorc', aorc_fp, rebuild_aorc), ('nldas', nldas_fp, rebuild_nldas)):
        bad = verify(fp)
        if not bad:
            continue
        try:
            ok = fn(fp, data_root)
        except Exception as e:
            ok = False; bad.append(f'rebuild: {e}')
        rec[name] = dict(file=os.path.basename(fp), errors=bad, rebuilt=bool(ok))
        log(f'    repair {os.path.basename(fp)} ({"; ".join(bad)[:120]}) → {"rebuilt" if ok else "FAILED"}')
    return rec
