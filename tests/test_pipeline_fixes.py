"""
Regression tests for the review corrections (run: python tests/test_pipeline_fixes.py,
or pytest). Uses small synthetic NetCDF files that mimic the real AORC / NLDAS-3
layouts, including the traps that produced the original errors:

  * the two products cover different boxes, and NLDAS-3 carries values (zeros for
    precip, warm/humid air) over an "ocean" strip where AORC is NaN;
  * NLDAS-3 precip is a rate (kg m-2 s-1), AORC a daily accumulation (kg/m^2/day);
  * NLDAS-3 latitudes descend, AORC latitudes ascend.
"""

import os, sys, tempfile
import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from pipeline.units import precip_factor_to_mm_day, WIND_10M_TO_2M
from pipeline.et_calc import hs_et0, pm_et0, et0_sensitivity
from pipeline import stats as ST
from pipeline.gridpass import discover_pairs, CommonGrid, run_gridpass


# ── synthetic data ────────────────────────────────────────────────────────────
DATES = pd.date_range('2001-01-01', periods=60, freq='D')
OCEAN_LAT = 30.15          # cells south of this are "ocean" in AORC (NaN)


def _truth(date, lats, lons):
    """Smooth 'true' fields on any grid (land everywhere)."""
    la, lo = np.meshgrid(lats, lons, indexing='ij')
    d = date.dayofyear
    rng = np.random.default_rng(d)
    precip = np.where(rng.random(la.shape) < 0.4, 8.0 + (lo + 95.0), 0.0)   # mm/day
    return dict(
        precip=precip,
        temp=290.0 + 0.2 * (la - 30) + 3 * np.sin(2 * np.pi * d / 365),
        shortwave=200.0 + 10 * np.cos(2 * np.pi * d / 365) + 0 * la,
        longwave=330.0 + 0 * la, pressure=100000.0 + 0 * la,
        specific_humidity=0.010 + 0 * la,
        u_wind=np.where((la * 100).astype(int) % 2 == 0, 3.0, -3.0),   # opposite directions
        v_wind=4.0 + 0 * la,
    )


def make_files(tmp):
    adir = os.path.join(tmp, 'aorc'); ndir = os.path.join(tmp, 'nldas')
    os.makedirs(adir); os.makedirs(ndir)
    a_lats = np.arange(30.0, 30.6, 0.00833); a_lons = np.arange(-95.0, -94.4, 0.00833)
    n_lats = np.arange(30.6, 29.9, -0.01);   n_lons = np.arange(-95.05, -94.45, 0.01)   # descending, wider box
    for d in DATES:
        ta = _truth(d, a_lats, a_lons)
        ocean_a = (a_lats < OCEAN_LAT)[:, None] & np.ones(len(a_lons), bool)[None, :]
        ds_a = xr.Dataset(coords=dict(time=[d], latitude=a_lats, longitude=a_lons))
        for v, arr in ta.items():
            arr = np.where(ocean_a, np.nan, arr)
            ds_a[v] = (('time', 'latitude', 'longitude'), arr[None])
        ds_a['precip'].attrs['units'] = 'kg/m^2/day'
        ds_a.to_netcdf(os.path.join(adir, f'aorc_{d:%Y%m%d}_south_states.nc'))

        tn = _truth(d, n_lats, n_lons)
        ocean_n = (n_lats < OCEAN_LAT)[:, None] & np.ones(len(n_lons), bool)[None, :]
        # Known product differences on land: NLDAS 20% drier, +1 K, -10 W/m² SW.
        # Over ocean NLDAS has zeros for precip and warm humid air (the trap).
        rain = np.where(ocean_n, 0.0, tn['precip'] * 0.8) / 86400.0          # → kg m-2 s-1
        tair = np.where(ocean_n, 300.0, tn['temp'] + 1.0)
        qair = np.where(ocean_n, 0.020, tn['specific_humidity'])
        ds_n = xr.Dataset(coords=dict(time=[d], lat=n_lats, lon=n_lons))
        ds_n['Rainf'] = (('time', 'lat', 'lon'), rain[None]); ds_n['Rainf'].attrs['units'] = 'kg m-2 s-1'
        ds_n['Tair'] = (('time', 'lat', 'lon'), tair[None])
        ds_n['SWdown'] = (('time', 'lat', 'lon'), (tn['shortwave'] - 10)[None])
        ds_n['LWdown'] = (('time', 'lat', 'lon'), tn['longwave'][None])
        ds_n['PSurf'] = (('time', 'lat', 'lon'), tn['pressure'][None])
        ds_n['Qair'] = (('time', 'lat', 'lon'), qair[None])
        ds_n['Wind_E'] = (('time', 'lat', 'lon'), tn['u_wind'][None])
        ds_n['Wind_N'] = (('time', 'lat', 'lon'), tn['v_wind'][None])
        ds_n.to_netcdf(os.path.join(ndir, f'{d:%Y%m%d}_southern.nc'))
    return adir, ndir


# ── tests ─────────────────────────────────────────────────────────────────────
def test_units():
    assert precip_factor_to_mm_day('kg m-2 s-1') == 86400
    assert precip_factor_to_mm_day('kg/m^2/day') == 1
    assert precip_factor_to_mm_day('kg m-2') == 1          # was ×86400 in domains.py
    assert precip_factor_to_mm_day('mm/hr') == 24
    try:
        precip_factor_to_mm_day('furlongs'); raise AssertionError('should raise')
    except ValueError:
        pass


def test_hs_and_pm():
    assert abs(float(hs_et0(28, 33, 23, 31.0, 196)) - 5.52) < 0.05     # was ~13.5 without ×0.408
    base = pm_et0(301.15, 250, 420, 101000, 0.018, 2.0, 196, 31.0)
    more_lw = pm_et0(301.15, 250, 440, 101000, 0.018, 2.0, 196, 31.0)
    assert more_lw > base + 0.3                                          # LW was ignored before


def test_sensitivity_longwave_nonzero():
    idx = pd.date_range('2001-06-01', periods=30, freq='D')
    df = pd.DataFrame(index=idx)
    for p, lw in (('aorc', 330.0), ('nldas', 350.0)):
        df[f'temp_{p}'] = 300.0; df[f'shortwave_{p}'] = 220.0; df[f'longwave_{p}'] = lw
        df[f'pressure_{p}'] = 100500.0; df[f'specific_humidity_{p}'] = 0.015; df[f'wind_2m_{p}'] = 1.5
    s = et0_sensitivity(df)
    lw = s.set_index('variable').loc['longwave', 'mean_contribution_mm_d']
    assert lw > 0.2, lw
    assert abs(s.attrs['closure']['interaction_residual_mm_d']) < 1e-3


def test_pettitt():
    rng = np.random.default_rng(0)
    idx = pd.date_range('2001-01-31', periods=240, freq='ME')
    step = pd.Series(np.r_[rng.normal(0, 1, 120), rng.normal(1.2, 1, 120)], index=idx)
    noise = pd.Series(rng.normal(0, 1, 240), index=idx)
    r = ST.pettitt_test(step)
    assert r['p_value'] < 0.001 and 105 <= r['change_point_index'] <= 130
    assert ST.pettitt_test(noise)['p_value'] > 0.05


def test_gridpass_common_footprint():
    with tempfile.TemporaryDirectory() as tmp:
        adir, ndir = make_files(tmp)
        pairs, only_a, only_n = discover_pairs(adir, ndir, '2001-01-01', '2001-12-31')
        assert len(pairs) == len(DATES) and not only_a and not only_n
        samples = [pairs[d] for d in list(pairs)[::15]]
        grid = CommonGrid(samples, domains={'southern': None})
        # ocean strip and NLDAS-only margins excluded from the footprint
        assert grid.lats[grid.footprint.any(axis=1)].min() >= OCEAN_LAT - 0.01
        res = run_gridpass(pairs, grid, map_stride=1, et_stride=1, n_sample_pixels=50,
                           progress_every=0, log=lambda *_: None)
        df = res['domain_dfs']['southern']

        # Known differences recovered exactly (they were masked by ocean pixels before)
        ratio = df['precip_nldas'].sum() / df['precip_aorc'].sum()
        assert abs(ratio - 0.8) < 0.01, ratio
        dT = (df['temp_nldas'] - df['temp_aorc']).mean()
        assert abs(dT - 1.0) < 0.02, dT
        dSW = (df['shortwave_nldas'] - df['shortwave_aorc']).mean()
        assert abs(dSW + 10) < 0.05, dSW
        # humidity identical on land → no spurious difference from Gulf air
        assert abs((df['specific_humidity_nldas'] - df['specific_humidity_aorc']).mean()) < 1e-6

        # Wind: speed per pixel (5 m/s at 10 m), not speed of the mean vector (4 m/s)
        assert abs(df['wind_10m_aorc'].mean() - 5.0) < 0.01, df['wind_10m_aorc'].mean()
        assert abs(df['wind_2m_aorc'].mean() - 5.0 * WIND_10M_TO_2M) < 0.01

        # ET₀ per pixel, both products, and OAT columns present
        for c in ['et0_aorc', 'et0_nldas', 'et0_swap_temp', 'et0_swap_longwave']:
            assert c in df and df[c].notna().all(), c

        # Intermittency maps agree with a direct per-pixel calculation
        inter = res['intermittency']['aorc']
        samp = res['samples']['aorc']
        px = samp.columns[0]
        lat, lon = res['samples']['coords'].set_index('pixel').loc[px, ['lat', 'lon']]
        i = int(np.argmin(np.abs(res['map_lats'] - lat))); j = int(np.argmin(np.abs(res['map_lons'] - lon)))
        wet = (samp[px] >= 1.0).values
        assert abs(inter['wet_day_frac'][i, j] - wet.mean()) < 1e-9

        # Maps: r = 1 for temperature (NLDAS = AORC + 1 K), bias = +1
        m = res['maps']['temp']
        ok = np.isfinite(m['r'])
        assert ok.any() and np.nanmin(m['r'][ok]) > 0.999
        assert abs(np.nanmean(m['bias']) - 1.0) < 0.02


def test_gridpass_file_with_different_extent():
    """A processed file clipped to a smaller box is mapped by its own coordinates."""
    with tempfile.TemporaryDirectory() as tmp:
        adir, ndir = make_files(tmp)
        pairs, _, _ = discover_pairs(adir, ndir, '2001-01-01', '2001-12-31')
        grid = CommonGrid([pairs[d] for d in list(pairs)[::15]], domains={'southern': None})
        kw = dict(map_stride=1, et_stride=1, n_sample_pixels=50, progress_every=0, log=lambda *_: None)
        ref = run_gridpass(pairs, grid, **kw)

        day = sorted(pairs)[20]
        fp = pairs[day][1]
        with xr.open_dataset(fp) as ds:
            cropped = ds.isel(lat=slice(5, -12), lon=slice(3, -7)).load()   # different extent
        os.remove(fp); cropped.to_netcdf(fp)
        grid2 = CommonGrid([pairs[d] for d in list(pairs)[::15]], domains={'southern': None})
        res = run_gridpass(pairs, grid2, **kw)

        assert len(res['grid_summary']['file_grids']) == 3, res['grid_summary']['file_grids']
        a = ref['domain_dfs']['southern'].drop(index=day)
        b = res['domain_dfs']['southern'].drop(index=day)
        pd.testing.assert_frame_equal(a, b)                                  # other days untouched
        r = res['domain_dfs']['southern'].loc[day]
        assert r['n_valid_pixels'] < ref['domain_dfs']['southern'].loc[day, 'n_valid_pixels']
        assert abs(r['temp_nldas'] - r['temp_aorc'] - 1.0) < 0.02            # still paired correctly


def test_gridpass_resume_and_flaky_reads():
    """Interrupted + resumed run == uninterrupted run; a cached bad read is recovered by a fresh read."""
    import pipeline.gridpass as GPm
    with tempfile.TemporaryDirectory() as tmp:
        adir, ndir = make_files(tmp)
        pairs, _, _ = discover_pairs(adir, ndir, '2001-01-01', '2001-12-31')
        grid = CommonGrid([pairs[d] for d in list(pairs)[::15]], domains={'southern': None})
        kw = dict(map_stride=1, et_stride=1, n_sample_pixels=50, progress_every=0, log=lambda *_: None)
        ref = run_gridpass(pairs, grid, **kw)

        # flaky day: ordinary reads of this file fail, fresh (uncached) reads succeed
        flaky = sorted(pairs)[7]
        orig = GPm.read_product_day

        def flaky_read(fp, raw_map, vars_, fresh=False):
            if fp == pairs[flaky][0] and not fresh:
                raise OSError('NetCDF: HDF error (simulated)')
            return orig(fp, raw_map, vars_, fresh=fresh)

        class Stop(BaseException):           # like KeyboardInterrupt: not caught by retry/skip
            pass
        calls = {'n': 0}

        def crashing_read(fp, raw_map, vars_, fresh=False):
            calls['n'] += 1
            if calls['n'] > 2 * 40:          # die partway through (after ~40 days)
                raise Stop()
            return flaky_read(fp, raw_map, vars_, fresh)

        ck = os.path.join(tmp, 'ck.pkl')
        GPm.read_product_day = crashing_read
        try:
            try:
                run_gridpass(pairs, grid, checkpoint_path=ck, checkpoint_every=10, **kw)
                raise AssertionError('expected the simulated crash')
            except Stop:
                pass
            assert os.path.exists(ck)
            GPm.read_product_day = flaky_read
            res = run_gridpass(pairs, grid, checkpoint_path=ck, checkpoint_every=10, **kw)
        finally:
            GPm.read_product_day = orig

        gs = res['grid_summary']
        assert not gs['skipped_days'], gs['skipped_days']
        assert [r['date'] for r in gs['recovered_days']] == [str(flaky.date())], gs['recovered_days']
        a, b = ref['domain_dfs']['southern'], res['domain_dfs']['southern']
        pd.testing.assert_frame_equal(a, b)
        for v in ref['maps']:
            for st in ref['maps'][v]:
                assert np.array_equal(ref['maps'][v][st], res['maps'][v][st], equal_nan=True), (v, st)
        for p in ('aorc', 'nldas'):
            for st in ref['intermittency'][p]:
                assert np.array_equal(ref['intermittency'][p][st], res['intermittency'][p][st], equal_nan=True)
            pd.testing.assert_frame_equal(ref['samples'][p], res['samples'][p])


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); print(f'PASS  {name}')
