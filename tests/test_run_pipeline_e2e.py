"""
End-to-end smoke test on three years of synthetic files laid out exactly as the
pipeline expects under HYDRO_DATA_DIR:

  * run_pipeline.run_period — every phase, no phase may log an error, and the tables
    the figure scripts read must be written; a second call must reuse the gridpass cache;
  * run_et_station_comparison.run — synthetic GHCN stations (one inland, one on a
    water cell, one off the grid): HS at the station recovers the truth, the water
    and off-grid stations are dropped, and the et07 table has the expected columns.

    py -3.11 tests/test_run_pipeline_e2e.py
"""

import os, sys, json, tempfile, importlib
import numpy as np
import pandas as pd
import netCDF4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_pipeline_fixes import _truth, OCEAN_LAT

DATES = pd.date_range('2001-01-01', '2003-12-31', freq='D')

EXPECTED_TABLES = [
    'bias_metrics.csv', 'taylor_stats.csv', 'et0_metrics.csv', 'et0_sensitivity.csv',
    'et0_sensitivity_closure.csv', 'return_period_pixel_summary.csv',
    'bias_correction_structure_summary.csv', 'spatial_coherence_summary.csv',
    'scale_sensitivity.csv', 'intermittency_comparison.csv',
]


def _write_nc(path, lat_name, lon_name, lats, lons, fields: dict, units: dict):
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('time', 1); ds.createDimension(lat_name, len(lats)); ds.createDimension(lon_name, len(lons))
        ds.createVariable(lat_name, 'f8', (lat_name,))[:] = lats
        ds.createVariable(lon_name, 'f8', (lon_name,))[:] = lons
        for k, a in fields.items():
            v = ds.createVariable(k, 'f4', ('time', lat_name, lon_name), fill_value=np.float32(np.nan))
            v[:] = np.asarray(a, 'f4')[None]
            if k in units:
                v.units = units[k]


def make_layout(root):
    adir = os.path.join(root, 'aorc', 'processed')
    ndir = os.path.join(root, 'data', 'processed', 'nldas_southern')
    os.makedirs(adir); os.makedirs(ndir)
    a_lats = np.arange(30.0, 30.3, 0.00833); a_lons = np.arange(-95.0, -94.7, 0.00833)
    n_lats = np.arange(30.3, 29.95, -0.01);  n_lons = np.arange(-95.02, -94.68, 0.01)
    ocean_a = (a_lats < OCEAN_LAT)[:, None] & np.ones(len(a_lons), bool)[None, :]
    ocean_n = (n_lats < OCEAN_LAT)[:, None] & np.ones(len(n_lons), bool)[None, :]
    for d in DATES:
        ta = _truth(d, a_lats, a_lons)
        _write_nc(os.path.join(adir, f'aorc_{d:%Y%m%d}_south_states.nc'), 'latitude', 'longitude',
                  a_lats, a_lons, {k: np.where(ocean_a, np.nan, v) for k, v in ta.items()},
                  {'precip': 'kg/m^2/day'})
        tn = _truth(d, n_lats, n_lons)
        _write_nc(os.path.join(ndir, f'{d:%Y%m%d}_southern.nc'), 'lat', 'lon', n_lats, n_lons, {
            'Rainf': np.where(ocean_n, 0.0, tn['precip'] * 0.8) / 86400.0,
            'Tair': np.where(ocean_n, 300.0, tn['temp'] + 1.0),
            'SWdown': tn['shortwave'] - 10, 'LWdown': tn['longwave'] + 5,
            'PSurf': tn['pressure'], 'Qair': tn['specific_humidity'],
            'Wind_E': tn['u_wind'], 'Wind_N': tn['v_wind']}, {'Rainf': 'kg m-2 s-1'})

    sub = os.path.join(root, 'sub.geojson')            # small sub-domain inside the land area
    ring = [[-94.95, 30.18], [-94.80, 30.18], [-94.80, 30.28], [-94.95, 30.28], [-94.95, 30.18]]
    with open(sub, 'w') as fh:
        json.dump({'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Polygon', 'coordinates': [ring]}}]}, fh)

    # GHCN: Tmean equal to the AORC truth at the station, DTR 10 °C
    gdir = os.path.join(root, 'ghcn'); os.makedirs(os.path.join(gdir, 'daily'))
    stations = [('USC_LAND', 30.20, -94.85, 'TX'), ('USC_WATER', 30.03, -94.85, 'TX'),
                ('USC_FAR', 32.00, -92.00, 'LA')]
    for sid, lat, lon, st in stations:
        tm = np.array([float(_truth(d, np.array([lat]), np.array([lon]))['temp'][0, 0]) for d in DATES]) - 273.15
        pd.DataFrame({'TMAX': tm + 5, 'TMIN': tm - 5}, index=DATES).to_parquet(
            os.path.join(gdir, 'daily', f'{sid}.parquet'))
    pd.DataFrame([dict(station_id=s, lat=la, lon=lo, state=st, name=s, elev=10.0, n_years=20)
                  for s, la, lo, st in stations]).to_csv(os.path.join(gdir, 'inventory.csv'), index=False)
    return sub


def _configure(tmp, sub):
    os.environ['HYDRO_DATA_DIR'] = tmp
    os.environ['HYDRO_GHCN_DIR'] = os.path.join(tmp, 'ghcn')
    import ext_config; importlib.reload(ext_config)
    EXT = ext_config.EXT
    EXT.SOUTHERN_BBOX = (-95.1, 29.9, -94.6, 30.4)
    EXT.DOMAINS = {'southern': None, 'subdomain': sub}
    EXT.OPTIONAL_DOMAINS = {}
    EXT.N_SAMPLE_PIXELS = 40
    return EXT


def test_e2e():
    with tempfile.TemporaryDirectory() as tmp:
        sub = make_layout(tmp)
        EXT = _configure(tmp, sub)

        # ── gridded pipeline ──
        import run_pipeline as RP; importlib.reload(RP)
        RP.EXT = EXT
        errors = []
        orig_log = RP.log
        RP.log = lambda m, level='INFO': (errors.append(m) if level == 'ERROR' else None, orig_log(m, level))
        RP.run_period._rebuild_cache = True
        RP.run_period('2001-01-01', '2003-12-31', 'e2e')
        assert not errors, '\n\n'.join(errors)

        tdir = os.path.join(tmp, 'results', 'e2e', 'tables')
        missing = [t for t in EXPECTED_TABLES if not os.path.exists(os.path.join(tdir, t))]
        assert not missing, f'missing tables: {missing}\npresent: {sorted(os.listdir(tdir))}'
        print(pd.read_csv(os.path.join(tdir, 'bias_metrics.csv')).to_string())
        scale = pd.read_csv(os.path.join(tdir, 'scale_sensitivity.csv'))
        assert set(scale['domain']) == {'southern', 'subdomain'}
        sens = pd.read_csv(os.path.join(tdir, 'et0_sensitivity.csv')).set_index('variable')
        assert sens.loc['longwave', 'mean_contribution_mm_d'] > 0     # +5 W/m² LW in NLDAS
        gpdir = os.path.join(tmp, 'results', 'e2e', 'gridpass_v2')
        assert os.path.exists(os.path.join(gpdir, 'unpaired_dates.json'))
        assert json.load(open(os.path.join(gpdir, 'grid_summary.json')))['domain_files']['subdomain'] == 'sub.geojson'

        RP.run_period._rebuild_cache = False                          # cache reuse
        RP.run_period('2001-01-01', '2003-12-31', 'e2e', phases={8})
        assert not errors, '\n\n'.join(errors)

        # ── publication figures from the new tables ──
        import make_clean_figures as MF; importlib.reload(MF)
        MF.TBL_DIR = tdir
        MF.FIG_DIR = os.path.join(tmp, 'figs'); os.makedirs(MF.FIG_DIR)
        failed = []
        for name in [n for n in dir(MF) if n.startswith('fig_')]:
            try:
                getattr(MF, name)()
            except Exception as e:
                failed.append(f'{name}: {type(e).__name__}: {e}')
        assert not failed, '\n'.join(failed)
        import regen_phase7 as R7; importlib.reload(R7)
        R7.FIG_DIR = MF.FIG_DIR
        res7 = R7.compute_and_cache('precip', period='e2e')
        assert res7 is not None
        R7.plot_gulf('precip', *res7)

        # ── station ET₀ ──
        import run_et_station_comparison as SC; importlib.reload(SC)
        SC.EXT = EXT
        SC.run('2001-01-01', '2003-12-31', 'e2e')
        sdir = os.path.join(tmp, 'results', 'et0_validation', 'e2e', 'tables')
        met = pd.read_csv(os.path.join(sdir, 'et0_station_metrics.csv'))
        print(met[['station_id', 'product', 'n', 'bias', 'rmse', 'r', 'obs_mean', 'sim_mean']].to_string())
        assert set(met['station_id']) == {'USC_LAND'}, met['station_id'].tolist()
        a = met.set_index('product').loc['AORC']
        assert abs(a['bias']) < 0.5 and a['r'] > 0.999, a           # mm/month; same Tmean and DTR
        assert 60 < a['obs_mean'] < 160, a['obs_mean']               # plausible HS, not ×2.45
        n = met.set_index('product').loc['NLDAS']
        assert n['bias'] > a['bias']                                  # NLDAS +1 K → higher HS
        sel = json.load(open(os.path.join(sdir, 'station_selection.json')))
        assert sel['used_common_to_both'] == 1 and sel['cell_invalid_or_off_grid']['AORC'] == 2, sel
        assert sel['usable_per_product']['NLDAS'] == 2      # the water cell is valid in NLDAS-3 only
        et07 = pd.read_csv(os.path.join(tmp, 'results', 'comparisons', 'e2e', 'tables',
                                        'et07_method_comparison.csv'))
        assert len(et07) == 12 and et07[['ghcn_hs', 'aorc_hs', 'aorc_pm', 'nldas_pm']].notna().all().all()

        # ── every publication figure via the one-shot runner ──
        import subprocess
        env = dict(os.environ, HYDRO_DATA_DIR=tmp, PYTHONIOENCODING='utf-8')
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'make_all_figures.py'),
                            '--period', 'e2e'], env=env, capture_output=True, text=True)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        figs = os.listdir(os.path.join(tmp, 'results', 'e2e', 'final_figures'))
        for f in ['phase2_taylor.png', 'phase8_scale_sensitivity.png', 'phase7_spatial_coherence_temp.png',
                  'et01_bias_map.png', 'et07_method_gap_climatology.png']:
            assert f in figs, (f, sorted(figs))


if __name__ == '__main__':
    test_e2e(); print('PASS  test_e2e')
