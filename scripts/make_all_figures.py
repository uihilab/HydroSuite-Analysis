"""
make_all_figures.py — Build every figure from a period's result tables
=================================================================================
Runs the existing figure scripts (make_clean_figures, regen_phase7, regen_et0_figures,
regen_et07_figures) against the tables written by run_pipeline.py and
run_et_station_comparison.py, and saves the figures to one folder.

  gridded tables   results/<period>/tables
  station tables   results/et0_validation/<period>/tables
  et07 table       results/comparisons/<period>/tables
  output           results/<period>/final_figures   (default)

    py -3.11 scripts/make_all_figures.py
    py -3.11 scripts/make_all_figures.py --period full_2001_2020 --out my_figures
"""

import os, sys, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_config import EXT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='full_2001_2020')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    tbl = os.path.join(EXT.RESULTS_ROOT, a.period, 'tables')
    st_tbl = os.path.join(EXT.RESULTS_ROOT, 'et0_validation', a.period, 'tables')
    out = a.out or os.path.join(EXT.RESULTS_ROOT, a.period, 'final_figures')
    os.makedirs(out, exist_ok=True)
    print(f'tables  : {tbl}\nstations: {st_tbl}\nfigures : {out}\n')
    done, failed = [], []

    def run(label, fn, *args):
        try:
            fn(*args); done.append(label)
        except Exception:
            failed.append(label)
            print(f'  FAILED {label}\n{traceback.format_exc()}')

    import make_clean_figures as MF
    MF.TBL_DIR, MF.FIG_DIR = tbl, out
    for name in [n for n in dir(MF) if n.startswith('fig_')]:
        run(f'make_clean_figures.{name}', getattr(MF, name))

    import regen_phase7 as R7
    R7.FIG_DIR = out
    for var in ('precip', 'temp', 'shortwave'):
        def phase7(v=var):
            res = R7.compute_and_cache(v, period=a.period)
            if res is None:
                raise RuntimeError(f'no gridpass maps for {v}')
            R7.plot_gulf(v, *res)
        run(f'regen_phase7.{var}', phase7)

    import regen_pipeline_figures as RP
    RP.FIG_DIR = out
    cache = os.path.join(EXT.RESULTS_ROOT, a.period, 'gridpass_v2', 'domain_southern.parquet')
    if os.path.exists(cache):
        import pandas as pd
        dfc = pd.read_parquet(cache)
        run('regen_pipeline_figures.phase2_error_pdfs', RP.fig_phase2_error_pdfs, dfc)
        run('regen_pipeline_figures.phase6_rolling_anomalies', RP.fig_rolling_anomalies, dfc)
    else:
        failed.append(f'regen_pipeline_figures (missing {cache})')

    def copy_pipeline_fig(name):             # drawn by run_pipeline.py itself
        import shutil
        src = os.path.join(EXT.RESULTS_ROOT, a.period, 'figures', name)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        shutil.copy2(src, os.path.join(out, name))
    run('run_pipeline.phase5_circular_seasonal', copy_pipeline_fig, 'phase5_circular_seasonal.png')

    def need(path):                      # the ET scripts only print SKIP when input is missing
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    import regen_et0_figures as RE
    RE.TBL_DIR, RE.FIG_DIR = st_tbl, out
    st_csv = os.path.join(st_tbl, 'et0_station_metrics.csv')
    run('regen_et0_figures.et01', lambda: (need(st_csv), RE.fig_et01_bias_map()))
    run('regen_et0_figures.et02', lambda: (need(st_csv), RE.fig_et02_taylor_all()))

    import regen_et07_figures as R07
    R07.FIG_DIR, R07.ET07_PERIOD = out, a.period
    et07_csv = os.path.join(EXT.RESULTS_ROOT, 'comparisons', a.period, 'tables', 'et07_method_comparison.csv')
    run('regen_et07_figures.et07', lambda: (need(et07_csv), R07.fix_et07()))

    print(f'\n{len(done)} figure step(s) done, {len(failed)} failed' + (f': {failed}' if failed else ''))
    print(f'Figures in {out}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
