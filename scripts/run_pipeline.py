"""
run_pipeline.py — HydroSuiteAnalysis Master Orchestrator
=========================================================
Runs all 10 analysis phases using the pipeline modules.
Reads the processed files under HYDRO_DATA_DIR and writes results to
HYDRO_DATA_DIR/results. Paths are defined in scripts/ext_config.py.

Usage
-----
  python scripts/run_pipeline.py --all              # full 2001-2020 + both decades + all annual
  python scripts/run_pipeline.py --full             # 2001-2020 single run
  python scripts/run_pipeline.py --decades          # both 2001-2010 and 2011-2020
  python scripts/run_pipeline.py --decade 1         # 2001-2010 only
  python scripts/run_pipeline.py --decade 2         # 2011-2020 only
  python scripts/run_pipeline.py --year 2005        # single year
  python scripts/run_pipeline.py --years 2001-2010  # annual for each year in range
  python scripts/run_pipeline.py --dry-run --full   # path/count check, no computation

Cache
-----
  After the first full load, the DataFrame is cached as a parquet file
  inside the results directory (df_cache.parquet). Any subsequent
  --phases run loads the parquet in seconds instead of re-reading
  all NetCDF files.

  To force a fresh load from NetCDF (e.g. after fixing source files):
    python scripts/run_pipeline.py --full --rebuild-cache
"""

import os, sys, argparse, time, json, traceback, warnings, re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, genextreme as gev

# Disable all titles for publication-ready figures
plt.rcParams.update({
    'axes.titlesize': 0,
    'axes.titlepad': 0,
    'figure.titlesize': 0
})

warnings.filterwarnings('ignore')

# ── Global plot style ───────────────────────────────────────────────────────
plt.rcParams.update({
    'axes.grid'      : True,
    'grid.alpha'     : 0.35,
    'grid.linestyle' : '--',
    'grid.linewidth' : 0.6,
    'grid.color'     : '#888888',
    'axes.axisbelow' : True,   # grid behind data
})

# ── Module paths ───────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

from ext_config import EXT

# Log lines contain non-ASCII (ET₀, °C, −); keep a redirected Windows console from crashing.
for _stream in (sys.stdout, sys.stderr):
    try: _stream.reconfigure(errors='replace')
    except (AttributeError, ValueError): pass
from pipeline.et_calc  import compute_et0_series, et0_sensitivity
from pipeline import stats as ST
from pipeline import gridpass as GP
from pipeline.extremes import gev_fit_lmom

# ── Config ─────────────────────────────────────────────────────────────────────
# Only used if per-pixel ET₀ columns are unavailable (fallback); per-pixel ET₀ from
# pipeline.gridpass uses each pixel's own latitude.
DOMAIN_CENTROID_LAT = 0.5 * (EXT.SOUTHERN_BBOX[1] + EXT.SOUTHERN_BBOX[3])

VAR_LABELS = {
    'precip': 'Precipitation',
    'temp': 'Temperature',
    'shortwave': 'Shortwave',
    'longwave': 'Longwave',
    'specific_humidity': 'Sp. Humidity',
    'wind_2m': '2-m Wind',
    'pressure': 'Pressure'
}

ALL_VARS = ['precip', 'temp', 'shortwave', 'longwave', 'specific_humidity', 'wind_2m', 'pressure']
try:
    from pipeline.fig_style import CA as _CA, CN as _CN, SEASON_COLORS as _SC, PALETTE as _PAL, VAR_COLORS as _VC
    COLORS = {'aorc': _CA, 'nldas': _CN}
except ImportError:
    COLORS = {'aorc': 'steelblue', 'nldas': 'tomato'}


# ── Logging ────────────────────────────────────────────────────────────────────
def log(msg, level='INFO'):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] [{level}] {msg}', flush=True)


# ── GeoJSON helper ─────────────────────────────────────────────────────────────
def draw_geojson(ax, path, color='black', lw=1.0, label=None):
    if not path or not os.path.exists(path): return
    with open(path) as fh: gj = json.load(fh)
    feats = gj.get('features', [gj])
    first = True
    for feat in feats:
        geom = feat.get('geometry', feat)
        polys = geom['coordinates']
        if geom['type'] == 'Polygon': polys = [polys]
        for poly in polys:
            for ring in poly:
                c = np.array(ring)
                ax.plot(c[:,0], c[:,1], color=color, lw=lw,
                        label=(label if first else None))
                first = False


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Climatological Characterization
# ══════════════════════════════════════════════════════════════════════════════
def phase1_climatology(df: pd.DataFrame, period: str):
    log('Phase 1: Climatological Characterization')
    records = []
    for var in ALL_VARS:
        for ds in ['aorc', 'nldas']:
            col = f'{var}_{ds}'
            if col not in df: continue
            s = df[col].dropna()
            records.append(dict(variable=var, dataset=ds.upper(),
                                n=len(s), mean=s.mean(), std=s.std(),
                                cv=s.std()/s.mean() if s.mean() else np.nan,
                                min=s.min(), max=s.max()))
    if records:
        pd.DataFrame(records).round(4).to_csv(EXT.tbl_path(period, 'climatology_stats.csv'), index=False)

    # Monthly climatology plots (one per variable)
    n_vars = len(ALL_VARS)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for idx, var in enumerate(ALL_VARS):
        ax = axes.flat[idx]
        x = np.arange(1, 13)
        for ds, color in COLORS.items():
            col = f'{var}_{ds}'
            if col not in df: continue
            mon = df[col].groupby(df.index.month).mean()
            ax.bar(x + (0.2 if ds == 'nldas' else -0.2), mon.reindex(range(1,13)).fillna(0),
                   0.4, color=color, alpha=0.8, label=ds.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=7)
        ax.set_ylabel(var, fontsize=8); ax.legend(fontsize=7)
    for ax in axes.flat[n_vars:]: ax.set_visible(False)
    # suptitle removed — panel titles are self-describing
    plt.tight_layout()
    fig.savefig(EXT.fig_path(period, 'phase1_monthly_climatology.png'), dpi=150)
    plt.close(fig)
    log('  Phase 1 done')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Inter-Dataset Bias Analysis + Taylor Diagram
# ══════════════════════════════════════════════════════════════════════════════
def phase2_bias(df: pd.DataFrame, period: str):
    log('Phase 2: Inter-Dataset Bias Analysis')
    seasons   = ST.seasonal_split(df)
    bias_rows = []

    # ── Per-variable metrics + error PDFs ──────────────────────────────────────
    n_vars = len(ALL_VARS)
    fig_pdf, axes_pdf = plt.subplots(3, 3, figsize=(12, 9))

    for idx, var in enumerate(ALL_VARS):
        ca, cn = f'{var}_aorc', f'{var}_nldas'
        if ca not in df or cn not in df: continue
        both = df[[ca, cn]].dropna()
        # Deseasonalized anomalies: raw daily r is dominated by the shared annual cycle
        anom = pd.DataFrame({'a': ST.deseasonalize(both[ca]), 'n': ST.deseasonalize(both[cn])})

        # Overall metrics
        m = ST.bias_metrics(both[ca].values, both[cn].values)
        m['r_anom'] = float(anom['a'].corr(anom['n']))
        bias_rows.append(dict(variable=var, season='All', **m))

        # Seasonal metrics
        for sname, sdf in seasons.items():
            sb = sdf[[ca, cn]].dropna() if ca in sdf and cn in sdf else pd.DataFrame()
            if not sb.empty:
                sm = ST.bias_metrics(sb[ca].values, sb[cn].values)
                sa = anom.loc[anom.index.intersection(sb.index)]
                sm['r_anom'] = float(sa['a'].corr(sa['n'])) if len(sa) > 30 else np.nan
                bias_rows.append(dict(variable=var, season=sname, **sm))

        # Error PDF
        ax = axes_pdf.flat[idx]
        errors = both[cn].values - both[ca].values
        ST.error_pdf(ax, errors, label=VAR_LABELS.get(var, var), color=_CA if '_CA' in dir() else 'purple')
        for sname, sdf in seasons.items():
            sb = sdf[[ca, cn]].dropna() if ca in sdf and cn in sdf else pd.DataFrame()
            if not sb.empty:
                se = sb[cn].values - sb[ca].values
                col = _SC.get(sname, 'grey') if '_SC' in dir() else {'DJF':'royalblue','MAM':'seagreen','JJA':'orange','SON':'brown'}.get(sname,'grey')
                ST.error_pdf(ax, se, label=sname, color=col)
        # User requested variable names as Y-axis labels instead of Titles!
        ax.set_ylabel(VAR_LABELS.get(var, var), fontsize=18, fontweight='bold')
        
        ax.tick_params(axis='y', labelsize=15)
        
        # Only show x-axis labels on bottom row plots (indices 4, 5, 6 for 7 variables in 3x3)
        if idx >= 4:
            ax.tick_params(axis='x', labelsize=15)
            ax.set_xlabel('Error (NLDAS − AORC)', fontsize=18)
        else:
            ax.tick_params(axis='x', labelbottom=False)
            ax.set_xlabel('')
        
        # Move specific convoluted legends to the left as requested
        locs = {
            'precip': 'upper left',
            'temp': 'upper left',
            'shortwave': 'upper left',
            'pressure': 'upper left'
        }
        loc = locs.get(var, 'upper right')
        
        # Add cleanly sized legend to EACH panel
        ax.legend(fontsize=10, loc=loc, framealpha=0.85, frameon=False)
            
    axes_pdf.flat[7].set_visible(False)
    axes_pdf.flat[8].set_visible(False)


    # suptitle removed — panel titles are self-describing
    plt.tight_layout()
    fig_pdf.savefig(EXT.fig_path(period, 'phase2_error_pdfs.png'), dpi=300)
    plt.close(fig_pdf)

    # Bias metrics CSV
    if bias_rows:
        pd.DataFrame(bias_rows).round(4).to_csv(
            EXT.tbl_path(period, 'bias_metrics.csv'), index=False)

    # ── Taylor diagram: each variable against ITS OWN AORC reference ───────────
    # The earlier version compared every variable's NLDAS series with the FIRST
    # variable's AORC series. Statistics use deseasonalized daily anomalies, so the
    # shared annual cycle does not inflate r; raw-series stats are kept in the table.
    td_rows, td_points = [], []
    for var in ALL_VARS:
        ca, cn = f'{var}_aorc', f'{var}_nldas'
        if ca not in df or cn not in df: continue
        both = df[[ca, cn]].dropna()
        if len(both) < 60: continue
        raw  = ST.taylor_stats(both[ca].values, both[cn].values)
        an   = ST.taylor_stats(ST.deseasonalize(both[ca]).values, ST.deseasonalize(both[cn]).values)
        td_rows.append(dict(variable=var, r_raw=raw['r'], std_ratio_raw=raw['std_ratio'],
                            crmse_norm_raw=raw['crmse_norm'], r_anom=an['r'],
                            std_ratio_anom=an['std_ratio'], crmse_norm_anom=an['crmse_norm']))
        td_points.append((VAR_LABELS.get(var, var), an['r'], an['std_ratio']))
    if td_rows:
        pd.DataFrame(td_rows).round(4).to_csv(EXT.tbl_path(period, 'taylor_stats.csv'), index=False)
    fig_td = plt.figure(figsize=(9, 9))
    ax_td  = fig_td.add_subplot(111, projection='polar')
    if td_points:
        ST.taylor_diagram_normalized(ax_td, td_points)
    fig_td.savefig(EXT.fig_path(period, 'phase2_taylor.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_td)

    log('  Phase 2 done')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Penman-Monteith ET₀ + Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
def phase3_et0(df: pd.DataFrame, period: str):
    log('Phase 3: Penman-Monteith ET₀ (per pixel, common footprint)')
    if 'et0_aorc' not in df or 'et0_nldas' not in df:
        # Fallback only: PM on domain-mean inputs is not the domain mean of PM
        log('  per-pixel ET₀ columns missing — using domain-mean inputs (approximate)', 'WARN')
        df['et0_aorc']  = compute_et0_series(df, 'aorc',  DOMAIN_CENTROID_LAT)
        df['et0_nldas'] = compute_et0_series(df, 'nldas', DOMAIN_CENTROID_LAT)

    both = df[['et0_aorc','et0_nldas']].dropna()
    if both.empty:
        log('  No ET₀ computed — check input variables', 'WARN'); return

    # Save time series
    both.to_csv(EXT.tbl_path(period, 'et0_timeseries.csv'))

    # Metrics
    m = ST.bias_metrics(both['et0_aorc'].values, both['et0_nldas'].values)
    m['r_anom'] = ST.anomaly_correlation(both['et0_aorc'], both['et0_nldas'])
    log(f'  ET₀ RMSE={m["rmse"]:.3f} mm/d  r={m["r"]:.3f}  PBias={m["pbias"]:.1f}%')
    pd.DataFrame([{'variable':'et0',**m}]).to_csv(
        EXT.tbl_path(period, 'et0_metrics.csv'), index=False)

    # OAT sensitivity (per-pixel swaps from gridpass when available) + closure
    sens = et0_sensitivity(df, ref='aorc', eval_='nldas', lat_deg=DOMAIN_CENTROID_LAT)
    sens.to_csv(EXT.tbl_path(period, 'et0_sensitivity.csv'), index=False)
    pd.DataFrame([sens.attrs.get('closure', {})]).to_csv(
        EXT.tbl_path(period, 'et0_sensitivity_closure.csv'), index=False)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # Time series
    ax1 = fig.add_subplot(gs[0, :2])
    mon = both.resample('ME').mean()
    ax1.plot(mon.index, mon['et0_aorc'],  label='AORC',  color='steelblue', lw=1.5)
    ax1.plot(mon.index, mon['et0_nldas'], label='NLDAS', color='tomato',    lw=1.5)
    ax1.set_ylabel('mm/day'); ax1.legend()
    ax1.tick_params(axis='x', rotation=35)

    # Seasonal climatology
    ax2 = fig.add_subplot(gs[0, 2])
    mon_mean = both.groupby(both.index.month).mean()
    x = np.arange(1, 13)
    ax2.bar(x-0.2, mon_mean['et0_aorc'],  0.4, color='steelblue', alpha=0.85, label='AORC')
    ax2.bar(x+0.2, mon_mean['et0_nldas'], 0.4, color='tomato',    alpha=0.85, label='NLDAS')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=8)
    ax2.set_ylabel('mm/day'); ax2.legend(fontsize=8)

    # Sensitivity bar chart
    ax3 = fig.add_subplot(gs[1, :2])
    if not sens.empty:
        colors_s = ['steelblue' if v > 0 else 'tomato' for v in sens['mean_contribution_mm_d']]
        ax3.barh(sens['variable'], sens['mean_contribution_mm_d'], color=colors_s, alpha=0.85)
        ax3.axvline(0, color='k', lw=0.8)
        ax3.set_xlabel('Contribution to ET₀ bias (mm/day)')
    for ax in [ax3]:
        ax.invert_yaxis()

    # Rad vs aero term comparison
    ax4 = fig.add_subplot(gs[1, 2])
    errors = both['et0_nldas'].values - both['et0_aorc'].values
    ST.error_pdf(ax4, errors, label='ET₀ error', color='purple')
    ax4.set_ylabel('Density')

    # suptitle removed — panel titles are self-describing
    plt.tight_layout()
    fig.savefig(EXT.fig_path(period, 'phase3_et0.png'), dpi=150)
    plt.close(fig)
    log('  Phase 3 done')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Precipitation Extremes
# ══════════════════════════════════════════════════════════════════════════════
def phase4_extremes(df: pd.DataFrame, period: str, gp: dict | None = None):
    """
    Domain-mean GEV (kept for continuity; it describes a spatial average and should
    not be read as point/basin extremes) plus pixel-scale GEV and POT/GPD from the
    pixels sampled by gridpass — the hydrologically interpretable version.
    """
    log('Phase 4: Precipitation Extremes')

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    seasons = ST.seasonal_split(df)
    records = []

    for pidx, (col, lbl, color) in enumerate([
        ('precip_aorc','AORC','steelblue'), ('precip_nldas','NLDAS','tomato')
    ]):
        if col not in df: continue
        ser = df[col].dropna()

        # Seasonal cycle
        ax1 = fig.add_subplot(gs[0, pidx])
        mon_mean = ser.groupby(ser.index.month).mean()
        ax1.bar(np.arange(1,13), mon_mean.reindex(range(1,13)).fillna(0),
                color=color, alpha=0.85)
        ax1.set_xticks(range(1,13))
        ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=7)
        ax1.set_ylabel('mm/day')

        # PDF
        ax2 = fig.add_subplot(gs[1, pidx])
        vals = ser.values
        vals_pos = vals[vals > 0.1]
        if len(vals_pos) > 10:
            kde = gaussian_kde(vals_pos, bw_method='scott')
            x   = np.linspace(0, np.percentile(vals_pos,99), 300)
            ax2.fill_between(x, kde(x), alpha=0.35, color=color)
            ax2.plot(x, kde(x), color=color, lw=1.5)
        ax2.set_xlabel('mm/day'); ax2.set_ylabel('Density')

        # GEV return period
        ann_max = ser.resample('YE').max().dropna()
        if len(ann_max) >= 3:
            c_, loc_, scale_ = gev_fit_lmom(ann_max.values)          # L-moments (see extremes.py)
            T   = np.array([2, 5, 10, 25, 50, 100])
            rp  = gev.ppf(1 - 1/T, c_, loc_, scale_)
            rec = {'dataset': lbl, 'basis': 'domain_mean', 'n_years': len(ann_max)}
            for t, v in zip(T, rp):
                rec[f'T{t}yr_mm'] = round(float(v), 2)
            # Bootstrap 90% CI (n=500)
            try:
                rng = np.random.default_rng(42)
                boot = np.full((500, len(T)), np.nan)
                data_v = ann_max.values
                for b in range(500):
                    samp = rng.choice(data_v, size=len(data_v), replace=True)
                    try:
                        bc, bl, bs = gev_fit_lmom(samp)
                        boot[b] = gev.ppf(1 - 1/T, bc, bl, bs)
                    except Exception:
                        pass
                for ti, t in enumerate(T):
                    rec[f'T{t}yr_ci_lo'] = round(float(np.nanpercentile(boot[:, ti], 5)), 2)
                    rec[f'T{t}yr_ci_hi'] = round(float(np.nanpercentile(boot[:, ti], 95)), 2)
            except Exception:
                pass
            records.append(rec)

    # Return period curves together
    ax3 = fig.add_subplot(gs[:, 2])
    gev_plotted = 0
    for col, lbl, color in [('precip_aorc','AORC','steelblue'),('precip_nldas','NLDAS','tomato')]:
        if col not in df: continue
        ann_max = df[col].dropna().resample('YE').max().dropna()
        if len(ann_max) < 3:
            log(f'  GEV skipped for {lbl}: only {len(ann_max)} year(s) of data (need >= 3)', 'WARN')
            continue
        c_, loc_, scale_ = gev_fit_lmom(ann_max.values)
        T_range = np.linspace(1.01, 120, 300)
        ax3.semilogx(T_range, gev.ppf(1-1/T_range, c_, loc_, scale_),
                     color=color, lw=2, label=f'{lbl} GEV (n={len(ann_max)}yr)')
        # Empirical plotting positions (Weibull)
        emp_T = (len(ann_max) + 1) / np.arange(1, len(ann_max) + 1)
        ax3.scatter(emp_T, sorted(ann_max.values, reverse=True),
                    color=color, s=25, zorder=5, alpha=0.75)
        gev_plotted += 1
    if gev_plotted == 0:
        ax3.text(0.5, 0.5, 'Insufficient years\nfor GEV fit (need >= 3)',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=10, color='grey')
    ax3.set_xlabel('Return period (yr)'); ax3.set_ylabel('Precip (mm/day)')
    ax3.legend()

    # suptitle removed — panel titles are self-describing
    plt.tight_layout()
    fig.savefig(EXT.fig_path(period, 'phase4_precip_extremes.png'), dpi=150)
    plt.close(fig)

    if records:
        pd.DataFrame(records).to_csv(EXT.tbl_path(period, 'return_period_values.csv'), index=False)

    # ── Pixel-scale extremes (GEV per sampled pixel, POT/GPD sensitivity) ──────
    if gp is not None and gp.get('samples') and not gp['samples']['aorc'].empty:
        from pipeline.extremes import pixel_gev_summary, pot_gpd, RETURN_PERIODS
        samples = gp['samples']
        per_px, summ = pixel_gev_summary(samples)
        per_px.round(3).to_csv(EXT.tbl_path(period, 'return_period_pixels.csv'), index=False)
        summ.to_csv(EXT.tbl_path(period, 'return_period_pixel_summary.csv'), index=False)

        pot_rows = []
        for lbl, col in [('AORC', 'precip_aorc'), ('NLDAS', 'precip_nldas')]:
            if col in df:
                pot_rows.append(dict(dataset=lbl, basis='domain_mean', **pot_gpd(df[col])))
        for p, lbl in [('aorc', 'AORC'), ('nldas', 'NLDAS')]:
            cols = samples[p].columns[:500]
            res = pd.DataFrame([r for c in cols if 'T2yr_mm' in (r := pot_gpd(samples[p][c]))])
            if not res.empty:
                pot_rows.append(dict(dataset=lbl, basis='pixel_median', n_pixels=len(res),
                                     **res.median(numeric_only=True).round(3).to_dict()))
        if pot_rows:
            pd.DataFrame(pot_rows).to_csv(EXT.tbl_path(period, 'pot_gpd.csv'), index=False)

        # Figure: distribution across pixels of the T-yr GEV quantiles
        fig, ax = plt.subplots(figsize=(10, 5))
        pos = np.arange(len(RETURN_PERIODS))
        for off, p, color, lbl in [(-0.18, 'aorc', COLORS['aorc'], 'AORC'),
                                   (0.18, 'nldas', COLORS['nldas'], 'NLDAS-3')]:
            data = [per_px[f'{p}_T{t}'].dropna().values for t in RETURN_PERIODS]
            bp = ax.boxplot(data, positions=pos + off, widths=0.32, patch_artist=True,
                            showfliers=False, medianprops=dict(color='k'))
            for b in bp['boxes']:
                b.set_facecolor(color); b.set_alpha(0.75)
            ax.plot([], [], 's', color=color, label=lbl)
        ax.set_xticks(pos); ax.set_xticklabels([f'{t}-yr' for t in RETURN_PERIODS])
        ax.set_ylabel('Pixel GEV quantile (mm/day)'); ax.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(EXT.fig_path(period, 'phase4_pixel_gev.png'), dpi=200)
        plt.close(fig)
        log(f'  Pixel GEV: {len(per_px)} pixels; POT rows: {len(pot_rows)}')
    log('  Phase 4 done')



# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Circular Seasonal Statistics
# ══════════════════════════════════════════════════════════════════════════════
def phase5_circular_seasonal(df: pd.DataFrame, period: str):
    log('Phase 5: Circular Seasonal Statistics')

    circ_stats = ST.circular_stats_by_variable(df, ALL_VARS + ['et0'])
    circ_stats.to_csv(EXT.tbl_path(period, 'circular_seasonal_stats.csv'), index=False)
    log(f'  Circular stats: {len(circ_stats)} rows')

    # ── Polar rose plots: one panel per variable ───────────────────────────────
    vars_to_plot = [v for v in ALL_VARS + ['et0'] if
                    any(f'{v}_{ds}' in df.columns for ds in ['aorc','nldas'])]
    n = len(vars_to_plot)
    ncols = 3; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows),
                             subplot_kw=dict(projection='polar'))
    axes = np.array(axes).flatten()

    for idx, var in enumerate(vars_to_plot):
        ax = axes[idx]
        for ds, color in COLORS.items():
            col = f'{var}_{ds}'
            if col in df:
                ST.polar_climatology_plot(ax, df, col, label=ds.upper(), color=color)
        ax.set_title(VAR_LABELS.get(var, var), fontsize=18, fontweight='bold', pad=10)
        ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.28, 1.1), frameon=False)

    for ax in axes[n:]: ax.set_visible(False)
    plt.tight_layout()
    fig.savefig(EXT.fig_path(period, 'phase5_circular_seasonal.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    # ── Mean Resultant Length comparison bar chart ─────────────────────────────
    if not circ_stats.empty and 'R' in circ_stats.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for ds, color in [('AORC', COLORS['aorc']),('NLDAS', COLORS['nldas'])]:
            sub = circ_stats[circ_stats['dataset']==ds]
            if sub.empty: continue
            x = np.arange(len(sub))
            ax2.bar(x + (0 if ds=='AORC' else 0.35), sub['R'].values,
                    0.35, label=ds, color=color, alpha=0.88,
                    edgecolor='black', linewidth=1.5)
            if ax2.get_xlim()[1] < len(sub):
                ax2.set_xticks(np.arange(len(sub)) + 0.175)
                ax2.set_xticklabels(sub['variable'].tolist(), rotation=35, ha='right')
        ax2.set_xlabel('Variable'); ax2.set_ylabel('Mean Resultant Length R\n(0=uniform, 1=concentrated)')
        ax2.axhline(0.5, color='k', linestyle='--', lw=0.8, alpha=0.5)
        ax2.legend(frameon=False)
        plt.tight_layout()
        fig2.savefig(EXT.fig_path(period, 'phase5_seasonal_concentration.png'), dpi=300,
                     facecolor='white', edgecolor='none')
        plt.close(fig2)

    log('  Phase 5 done')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Long-term Trends + Bias Correction
# ══════════════════════════════════════════════════════════════════════════════
def phase6_trends_biascorr(df: pd.DataFrame, period: str):
    log('Phase 6: Trends + Bias Correction')

    # ── Mann-Kendall + Pettitt ─────────────────────────────────────────────────
    trend_rows = []
    for var in ALL_VARS + ['et0']:
        for ds in ['aorc', 'nldas']:
            col = f'{var}_{ds}'
            if col not in df: continue
            monthly = df[col].resample('ME').mean().dropna()
            if len(monthly) < 12: continue
            mk = ST.mann_kendall(monthly)
            pt = ST.pettitt_test(monthly)
            trend_rows.append(dict(variable=var, dataset=ds.upper(),
                                   mk_trend=mk['trend'], mk_p=mk['p_value'],
                                   mk_slope_per_mo=mk['slope'],
                                   pettitt_p=pt['p_value'],
                                   pettitt_date=str(pt['change_point_date'])))

    if trend_rows:
        pd.DataFrame(trend_rows).to_csv(EXT.tbl_path(period, 'trend_tests.csv'), index=False)

    # ── STL decomposition (described in the methods; previously not implemented) ─
    from pipeline.structure import stl_decompose
    stl_rows, stl_method = [], None
    fig_stl, axes_stl = plt.subplots(len(ALL_VARS), 1, figsize=(12, 2.2 * len(ALL_VARS)), sharex=True)
    for ax, var in zip(np.atleast_1d(axes_stl), ALL_VARS):
        for ds, color in COLORS.items():
            col = f'{var}_{ds}'
            if col not in df: continue
            monthly = df[col].resample('ME').mean().dropna()
            if len(monthly) < 36: continue
            comp, stl_method = stl_decompose(monthly)
            ax.plot(comp.index, comp['trend'], color=color, lw=1.6, label=ds.upper())
            stl_rows.append(dict(variable=var, dataset=ds.upper(),
                                 seasonal_amplitude=float(comp['seasonal'].max() - comp['seasonal'].min()),
                                 trend_range=float(comp['trend'].max() - comp['trend'].min()),
                                 resid_std=float(comp['resid'].std()),
                                 var_explained_seasonal=(float(1 - (monthly - comp['seasonal']).var() / monthly.var())
                                                         if monthly.var() > 0 else np.nan)))
        ax.set_ylabel(VAR_LABELS.get(var, var), fontsize=8)
        ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    fig_stl.savefig(EXT.fig_path(period, 'phase6_stl_trends.png'), dpi=150)
    plt.close(fig_stl)
    if stl_rows:
        out = pd.DataFrame(stl_rows).round(5)
        out['method'] = stl_method
        out.to_csv(EXT.tbl_path(period, 'stl_components.csv'), index=False)

    # ── Rolling anomaly plot ───────────────────────────────────────────────────
    # Physical bounds for clipping obvious fill-value leakage
    _BOUNDS = {
        'precip': (0, 30), 'temp': (240, 330), 'shortwave': (0, 500),
        'longwave': (150, 550), 'pressure': (80000, 110000),
        'specific_humidity': (0, 0.04), 'wind_2m': (0, 25),
    }

    def _flag_artifact_years(mon_series, z_thresh=2.5):
        """
        NaN out months belonging to years whose annual mean deviates
        more than z_thresh standard deviations from the multi-year mean.
        Catches systematic inter-annual data artifacts (e.g. NLDAS pressure
        2013, AORC wind 2018-2020) that are within physical bounds but
        represent known processing discontinuities.
        """
        ann = mon_series.resample('YE').mean().dropna()
        if len(ann) < 5:
            return mon_series
        mu, sigma = ann.mean(), ann.std()
        if sigma == 0:
            return mon_series
        bad_years = ann.index.year[np.abs((ann - mu) / sigma) > z_thresh]
        if len(bad_years) == 0:
            return mon_series
        cleaned = mon_series.copy()
        for yr in bad_years:
            cleaned[cleaned.index.year == yr] = np.nan
        return cleaned

    # AORC wind_2m has a documented source-data discontinuity starting 2018
    # (systematic reduction in the raw NOAA AORC v1.1 product)
    _AORC_WIND_BREAK = pd.Timestamp('2018-01-01')

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    plot_vars = [v for v in ALL_VARS if any(f'{v}_{ds}' in df.columns for ds in ['aorc','nldas'])]
    for idx, var in enumerate(plot_vars[:8]):
        ax = axes.flat[idx]
        lo, hi = _BOUNDS.get(var, (-1e9, 1e9))
        for ds, color in COLORS.items():
            col = f'{var}_{ds}'
            if col not in df: continue
            mon = df[col].resample('ME').mean()
            # Step 1: physical bounds (fill-value leakage)
            mon = mon.where((mon >= lo) & (mon <= hi))
            # Step 2: inter-annual z-score filter (processing discontinuities)
            mon = _flag_artifact_years(mon)
            # Step 3: for AORC wind, mask post-2018 to avoid structural artifact
            if var == 'wind_2m' and ds == 'aorc':
                mon_pre  = mon[mon.index < _AORC_WIND_BREAK]
                mon_post = mon[mon.index >= _AORC_WIND_BREAK]
                clim_pre = mon_pre.groupby(mon_pre.index.month).transform('mean')
                anom_pre = (mon_pre - clim_pre).dropna()
                if len(anom_pre) >= 3:
                    roll_pre = anom_pre.rolling(12, min_periods=6).mean()
                    ax.plot(roll_pre.index, roll_pre.values, color=color, lw=1.4, label=ds.upper())
                # Annotate the discontinuity
                ax.axvline(_AORC_WIND_BREAK, color='red', lw=1.2, linestyle='--', alpha=0.7)
                ax.text(_AORC_WIND_BREAK, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.1,
                        ' AORC\n data\n break', color='red', fontsize=6, va='bottom')
                continue
            clim_aligned = mon.groupby(mon.index.month).transform('mean')
            anom = (mon - clim_aligned).dropna()
            if len(anom) < 3: continue
            roll = anom.rolling(12, min_periods=6).mean()
            ax.plot(roll.index, roll.values, color=color, lw=1.4, label=ds.upper())
            mk = ST.mann_kendall(anom)
            if mk.get('trend', 'no trend') != 'no trend':
                ax.set_facecolor('#fff8f0')
        ax.axhline(0, color='k', lw=0.6, linestyle=':')
        ax.set_ylabel(var, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', labelsize=7)
    # Custom 8th Panel for Legend and Explanation
    ax8 = axes.flat[7]
    ax8.axis('off')
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    aorc_line = mlines.Line2D([], [], color=COLORS['aorc'], lw=3, label='AORC (Reference)')
    nldas_line = mlines.Line2D([], [], color=COLORS['nldas'], lw=3, label='NLDAS-3 (Evaluated)')
    sig_patch = mpatches.Patch(color='#fff8f0', ec='gray', label='Statistically Significant Trend (p < 0.05)')
    break_line = mlines.Line2D([], [], color='red', lw=2, linestyle='--', label='AORC v1.1 Data Discontinuity')
    ax8.legend(handles=[aorc_line, nldas_line, sig_patch, break_line], loc='center', fontsize=20, frameon=True, framealpha=1, edgecolor='gray', borderpad=1.5, labelspacing=1.2)
    fig.supylabel('Anomaly', fontsize=24, fontweight='bold', x=0.01)
    plt.tight_layout()
    fig.savefig(EXT.fig_path(period, 'phase6_rolling_anomalies.png'), dpi=150)
    plt.close(fig)

    # ── Bias correction: QM + Linear on precipitation ─────────────────────────
    if 'precip_aorc' not in df or 'precip_nldas' not in df:
        log('  Skipping bias correction — no precip overlap', 'WARN'); return

    from scipy import stats as spstats
    both = df[['precip_aorc','precip_nldas']].dropna()
    if len(both) < 60:
        log('  Insufficient overlap for bias correction', 'WARN'); return

    aorc_vals  = both['precip_aorc'].values
    nldas_vals = both['precip_nldas'].values
    years      = both.index.year.unique()

    from pipeline.structure import structure_metrics
    sw_a = df['shortwave_aorc'] if 'shortwave_aorc' in df else None
    sw_n = df['shortwave_nldas'] if 'shortwave_nldas' in df else None
    struct_rows = []
    loyo_rows = []
    for yr in years:
        mask_test  = both.index.year == yr
        mask_train = ~mask_test
        if mask_train.sum() < 30: continue
        at, nt = aorc_vals[mask_train], nldas_vals[mask_train]
        av, nv = aorc_vals[mask_test],  nldas_vals[mask_test]

        # Quantile Mapping
        pct  = np.linspace(0, 100, 101)
        qa   = np.percentile(at, pct)
        qn   = np.percentile(nt, pct)
        qm_p = np.interp(nv, qn, qa)

        # Linear Regression (clipped at 0: negative precipitation is not physical)
        sl, ic, *_ = spstats.linregress(nt, at)
        lr_p = np.maximum(sl * nv + ic, 0.0)

        # Temporal structure of each version against the AORC test year
        idx_test = both.index[mask_test]
        ref_s = pd.Series(av, index=idx_test)
        for name, arr, sw in [('aorc', av, sw_a), ('raw', nv, sw_n), ('qm', qm_p, sw_n), ('lr', lr_p, sw_n)]:
            struct_rows.append(dict(year=int(yr), series=name,
                                    **structure_metrics(pd.Series(arr, index=idx_test), sw=sw,
                                                        event_ref=ref_s)))

        def _m(pred, obs):
            msk = np.isfinite(pred) & np.isfinite(obs)
            if msk.sum() < 2: return np.nan, np.nan, np.nan
            rm = float(np.sqrt(np.mean((pred[msk]-obs[msk])**2)))
            r_ = float(np.corrcoef(pred[msk], obs[msk])[0,1])
            pb = float((pred[msk].mean()-obs[msk].mean())/obs[msk].mean()*100) if obs[msk].mean() else np.nan
            return rm, r_, pb

        raw_m = _m(nv, av); qm_m = _m(qm_p, av); lr_m = _m(lr_p, av)
        loyo_rows.append(dict(year=yr,
            raw_rmse=raw_m[0], raw_r=raw_m[1], raw_pbias=raw_m[2],
            qm_rmse=qm_m[0],   qm_r=qm_m[1],   qm_pbias=qm_m[2],
            lr_rmse=lr_m[0],   lr_r=lr_m[1],   lr_pbias=lr_m[2]))

    if struct_rows:
        st = pd.DataFrame(struct_rows)
        st.round(4).to_csv(EXT.tbl_path(period, 'bias_correction_structure.csv'), index=False)
        summ = st.drop(columns='year').groupby('series').mean().round(4)
        summ.to_csv(EXT.tbl_path(period, 'bias_correction_structure_summary.csv'))
        log('  Structure (mean over left-out years):\n' + summ.to_string())

    if loyo_rows:
        res = pd.DataFrame(loyo_rows)
        res.to_csv(EXT.tbl_path(period, 'bias_correction_loyo.csv'), index=False)

        fig_bc, axes_bc = plt.subplots(1, 3, figsize=(14, 4))
        year_vals = res['year'].astype(int).values
        for ax, metric, lbl in zip(axes_bc,
                                   ['rmse', 'r', 'pbias'],
                                   ['RMSE (mm/day)', 'Pearson r', '% Bias']):
            for col, clr, nlbl in [
                (f'raw_{metric}','grey','Raw'),
                (f'qm_{metric}','steelblue','QM'),
                (f'lr_{metric}','tomato','Linear'),
            ]:
                if col in res:
                    ax.plot(year_vals, res[col].values, 'o-', color=clr, label=nlbl, markersize=5)
            ax.set_xlabel('Left-out year')
            ax.set_ylabel(lbl)
            ax.set_xticks(year_vals[::2])          # every other year for legibility
            ax.set_xticklabels(year_vals[::2], rotation=45, fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            if metric == 'pbias':
                ax.axhline(0, color='k', lw=0.8, linestyle=':')
                ax.axhline(10,  color='green', lw=0.6, linestyle='--', alpha=0.6)
                ax.axhline(-10, color='green', lw=0.6, linestyle='--', alpha=0.6)
        plt.tight_layout()
        fig_bc.savefig(EXT.fig_path(period, 'phase6_bias_correction.png'), dpi=150)
        plt.close(fig_bc)

    log('  Phase 6 done')


# ══════════════════════════════════════════════════════════════════════════════
# PERIOD RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def _load_or_run_gridpass(start: str, end: str, label: str):
    """
    Common-footprint pass (pipeline.gridpass), cached under results/<label>/gridpass_v2.
    The old df_cache.parquet came from the per-product loaders and is NOT reused.
    """
    out_dir = os.path.join(EXT.results_dir(label), 'gridpass_v2')
    if not getattr(run_period, '_rebuild_cache', False):
        gp = GP.load_gridpass(out_dir)
        if gp is not None:
            log(f'  gridpass cache loaded: {out_dir}')
            return gp
    pairs, only_a, only_n = GP.discover_pairs(EXT.AORC_DIR, EXT.NLDAS_DIR, start, end,
                                              EXT.AORC_PATTERN, EXT.NLDAS_PATTERN)
    log(f'  paired days: {len(pairs)} | AORC-only: {len(only_a)} | NLDAS-only: {len(only_n)}')
    if not pairs:
        return None
    dates = sorted(pairs)
    k = max(1, min(EXT.FOOTPRINT_SAMPLE_DAYS, len(dates)))
    sample = []
    for i in np.linspace(0, len(dates) - 1, k).astype(int):
        # take the first readable day at or after each evenly spaced position
        for d in dates[i:]:
            try:
                for fp, raw in zip(pairs[d], (GP.AORC_RAW, GP.NLDAS_RAW)):
                    GP.read_product_day(fp, raw, ['temp', 'precip', 'shortwave'])
                sample.append(pairs[d]); break
            except Exception as e:
                from pipeline.repair import repair_pair
                log(f'  footprint sample {d.date()} unreadable ({e}); repairing', 'WARN')
                repair_pair(*pairs[d], EXT.DATA_ROOT, log=log)
                try:
                    for fp, raw in zip(pairs[d], (GP.AORC_RAW, GP.NLDAS_RAW)):
                        GP.read_product_day(fp, raw, ['temp', 'precip', 'shortwave'])
                    sample.append(pairs[d]); break
                except Exception:
                    log(f'  footprint sample {d.date()} still unreadable, trying next day', 'WARN')
    for name, path in EXT.active_domains().items():
        log(f'  domain {name}: {os.path.basename(path) if path else "whole common footprint"}')
    grid = GP.CommonGrid(sample, bbox=EXT.SOUTHERN_BBOX, domains=EXT.active_domains(),
                         landmask_geojson=EXT.LANDMASK_GEOJSON)
    log(f'  common grid: {grid.summary()}')
    from pipeline.repair import repair_pair
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, 'checkpoint.pkl')
    if getattr(run_period, '_no_resume', False) and os.path.exists(ckpt):
        os.remove(ckpt)
    gp = GP.run_gridpass(pairs, grid, map_stride=EXT.MAP_STRIDE, et_stride=EXT.ET_STRIDE,
                         n_sample_pixels=EXT.N_SAMPLE_PIXELS, wet_threshold=EXT.WET_THRESHOLD_MM,
                         log=log, checkpoint_path=ckpt,
                         repair=lambda d, af, nf: repair_pair(af, nf, EXT.DATA_ROOT, log=log))
    if gp['grid_summary'].get('skipped_days'):
        log(f"  {len(gp['grid_summary']['skipped_days'])} unreadable day(s) skipped — "
            f"listed in {os.path.join(out_dir, 'grid_summary.json')} (skipped_days); "
            f"re-download/re-clip them", 'WARN')
    gp['grid_summary']['domain_files'] = {k: (os.path.basename(v) if v else None)
                                          for k, v in EXT.active_domains().items()}
    GP.save_gridpass(gp, out_dir)
    if os.path.exists(ckpt):
        os.remove(ckpt)                          # complete: the checkpoint is no longer needed
    if gp['grid_summary'].get('repaired_days'):
        log(f"  {len(gp['grid_summary']['repaired_days'])} corrupt day(s) rebuilt from raw "
            f"(originals moved to _corrupt/)")
    with open(os.path.join(out_dir, 'unpaired_dates.json'), 'w') as fh:
        json.dump({'aorc_only': [str(d.date()) for d in only_a],
                   'nldas_only': [str(d.date()) for d in only_n]}, fh, indent=1)
    return gp


def run_period(start: str, end: str, label: str,
               dry_run: bool = False,
               phases: set | None = None):
    """
    Run analysis phases for one time period.
    phases: set of ints (1-10). If None, runs all phases.
    """
    log(f'\n{"="*58}')
    log(f'  PERIOD: {label}  ({start}  ->  {end})')
    if phases:
        log(f'  PHASES: {sorted(phases)}')
    log(f'{"="*58}')

    _ = EXT.results_dir(label)   # create output dirs

    if dry_run:
        ao = EXT.discover_aorc(start, end)
        nl = EXT.discover_nldas(start, end)
        log(f'[DRY-RUN] AORC={len(ao)} | NLDAS={len(nl)} | results->{EXT.results_dir(label)}')
        return

    def _should(n): return phases is None or n in phases

    t0 = time.time()

    # One common-footprint gridded pass feeds every phase (cached per period).
    gp = _load_or_run_gridpass(start, end, label)
    df = gp['domain_dfs'].get('southern', pd.DataFrame()) if gp else pd.DataFrame()
    if df.empty:
        log('No paired AORC/NLDAS-3 data for this period', 'WARN')

    scalar_phases = [
        (1, phase1_climatology,       'climatology'),
        (2, phase2_bias,              'bias'),
        (3, phase3_et0,               'et0'),
        (4, phase4_extremes,          'extremes'),
        (5, phase5_circular_seasonal, 'circular_seasonal'),
        (6, phase6_trends_biascorr,   'trends_biascorr'),
    ]
    for n, phase_fn, name in scalar_phases:
        if not _should(n) or df.empty: continue
        try:
            if n == 4:
                phase_fn(df.copy(), label, gp)
            else:
                phase_fn(df.copy(), label)
        except Exception:
            log(f'  PHASE {n} ({name}) failed:\n{traceback.format_exc()}', 'ERROR')

    for n, fn, name, args in [
        (7,  phase7_spatial_coherence, 'spatial',       (gp, label)),
        (8,  phase8_scale_sensitivity, 'scale',         (gp, label)),
        (9,  phase9_intermittency,     'intermittency', (df.copy(), label, gp)),
        (10, phase10_spectral,         'spectral',      (df.copy(), label)),
    ]:
        if not _should(n): continue
        try: fn(*args)
        except Exception: log(f'  PHASE {n} ({name}) failed:\n{traceback.format_exc()}', 'ERROR')

    elapsed = time.time() - t0
    log(f'Period {label} done in {elapsed/60:.1f} min')





# ══════════════════════════════════════════════════════════════════════════════
# PERIOD DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
PERIODS = {
    'full'      : ('2001-01-01', '2020-12-31', 'full_2001_2020'),
    'decade1'   : ('2001-01-01', '2010-12-31', 'decade_2001_2010'),
    'decade2'   : ('2011-01-01', '2020-12-31', 'decade_2011_2020'),
}


def annual_periods(year_start: int, year_end: int):
    return [(f'{y}-01-01', f'{y}-12-31', f'annual/{y}') for y in range(year_start, year_end+1)]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Spatial Coherence Maps
# ══════════════════════════════════════════════════════════════════════════════
def phase7_spatial_coherence(gp: dict | None, period: str):
    """
    Full-period pixel maps (mean, bias, daily Pearson r) on the common footprint,
    accumulated over every paired day by gridpass. The earlier version used only the
    first 730 files ("cap at 2 years ... for prototype") and drew no correlation map.
    Bias is NLDAS-3 − AORC, consistent with all other error definitions.
    """
    log('Phase 7: Spatial Coherence Maps (full period, common footprint)')
    if not gp or not gp.get('maps'):
        log('  no gridpass maps available', 'WARN'); return
    lats, lons = gp['map_lats'], gp['map_lons']
    lon_g, lat_g = np.meshgrid(lons, lats)
    units = {'precip': 'mm/day', 'temp': '°C', 'shortwave': 'W/m²'}
    rows = []
    for var, st in gp['maps'].items():
        conv = (lambda a: a - 273.15) if var == 'temp' else (lambda a: a)
        mean_a, mean_n, bias, rmap = conv(st['mean_aorc']), conv(st['mean_nldas']), st['bias'], st['r']
        fin = np.concatenate([mean_a[np.isfinite(mean_a)], mean_n[np.isfinite(mean_n)]])
        vmin, vmax = (np.percentile(fin, 2), np.percentile(fin, 98)) if fin.size else (0, 1)
        fb = bias[np.isfinite(bias)]
        vb = float(np.percentile(np.abs(fb), 98)) if fb.size else 1.0
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        u = units.get(var, '')
        for ax, data, title, cmap, lo, hi in [
            (axes[0, 0], mean_a, f'AORC mean ({u})', 'viridis', vmin, vmax),
            (axes[0, 1], mean_n, f'NLDAS-3 mean ({u})', 'viridis', vmin, vmax),
            (axes[1, 0], bias,   f'Bias, NLDAS-3 − AORC ({u})', 'RdBu_r', -vb, vb),
            (axes[1, 1], rmap,   'Daily Pearson r', 'magma', 0, 1),
        ]:
            im = ax.pcolormesh(lon_g, lat_g, data, cmap=cmap, vmin=lo, vmax=hi, shading='auto')
            plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            ax.set_title(title, fontsize=13)
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            for path in EXT.active_domains().values():
                if path: draw_geojson(ax, path, color='k', lw=0.8)
        plt.tight_layout()
        fig.savefig(EXT.fig_path(period, f'phase7_spatial_coherence_{var}.png'), dpi=300,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        r = rmap[np.isfinite(rmap)]
        rows.append(dict(variable=var, n_pixels=int(r.size),
                         r_mean=float(r.mean()) if r.size else np.nan,
                         r_median=float(np.median(r)) if r.size else np.nan,
                         frac_r_gt_0_85=float((r > 0.85).mean()) if r.size else np.nan,
                         bias_mean=float(fb.mean()) if fb.size else np.nan,
                         bias_p05=float(np.percentile(fb, 5)) if fb.size else np.nan,
                         bias_p95=float(np.percentile(fb, 95)) if fb.size else np.nan))
    pd.DataFrame(rows).round(4).to_csv(EXT.tbl_path(period, 'spatial_coherence_summary.csv'), index=False)
    log('  Phase 7 done')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — Scale-Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════════════════
def phase8_scale_sensitivity(gp: dict | None, period: str):
    """
    Bias metrics for every configured domain on the common footprint. Replaces the
    domains.py path, which multiplied NLDAS precip by 86400 whenever the units string
    contained "kg" (scale_sensitivity.csv showed n = 29 days and PBIAS ≈ 10⁶ %).
    """
    log('Phase 8: Scale-Sensitivity Analysis (common footprint)')
    if not gp:
        log('  no gridpass results', 'WARN'); return
    vars8 = ['precip', 'temp', 'shortwave', 'longwave', 'specific_humidity', 'wind_2m', 'et0']
    rows = []
    for dom, df in gp['domain_dfs'].items():
        if df.empty: continue
        for var in vars8:
            ca, cn = f'{var}_aorc', f'{var}_nldas'
            if ca not in df or cn not in df: continue
            both = df[[ca, cn]].dropna()
            if len(both) < 30: continue
            m = ST.bias_metrics(both[ca].values, both[cn].values)
            m['r_anom'] = ST.anomaly_correlation(both[ca], both[cn])
            mean_a = float(both[ca].mean())
            m['rmse_norm'] = m['rmse'] / abs(mean_a) if mean_a else np.nan
            m['median_valid_pixels'] = float(df['n_valid_pixels'].median()) if 'n_valid_pixels' in df else np.nan
            rows.append(dict(domain=dom, variable=var, **m))
    if not rows:
        log('  no rows', 'WARN'); return
    res = pd.DataFrame(rows).round(4)
    res.to_csv(EXT.tbl_path(period, 'scale_sensitivity.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in [(axes[0], 'rmse_norm', 'RMSE / AORC domain mean'),
                               (axes[1], 'r_anom', 'Anomaly correlation r')]:
        for dom in res['domain'].unique():
            sub = res[res['domain'] == dom]
            ax.plot(sub['variable'], sub[metric], marker='o', lw=1.5, ms=7, label=dom)
        ax.set_ylabel(ylabel); ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=8, frameon=False); ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    fig.savefig(EXT.fig_path(period, 'phase8_scale_sensitivity.png'), dpi=150)
    plt.close(fig)
    log(f'  Phase 8 done ({len(rows)} domain×variable rows)')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9 — Intermittency Analysis
# ══════════════════════════════════════════════════════════════════════════════
def phase9_intermittency(df: pd.DataFrame, period: str, gp: dict | None = None):
    """
    Precipitation intermittency at PIXEL scale (median across common-footprint pixels).
    A domain-mean series over the whole south is "wet" almost every day, so its spell
    statistics are not hydrologically interpretable; that version is still written,
    labelled, to intermittency_comparison_domain_mean.csv.
    """
    log('Phase 9: Intermittency Analysis (pixel scale)')
    from pipeline.spectral import compare_intermittency
    from pipeline.extremes import pixel_intensity_summary, pixel_seasonal_intermittency
    thr = EXT.WET_THRESHOLD_MM

    if 'precip_aorc' in df and 'precip_nldas' in df:
        dm = compare_intermittency(df['precip_aorc'], df['precip_nldas'], thr)
        if not dm.empty:
            dm.assign(basis='domain_mean').to_csv(
                EXT.tbl_path(period, 'intermittency_comparison_domain_mean.csv'), index=False)

    if not gp or not gp.get('intermittency'):
        log('  no pixel intermittency available', 'WARN'); return
    ia, inn = gp['intermittency']['aorc'], gp['intermittency']['nldas']

    def med(a, st):
        # only cells with data: the maps cover the whole grid and cells outside the
        # footprint hold 0 for the spell maxima (not NaN), which pulled medians to 0
        v = np.asarray(a, float)[np.asarray(st['n_days']) > 0]; v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else np.nan

    inten = pixel_intensity_summary(gp['samples'], thr)
    metrics = [(k, med(ia[k], ia), med(inn[k], inn)) for k in
               ['wet_day_frac', 'wet_mean_mm', 'wet_spell_mean', 'wet_spell_max',
                'dry_spell_mean', 'dry_spell_max']]
    metrics.insert(1, ('dry_day_frac', 1 - metrics[0][1], 1 - metrics[0][2]))
    for k in ['P90_intensity_mm', 'frac_intense', 'total_mm']:
        metrics.append((k, float(inten.loc['aorc', k]), float(inten.loc['nldas', k])))
    rows = [dict(metric=k, aorc=round(a, 4), nldas=round(n, 4), diff=round(n - a, 4),
                 pct_diff=round((n - a) / abs(a) * 100, 2) if a else np.nan, basis='pixel_median')
            for k, a, n in metrics]
    pd.DataFrame(rows).to_csv(EXT.tbl_path(period, 'intermittency_comparison.csv'), index=False)
    for p, sdf in pixel_seasonal_intermittency(gp['samples'], thr).items():
        sdf.to_csv(EXT.tbl_path(period, f'intermittency_seasonal_{p}.csv'))

    lats, lons = gp['map_lats'], gp['map_lons']
    lon_g, lat_g = np.meshgrid(lons, lats)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    wa, wn = ia['wet_day_frac'], inn['wet_day_frac']
    top = float(np.nanpercentile(wa, 98)) if np.isfinite(wa).any() else 1.0
    for ax, data, title, cmap, lo, hi in [
        (axes[0, 0], wa, 'AORC wet-day fraction', 'YlGnBu', 0, top),
        (axes[0, 1], wn, 'NLDAS-3 wet-day fraction', 'YlGnBu', 0, top),
        (axes[0, 2], wn - wa, 'Difference (NLDAS-3 − AORC)', 'RdBu', -0.2, 0.2),
    ]:
        im = ax.pcolormesh(lon_g, lat_g, data, cmap=cmap, vmin=lo, vmax=hi, shading='auto')
        plt.colorbar(im, ax=ax, shrink=0.8); ax.set_title(title, fontsize=12)
    for ax, key, xlabel in [(axes[1, 0], 'wet_day_frac', f'Wet-day fraction (≥{thr} mm)'),
                            (axes[1, 1], 'dry_spell_mean', 'Mean dry-spell length (days)'),
                            (axes[1, 2], 'wet_mean_mm', 'Mean wet-day intensity (mm/day)')]:
        for st, color, lbl in [(ia, COLORS['aorc'], 'AORC'), (inn, COLORS['nldas'], 'NLDAS-3')]:
            v = np.asarray(st[key], float); v = v[np.isfinite(v)]
            if v.size: ax.hist(v, bins=60, color=color, alpha=0.55, density=True, label=lbl)
        ax.set_xlabel(xlabel); ax.set_ylabel('Pixel density'); ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(EXT.fig_path(period, 'phase9_intermittency.png'), dpi=200)
    plt.close(fig)
    log('  Phase 9 done')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 10 — Spectral Analysis
# ══════════════════════════════════════════════════════════════════════════════
def phase10_spectral(df: pd.DataFrame, period: str):
    """
    FFT power spectra of all variables at monthly resolution.
    Compares AORC vs NLDAS on: dominant period, annual/semi-annual energy,
    and spectral ratio at each frequency.
    """
    log('Phase 10: Spectral Analysis')
    from pipeline.spectral import power_spectrum, compare_spectra, spectral_peaks

    spec_rows = []
    n_vars = len(ALL_VARS)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    _VAR_UNITS = {
        'precip': 'mm/day', 'temp': 'K', 'shortwave': 'W/m²',
        'longwave': 'W/m²', 'pressure': 'Pa',
        'specific_humidity': 'kg/kg', 'wind_2m': 'm/s',
    }

    for idx, var in enumerate(ALL_VARS):
        ca, cn = f'{var}_aorc', f'{var}_nldas'
        ax = axes.flat[idx]

        has_data = False
        for ds, col, color in [('AORC', ca, 'steelblue'), ('NLDAS', cn, 'tomato')]:
            if col not in df: continue
            monthly = df[col].resample('ME').mean().dropna()
            if len(monthly) < 12: continue
            spec = power_spectrum(monthly)
            if spec.empty: continue
            finite = spec[np.isfinite(spec['period_months']) &
                          (spec['period_months'] < 100) &
                          (spec['power'] > 0)]
            if finite.empty: continue
            has_data = True
            ax.semilogy(finite['period_months'], finite['power'],
                        color=color, lw=1.5, label=ds, alpha=0.85)
            # Save peaks
            peaks = spectral_peaks(spec, top_n=3)
            for _, pk in peaks.iterrows():
                spec_rows.append(dict(variable=var, dataset=ds,
                                      period_months=pk['period_months'],
                                      power=pk['power'],
                                      relative_power=pk['relative_power']))

        if has_data:
            # Mark annual and semi-annual AFTER data is plotted so ylim is set
            for p_mark in [3, 6, 12]:
                ax.axvline(p_mark, color='k', lw=0.6, linestyle=':', alpha=0.5)
            ylim = ax.get_ylim()
            for p_mark, label_m in [(3, '3mo'), (6, '6mo'), (12, '1yr')]:
                ax.text(p_mark, ylim[1], label_m, fontsize=6,
                        ha='center', va='bottom', color='k')

        units = _VAR_UNITS.get(var, '')
        # removed set_title', fontsize=8)
        ax.set_xlabel('Period (months)', fontsize=7)
        ax.set_ylabel('Power', fontsize=7)
        ax.legend(fontsize=7)

    for ax in axes.flat[n_vars:]: ax.set_visible(False)
    # suptitle removed — panel titles are self-describing
    plt.tight_layout()
    fig.savefig(EXT.fig_path(period, 'phase10_spectra.png'), dpi=150)
    plt.close(fig)

    if spec_rows:
        pd.DataFrame(spec_rows).round(6).to_csv(
            EXT.tbl_path(period, 'spectral_peaks.csv'), index=False)

    # Spectral ratio figure (NLDAS/AORC power at each frequency)
    ratio_rows = []
    for var in ['precip', 'temp', 'shortwave']:
        ca, cn = f'{var}_aorc', f'{var}_nldas'
        if ca not in df or cn not in df: continue
        result = compare_spectra(
            df[ca].resample('ME').mean().dropna(),
            df[cn].resample('ME').mean().dropna(),
        )
        if result and 'table' in result:
            result['table']['variable'] = var
            ratio_rows.append(result['table'])

    if ratio_rows:
        ratio_df = pd.concat(ratio_rows, ignore_index=True)
        ratio_df.to_csv(EXT.tbl_path(period, 'spectral_ratios.csv'), index=False)

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        for var, color in [('precip','steelblue'),('temp','orange'),('shortwave','green')]:
            sub = ratio_df[(ratio_df['variable'] == var) &
                           np.isfinite(ratio_df['period_months']) &
                           (ratio_df['period_months'] < 60)]
            if sub.empty: continue
            ax2.plot(sub['period_months'], sub['spectral_ratio'],
                     color=color, lw=1.5, label=var)
        ax2.axhline(1.0, color='k', lw=0.8, linestyle='--')
        ax2.set_xlabel('Period (months)')
        ax2.set_ylabel('NLDAS / AORC power ratio')
        ax2.legend()
        # suptitle removed — panel titles are self-describing
        plt.tight_layout()
        fig2.savefig(EXT.fig_path(period, 'phase10_spectral_ratio.png'), dpi=150)
        plt.close(fig2)

    log('  Phase 10 done')


# ══════════════════════════════════════════════════════════════════════════════
# PERIOD DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
PERIODS = {
    'full'      : ('2001-01-01', '2020-12-31', 'full_2001_2020'),
    'decade1'   : ('2001-01-01', '2010-12-31', 'decade_2001_2010'),
    'decade2'   : ('2011-01-01', '2020-12-31', 'decade_2011_2020'),
}


def annual_periods(year_start: int, year_end: int):
    return [(f'{y}-01-01', f'{y}-12-31', f'annual/{y}') for y in range(year_start, year_end+1)]



# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HydroSuiteAnalysis Pipeline')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--all',     action='store_true',
                     help='Run everything: full + both decades + annual 2001-2020')
    grp.add_argument('--full',    action='store_true',
                     help='Run full 2001-2020 period')
    grp.add_argument('--decades', action='store_true',
                     help='Run both decades: 2001-2010 and 2011-2020')
    grp.add_argument('--decade',  type=int, choices=[1, 2],
                     help='Run one decade: 1=2001-2010, 2=2011-2020')
    grp.add_argument('--year',    type=int,
                     help='Run a single year (e.g. --year 2005)')
    grp.add_argument('--years',   type=str,
                     help='Run annual for a range (e.g. --years 2001-2010)')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Re-run the common-footprint gridpass, ignoring results/<period>/gridpass_v2')
    parser.add_argument('--no-resume', action='store_true',
                        help='Discard an existing gridpass checkpoint instead of resuming from it')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check paths/file counts without running analysis')
    parser.add_argument('--phases', type=str, default=None,
                        help='Phases to run, e.g. --phases 7 or --phases 7,8 (default: all 1-10)')
    args = parser.parse_args()

    # Wire --rebuild-cache into run_period via a flag attribute
    run_period._rebuild_cache = args.rebuild_cache
    run_period._no_resume = args.no_resume

    # Parse phase filter
    phase_filter = None
    if args.phases:
        phase_filter = set(int(p.strip()) for p in args.phases.replace(',', ' ').split())
        log(f'Phase filter active: {sorted(phase_filter)}')

    wall = time.time()
    log(f'HydroSuiteAnalysis Pipeline — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    log(f'Results root: {EXT.RESULTS_ROOT}')

    to_run = []   # list of (start, end, label)

    if args.all:
        to_run.append(PERIODS['full'])
        to_run.append(PERIODS['decade1'])
        to_run.append(PERIODS['decade2'])
        to_run.extend(annual_periods(2001, 2020))

    elif args.full:
        to_run.append(PERIODS['full'])

    elif args.decades:
        to_run.append(PERIODS['decade1'])
        to_run.append(PERIODS['decade2'])

    elif args.decade == 1:
        to_run.append(PERIODS['decade1'])

    elif args.decade == 2:
        to_run.append(PERIODS['decade2'])

    elif args.year:
        yr = args.year
        to_run.append((f'{yr}-01-01', f'{yr}-12-31', f'annual/{yr}'))

    elif args.years:
        # parse '2001-2010' or '2005'
        parts = args.years.split('-')
        if len(parts) == 2:
            to_run.extend(annual_periods(int(parts[0]), int(parts[1])))
        else:
            yr = int(parts[0])
            to_run.append((f'{yr}-01-01', f'{yr}-12-31', f'annual/{yr}'))

    else:
        # Default: run everything
        log('No period flag specified — running --all')
        to_run.append(PERIODS['full'])
        to_run.append(PERIODS['decade1'])
        to_run.append(PERIODS['decade2'])
        to_run.extend(annual_periods(2001, 2020))

    for start, end, label in to_run:
        run_period(start, end, label, dry_run=args.dry_run, phases=phase_filter)

    h, rem = divmod(int(time.time()-wall), 3600)
    m, s   = divmod(rem, 60)
    log(f'\nAll done. Wall time: {h}h {m}m {s}s')
