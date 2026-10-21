"""
Regenerate Phase 2 error PDFs (with fixed mu) and Phase 6 rolling anomalies
from the cached parquet, using figures4papers style.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pipeline.fig_style import (
    apply_publication_style, CA, CN, PALETTE, SEASON_COLORS,
    VAR_LABELS, VAR_ORDER, BAR_KW, vlab,
)
from pipeline import stats as ST

apply_publication_style(font_size=16, axes_linewidth=2.5)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from ext_config import EXT
FIG_DIR = os.path.join(EXT.RESULTS_ROOT, 'full_2001_2020', 'final_figures')
# Daily domain-mean series on the common footprint (run_pipeline.py → gridpass_v2).
# The old df_cache.parquet came from the per-product loaders and is not used.
def _cache_path(period='full_2001_2020'):
    from ext_config import EXT
    return os.path.join(EXT.RESULTS_ROOT, period, 'gridpass_v2', 'domain_southern.parquet')
CACHE   = None
ALL_VARS = ['precip','temp','shortwave','longwave','specific_humidity','wind_2m','pressure']

def _save(name):
    # Save to temp then rename to avoid file-lock errors
    tmp = os.path.join(FIG_DIR, '_tmp_' + name)
    plt.savefig(tmp, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close('all')
    final = os.path.join(FIG_DIR, name)
    try: os.remove(final)
    except: pass
    os.rename(tmp, final)
    print(f'  saved  {name}')


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Error PDFs  (with fixed $\mu$ rendering)
# ══════════════════════════════════════════════════════════════════════════════
def fig_phase2_error_pdfs(df):
    print('  Phase 2: Error PDFs ...')
    seasons = ST.seasonal_split(df)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    for idx, var in enumerate(ALL_VARS):
        ca, cn = f'{var}_aorc', f'{var}_nldas'
        if ca not in df or cn not in df: continue
        both = df[[ca, cn]].dropna()
        ax = axes.flat[idx]

        errors = both[cn].values - both[ca].values
        ST.error_pdf(ax, errors, label=VAR_LABELS.get(var, var), color=CA)

        for sname, sdf in seasons.items():
            sb = sdf[[ca, cn]].dropna() if ca in sdf and cn in sdf else pd.DataFrame()
            if not sb.empty:
                se = sb[cn].values - sb[ca].values
                ST.error_pdf(ax, se, label=sname, color=SEASON_COLORS.get(sname, 'grey'))

        ax.set_ylabel(VAR_LABELS.get(var, var), fontsize=12)
        ax.tick_params(axis='y', labelsize=11)

        if idx >= 4:
            ax.tick_params(axis='x', labelsize=11)
            ax.set_xlabel('Error (NLDAS \u2212 AORC)', fontsize=11)
        else:
            ax.tick_params(axis='x', labelbottom=False)
            ax.set_xlabel('')

        locs = {'precip': 'upper left', 'temp': 'upper left',
                'shortwave': 'upper left', 'pressure': 'upper left'}
        ax.legend(fontsize=9, loc=locs.get(var, 'upper right'), frameon=False)

    axes.flat[7].set_visible(False)
    axes.flat[8].set_visible(False)
    plt.tight_layout()
    _save('phase2_error_pdfs.png')


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6 — 12-Month Rolling Anomalies  (figures4papers style)
# ══════════════════════════════════════════════════════════════════════════════
def fig_rolling_anomalies(df):
    print('  Phase 6: Rolling anomalies ...')

    _BOUNDS = {
        'precip': (0, 200), 'temp': (200, 340),
        'shortwave': (0, 600), 'longwave': (100, 600),
        'specific_humidity': (0, 0.03), 'wind_2m': (0, 30),
        'pressure': (50000, 120000),
    }
    _AORC_WIND_BREAK = pd.Timestamp('2018-01-01')

    def _flag_artifact_years(mon_series, z_thresh=2.5):
        ann = mon_series.resample('YE').mean().dropna()
        if len(ann) < 5: return mon_series
        mu, sigma = ann.mean(), ann.std()
        if sigma == 0: return mon_series
        bad_years = ann.index.year[np.abs((ann - mu) / sigma) > z_thresh]
        if len(bad_years) == 0: return mon_series
        cleaned = mon_series.copy()
        for yr in bad_years:
            cleaned[cleaned.index.year == yr] = np.nan
        return cleaned

    colors = {'aorc': CA, 'nldas': CN}

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    plot_vars = [v for v in ALL_VARS if any(f'{v}_{ds}' in df.columns for ds in ['aorc','nldas'])]

    for idx, var in enumerate(plot_vars[:8]):
        ax = axes.flat[idx]
        lo, hi = _BOUNDS.get(var, (-1e9, 1e9))
        for ds, color in colors.items():
            col = f'{var}_{ds}'
            if col not in df: continue
            mon = df[col].resample('ME').mean()
            mon = mon.where((mon >= lo) & (mon <= hi))
            mon = _flag_artifact_years(mon)

            if var == 'wind_2m' and ds == 'aorc':
                mon_pre = mon[mon.index < _AORC_WIND_BREAK]
                clim_pre = mon_pre.groupby(mon_pre.index.month).transform('mean')
                anom_pre = (mon_pre - clim_pre).dropna()
                if len(anom_pre) >= 3:
                    roll_pre = anom_pre.rolling(12, min_periods=6).mean()
                    ax.plot(roll_pre.index, roll_pre.values, color=color, lw=2.5, label=ds.upper())
                ax.axvline(_AORC_WIND_BREAK, color='red', lw=1.5, linestyle='--', alpha=0.7)
                ax.text(_AORC_WIND_BREAK, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.1,
                        ' AORC\n data\n break', color='red', fontsize=9, va='bottom')
                continue

            clim_aligned = mon.groupby(mon.index.month).transform('mean')
            anom = (mon - clim_aligned).dropna()
            if len(anom) < 3: continue
            roll = anom.rolling(12, min_periods=6).mean()
            ax.plot(roll.index, roll.values, color=color, lw=2.5, label=ds.upper())
            mk = ST.mann_kendall(anom)
            if mk.get('trend', 'no trend') != 'no trend':
                ax.set_facecolor('#fff8f0')

        ax.axhline(0, color='k', lw=0.8, linestyle=':')
        ax.set_ylabel(VAR_LABELS.get(var, var), fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, frameon=False)
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

    # 8th panel: legend
    if len(plot_vars) < 8:
        ax8 = axes.flat[len(plot_vars)]
    else:
        ax8 = axes.flat[7]
    ax8.axis('off')
    aorc_line  = mlines.Line2D([], [], color=CA, lw=3, label='AORC (Reference)')
    nldas_line = mlines.Line2D([], [], color=CN, lw=3, label='NLDAS-3 (Evaluated)')
    sig_patch  = mpatches.Patch(color='#fff8f0', ec='gray', label='Significant Trend (p < 0.05)')
    break_line = mlines.Line2D([], [], color='red', lw=2, linestyle='--', label='AORC v1.1 Data Discontinuity')
    ax8.legend(handles=[aorc_line, nldas_line, sig_patch, break_line],
               loc='center', fontsize=18, frameon=False, borderpad=1.5, labelspacing=1.2)

    fig.supylabel('Anomaly', fontsize=22, fontweight='bold', x=0.01)
    plt.tight_layout()
    _save('phase6_rolling_anomalies.png')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    cache = CACHE or _cache_path()
    print(f'Loading parquet cache: {cache}')
    df = pd.read_parquet(cache)
    print(f'  Loaded: {len(df):,} rows, {len(df.columns)} columns\n')

    fig_phase2_error_pdfs(df)
    fig_rolling_anomalies(df)

    print('\nDone.')
