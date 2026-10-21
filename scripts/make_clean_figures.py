"""
make_clean_figures.py
=====================
Figure functions for the result tables (called by make_all_figures.py).
Style: figures4papers (ChenLiu-1996/figures4papers)

Usage:
    python scripts/make_clean_figures.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Import shared style ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pipeline.fig_style import (
    apply_publication_style, finalize_figure,
    CA, CN, PALETTE, SEASON_COLORS, VAR_COLORS, VAR_LABELS, VAR_ORDER,
    BAR_KW, BAR_WIDTH, annotate_bars, make_legend_panel, vlab,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from ext_config import EXT                      # defaults; make_all_figures.py sets both per period
TBL_DIR = os.path.join(EXT.RESULTS_ROOT, 'full_2001_2020', 'tables')
FIG_DIR = os.path.join(EXT.RESULTS_ROOT, 'full_2001_2020', 'final_figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Apply the figure style ONCE ─────────────────────────────────────────────
apply_publication_style(font_size=16, axes_linewidth=2.5)

# ── Helpers ──────────────────────────────────────────────────────────────────
def _tbl(name):
    return pd.read_csv(os.path.join(TBL_DIR, name))

def _save(name):
    tmp = os.path.join(FIG_DIR, '_tmp_' + name)
    plt.savefig(tmp, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close('all')
    final = os.path.join(FIG_DIR, name)
    try: os.remove(final)
    except: pass
    os.rename(tmp, final)
    print(f'  saved  {name}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1  ─  Climatological means  (grouped bar, one panel per unit family)
# ══════════════════════════════════════════════════════════════════════════════
def fig_climatology():
    clim = _tbl('climatology_stats.csv')
    bm   = _tbl('bias_metrics.csv')
    ann  = bm[bm['season'] == 'All'].set_index('variable')

    groups = [
        (['precip'],                           'mm day$^{-1}$'),
        (['temp'],                              'K'),
        (['shortwave', 'longwave'],             'W m$^{-2}$'),
        (['specific_humidity'],                 'kg kg$^{-1}$'),
        (['wind_2m'],                           'm s$^{-1}$'),
        (['pressure'],                          'Pa'),
    ]
    groups = [(vl, u) for vl, u in groups
              if any(v in clim['variable'].values for v in vl)]

    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(max(3.5 * n, 16), 5.5),
                             sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (vars_, unit) in zip(axes, groups):
        vars_ = [v for v in vars_ if v in clim['variable'].values]
        x = np.arange(len(vars_))
        w = BAR_WIDTH
        aorc = clim[clim['dataset'] == 'AORC'].set_index('variable')
        nld  = clim[clim['dataset'] == 'NLDAS'].set_index('variable')
        ma = [aorc.loc[v, 'mean'] if v in aorc.index else np.nan for v in vars_]
        mn = [nld.loc[v,  'mean'] if v in nld.index  else np.nan for v in vars_]
        ax.bar(x - w/2, ma, w, color=CA, label='AORC', **BAR_KW)
        ax.bar(x + w/2, mn, w, color=CN, label='NLDAS', **BAR_KW)
        ax.set_xticks(x)
        ax.set_xticklabels([vlab(v) for v in vars_], rotation=15, ha='right')
        ax.set_ylabel(unit)
        if ax == axes[0]:
            ax.legend(fontsize=14)

    plt.tight_layout(w_pad=1.5)
    _save('phase1_monthly_climatology.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2a  ─  Taylor diagram  (z-normalised, NLDAS vs AORC)
# ══════════════════════════════════════════════════════════════════════════════
def fig_taylor():
    # Each variable is compared with ITS OWN AORC series (normalized std = σ_NLDAS/σ_AORC)
    # on deseasonalized daily anomalies (taylor_stats.csv, Phase 2). Raw daily r is
    # dominated by the shared annual cycle (≈1.00 for temperature whatever the skill).
    ts = _tbl('taylor_stats.csv').set_index('variable')

    fig = plt.figure(figsize=(7, 6.5))
    ax  = fig.add_subplot(111, projection='polar')

    std_ref = 1.0
    std_max = 1.65

    # RMSE dashed arcs centred on (r=1, sigma=0)
    for rv in [0.25, 0.5, 0.75, 1.0, 1.25]:
        th = np.linspace(0, np.pi / 2, 300)
        xc = std_ref + rv * np.cos(th + np.pi)
        yc =           rv * np.sin(th + np.pi)
        ra = np.sqrt(xc**2 + yc**2)
        ta = np.arctan2(yc, xc)
        m  = (ta >= 0) & (ta <= np.pi / 2) & (ra <= std_max)
        if m.any():
            ax.plot(ta[m], ra[m], '--', color='grey', lw=0.6, alpha=0.5)

    # Reference star (AORC)
    ax.plot(0, std_ref, 'k*', markersize=14, zorder=5, label='AORC (ref.)')

    legend_handles = []
    for i, var in enumerate(VAR_ORDER):
        if var not in ts.index:
            continue
        r_val   = float(ts.loc[var, 'r_anom'])
        sn_norm = float(ts.loc[var, 'std_ratio_anom'])
        theta_  = np.arccos(np.clip(r_val, -1, 1))

        c = VAR_COLORS.get(var, '#4D4D4D')
        h, = ax.plot(theta_, sn_norm, 'o', color=c, markersize=10, zorder=4,
                     label=f'{vlab(var)}  (r = {r_val:.2f})')
        legend_handles.append(h)

    ax.set_thetamax(90)
    ax.set_rlim(0, std_max)
    ax.set_thetagrids([0, 15, 30, 45, 60, 75, 90],
                      ['1.0', '0.97', '0.87', '0.71', '0.50', '0.26', '0.0'])
    ax.set_xlabel('Correlation coefficient', labelpad=20, fontsize=14)

    ax.legend(handles=legend_handles, loc='center left',
              bbox_to_anchor=(1.15, 0.5), fontsize=12,
              frameon=False, title='Variable', title_fontsize=12)

    plt.tight_layout()
    _save('phase2_taylor.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2b  ─  Bias heatmap  (PBIAS % by variable x season)
# ══════════════════════════════════════════════════════════════════════════════
def fig_bias_heatmap():
    df  = _tbl('bias_metrics.csv')
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    vars_   = [v for v in VAR_ORDER if v in df['variable'].values]
    pivot   = (df[df['season'].isin(seasons)]
               .pivot(index='variable', columns='season', values='pbias')
               .reindex(index=vars_, columns=seasons))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap='RdBu', vmin=-55, vmax=55, aspect='auto')
    cb = plt.colorbar(im, ax=ax, label='PBIAS (%)', fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=12)
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons)
    ax.set_yticks(range(len(vars_)))
    ax.set_yticklabels([vlab(v) for v in vars_])
    ax.set_xlabel('Season')
    ax.set_ylabel('Variable')

    for i in range(len(vars_)):
        for j in range(len(seasons)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                tc = 'white' if abs(v) > 35 else 'black'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=12, color=tc, fontweight='bold')
    ax.grid(False)
    plt.tight_layout()
    _save('phase2_bias_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3  ─  ET0 attribution bar
# ══════════════════════════════════════════════════════════════════════════════
def fig_et0_sensitivity():
    df  = _tbl('et0_sensitivity.csv').sort_values('mean_contribution_mm_d')
    met = _tbl('et0_metrics.csv').iloc[0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors  = [CA if v > 0 else CN for v in df['mean_contribution_mm_d']]
    ylabels = df['variable'].map(VAR_LABELS).fillna(df['variable'])

    ax.barh(ylabels, df['mean_contribution_mm_d'], color=colors,
            height=0.55, edgecolor='black', linewidth=1.5, alpha=0.88)
    ax.axvline(0, color='k', lw=0.9)

    vals = df['mean_contribution_mm_d']
    span = max(vals.max(), 0) - min(vals.min(), 0)
    xoff = 0.015 * span
    for val, label in zip(vals, ylabels):
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + (xoff if val >= 0 else -xoff), label, f'{val:+.3f}', va='center',
                ha=ha, fontsize=12, fontweight='bold')
    # room for the value labels beyond the longest bars on both sides
    ax.set_xlim(min(vals.min(), 0) - 0.22 * span, max(vals.max(), 0) + 0.22 * span)

    ax.set_xlabel(r'Mean contribution to ET$_0$ difference (mm day$^{-1}$)', fontsize=12)
    ax.set_ylabel('Forcing variable', fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    info = (f'Net: {met["bias"]:+.3f} mm day' + r'$^{-1}$' + '\n'
            f'RMSE = {met["rmse"]:.3f}   r = {met["r"]:.3f}   PBIAS = {met["pbias"]:.1f}%')
    ax.text(0.98, 0.04, info, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.35', fc='white',
                                    ec=PALETTE['neutral'], alpha=0.9))
    plt.tight_layout()
    _save('phase3_et0.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4  ─  GEV return periods
# ══════════════════════════════════════════════════════════════════════════════
def fig_gev():
    df = _tbl('return_period_values.csv')
    Ts = [2, 5, 10, 25, 50, 100]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in df.iterrows():
        ds    = row['dataset']
        color = CA if ds == 'AORC' else CN
        qts   = [row[f'T{t}yr_mm']    for t in Ts]
        cilo  = [row[f'T{t}yr_ci_lo'] for t in Ts]
        cihi  = [row[f'T{t}yr_ci_hi'] for t in Ts]
        ax.semilogx(Ts, qts, 'o-', color=color, lw=2.5, ms=8, label=ds)
        ax.fill_between(Ts, cilo, cihi, color=color, alpha=0.12)

    ax.set_xlabel('Return period (yr)')
    ax.set_ylabel(r'Daily precipitation (mm day$^{-1}$)')
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(t) for t in Ts])
    ax.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    _save('phase4_precip_extremes.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5  ─  Circular seasonal stats  (R bars + mean DOY bars)
# ══════════════════════════════════════════════════════════════════════════════
def fig_circular():
    df    = _tbl('circular_seasonal_stats.csv')
    vars_ = [v for v in VAR_ORDER if v in df['variable'].values]
    aorc  = df[df['dataset'] == 'AORC'].set_index('variable')
    nld   = df[df['dataset'] == 'NLDAS'].set_index('variable')
    x  = np.arange(len(vars_))
    w  = BAR_WIDTH
    xlbls = [vlab(v) for v in vars_]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ── Panel A: mean resultant length R ──────────────────────────────────────
    ax = axes[0]
    ra = [aorc.loc[v, 'R'] if v in aorc.index else np.nan for v in vars_]
    rn = [nld.loc[v,  'R'] if v in nld.index  else np.nan for v in vars_]
    bars_a = ax.bar(x - w/2, ra, w, color=CA, label='AORC', **BAR_KW)
    bars_n = ax.bar(x + w/2, rn, w, color=CN, label='NLDAS', **BAR_KW)
    for xi, (a, n) in enumerate(zip(ra, rn)):
        if np.isfinite(a):
            ax.text(xi - w/2, a + 0.01, f'{a:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        if np.isfinite(n):
            ax.text(xi + w/2, n + 0.01, f'{n:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, rotation=28, ha='right')
    ax.set_ylabel('Mean Resultant Length $R$\n(0 = uniform,  1 = concentrated)')
    ax.set_ylim(0, min(max(max(ra, default=0), max(rn, default=0)) * 1.25, 1.05))
    ax.legend(fontsize=14, frameon=False)

    # ── Panel B: mean DOY ─────────────────────────────────────────────────────
    ax2 = axes[1]
    da = [aorc.loc[v, 'mean_doy'] if v in aorc.index else np.nan for v in vars_]
    dn = [nld.loc[v,  'mean_doy'] if v in nld.index  else np.nan for v in vars_]
    ax2.bar(x - w/2, da, w, color=CA, label='AORC', **BAR_KW)
    ax2.bar(x + w/2, dn, w, color=CN, label='NLDAS', **BAR_KW)
    for xi, (a, n) in enumerate(zip(da, dn)):
        if np.isfinite(a):
            ax2.text(xi - w/2, a + 4, f'{int(a)}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        if np.isfinite(n):
            ax2.text(xi + w/2, n + 4, f'{int(n)}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    month_ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    ax2.set_yticks(month_ticks)
    ax2.set_yticklabels(month_names)
    ax2.set_ylabel('Mean Day of Annual Peak')
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlbls, rotation=28, ha='right')

    plt.tight_layout(h_pad=2.5)
    _save('phase5_seasonal_concentration.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6a  ─  LOYO-CV bias correction  (3 metrics)
# ══════════════════════════════════════════════════════════════════════════════
def fig_bias_correction():
    df   = _tbl('bias_correction_loyo.csv')
    yrs  = df['year'].astype(int).values

    metrics = [
        ('raw_rmse',  'qm_rmse',  'lr_rmse',  r'RMSE (mm day$^{-1}$)', 'A) Root Mean Square Error'),
        ('raw_r',     'qm_r',     'lr_r',      'Pearson r',                'B) Pearson Correlation'),
        ('raw_pbias', 'qm_pbias', 'lr_pbias',  'PBIAS (%)',                'C) Percent Bias'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False, sharex=False)
    fig.supxlabel('Left-out year (LOYO-CV)', fontsize=16, y=-0.01, fontweight='bold')

    for i, (ax, (rc, qc, lc, ylabel, title)) in enumerate(zip(axes.flat, metrics)):
        ax.set_title(title, fontsize=16, pad=12, fontweight='bold')
        
        ax.plot(yrs, df[rc].values, marker='o', color=PALETTE['neutral'],
                lw=4, alpha=0.5, ms=7, label='Raw NLDAS')
        
        if i == 0:
            ax.fill_between(yrs, df[rc].values, df[qc].values,
                            color=CA, alpha=0.15, label='QM Bias Reduction')
            
        ax.plot(yrs, df[qc].values, marker='D', color=CA, lw=2.5, ms=6, label='Quantile Mapping')
        ax.plot(yrs, df[lc].values, marker='^', color=CN, lw=2.5, ms=6, label='Linear Regression')
        
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(0, color='k', lw=1, ls='-', alpha=0.3)
        
        if 'pbias' in rc:
            ax.axhspan(-12, 12, color=PALETTE['green_1'], alpha=0.3, label='\u00b112% Acceptable')
            ax.axhline(12,  color=PALETTE['green_3'], lw=1.2, ls='--', alpha=0.6)
            ax.axhline(-12, color=PALETTE['green_3'], lw=1.2, ls='--', alpha=0.6)
            
        if i == 0:
            ax.legend(fontsize=11, loc='upper right', frameon=False)
        elif i == 2:
            ax.legend(fontsize=11, loc='lower right', frameon=False)

    for a in axes.flat:
        a.set_xticks(yrs[::2])
        a.set_xticklabels([str(y) for y in yrs[::2]])

    ax4 = axes.flat[3]
    ax4.set_title('D) RMSE Reduction (%)', fontsize=16, pad=12, fontweight='bold')
    qm_red = (df['raw_rmse'] - df['qm_rmse']) / df['raw_rmse'] * 100
    lr_red = (df['raw_rmse'] - df['lr_rmse']) / df['raw_rmse'] * 100
    ax4.plot(yrs, qm_red.values, marker='D', color=CA, lw=2.5, ms=6, label='Quantile Mapping')
    ax4.plot(yrs, lr_red.values, marker='^', color=CN, lw=2.5, ms=6, label='Linear Regression')
    ax4.set_ylabel('RMSE Reduction (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(0, color='k', lw=1, ls='-', alpha=0.3)

    plt.tight_layout(h_pad=2.5, w_pad=2.0)
    _save('phase6_bias_correction.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6b  ─  Trend heatmap  (neg-log10 p, direction arrows)
# ══════════════════════════════════════════════════════════════════════════════
def fig_trends():
    df    = _tbl('trend_tests.csv')
    vars_ = [v for v in VAR_ORDER if v in df['variable'].values]
    dsets = ['AORC', 'NLDAS']
    pmat  = np.full((len(dsets), len(vars_)), np.nan)
    tmat  = {}

    for i, ds in enumerate(dsets):
        for j, v in enumerate(vars_):
            row = df[(df['dataset'] == ds) & (df['variable'] == v)]
            if row.empty:
                continue
            p = float(row['mk_p'].values[0])
            pmat[i, j] = -np.log10(max(p, 1e-4))
            tmat[(i, j)] = (row['mk_trend'].values[0], p)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pmat, cmap='YlOrRd', vmin=0, vmax=3, aspect='auto')
    cb = plt.colorbar(im, ax=ax,
                      label=r'$-\log_{10}(p)$  [$\geq$2 $\Rightarrow$ $p \leq$0.01]',
                      fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=13)
    ax.set_xticks(range(len(vars_)))
    ax.set_xticklabels([vlab(v) for v in vars_], rotation=25, ha='right')
    ax.set_yticks(range(len(dsets)))
    ax.set_yticklabels(dsets)
    ax.set_xlabel('Variable')

    for (i, j), (trend, p) in tmat.items():
        arrow = '\u2191' if trend == 'increasing' else ('\u2193' if trend == 'decreasing' else '\u2014')
        tc    = 'white' if pmat[i, j] > 2 else 'black'
        ax.text(j, i, f'{arrow}\np={p:.2f}',
                ha='center', va='center', fontsize=13, color=tc)

    ax.grid(False)
    plt.tight_layout()
    _save('phase6_trend_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8  ─  Scale sensitivity  (Southern vs Louisiana, r and RMSE)
# ══════════════════════════════════════════════════════════════════════════════
def fig_scale_sensitivity():
    df    = _tbl('scale_sensitivity.csv')
    vars_ = [v for v in VAR_ORDER if v in df['variable'].values]
    xlbls = [vlab(v) for v in vars_]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, (ax, (metric, ylabel)) in enumerate(zip(axes,
            [('r',    'Pearson $r$  (NLDAS vs AORC)'),
             ('rmse', 'RMSE  (native units)')])):
        for dom, color, marker, lbl in [
            ('southern',  CA, 'o', 'Southern US'),
            ('louisiana', CN, 's', 'Louisiana'),
        ]:
            sub = df[df['domain'] == dom].set_index('variable')
            vals = [sub.loc[v, metric] if v in sub.index else np.nan for v in vars_]
            ax.plot(range(len(xlbls)), vals, marker=marker, color=color,
                    lw=2.5, ms=10, label=lbl)
        ax.set_xticks(range(len(xlbls)))
        ax.set_xticklabels(xlbls, rotation=30, ha='right')
        ax.set_xlabel('Variable')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=14, frameon=False)

    plt.tight_layout(w_pad=2)
    _save('phase8_scale_sensitivity.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9  ─  Intermittency  (3 panels: bar comparison, % diff, seasonal wet frac)
# ══════════════════════════════════════════════════════════════════════════════
def fig_intermittency():
    df   = _tbl('intermittency_comparison.csv')
    asea = _tbl('intermittency_seasonal_aorc.csv')
    nsea = _tbl('intermittency_seasonal_nldas.csv')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A - absolute comparison of key metrics
    ax = axes[0,0]
    show = ['wet_day_frac', 'wet_mean_mm', 'wet_spell_mean', 'P90_intensity_mm']
    lbls = ['Wet-day\nfraction', 'Mean wet\nintensity\n(mm day$^{-1}$)',
            'Mean wet\nspell (days)', 'P90 intensity\n(mm day$^{-1}$)']
    sub  = df[df['metric'].isin(show)].set_index('metric').reindex(show)
    x = np.arange(len(show))
    w = BAR_WIDTH
    ax.bar(x - w/2, sub['aorc'],  w, color=CA, label='AORC',  **BAR_KW)
    ax.bar(x + w/2, sub['nldas'], w, color=CN, label='NLDAS', **BAR_KW)
    for xi, (a, n) in enumerate(zip(sub['aorc'], sub['nldas'])):
        ax.text(xi - w/2, a + 0.01 * a, f'{a:.2f}', ha='center', va='bottom', fontsize=12)
        ax.text(xi + w/2, n + 0.01 * n, f'{n:.2f}', ha='center', va='bottom', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls)
    ax.set_ylabel('Value')
    ax.legend(fontsize=14, frameon=False)

    # Panel B - % difference (all metrics); horizontal bars
    ax2 = axes[0,1]
    lmap = {
        'wet_day_frac':     'Wet-day fraction',
        'dry_day_frac':     'Dry-day fraction',
        'wet_mean_mm':      'Mean wet intensity',
        'wet_spell_mean':   'Mean wet spell',
        'wet_spell_max':    'Max wet spell',
        'dry_spell_mean':   'Mean dry spell',
        'dry_spell_max':    'Max dry spell',
        'P90_intensity_mm': 'P90 intensity',
        'frac_intense':     'Frac. intense',
        'total_mm':         '20-yr total',
    }
    ylabs  = [lmap.get(m, m) for m in df['metric']]
    colors = [CN if v < 0 else CA for v in df['pct_diff']]
    ax2.barh(ylabs, df['pct_diff'].values, color=colors, alpha=0.88,
             height=0.6, edgecolor='black', linewidth=1.5)
    ax2.axvline(0, color='k', lw=0.9)
    for val, lab in zip(df['pct_diff'].values, ylabs):
        xoff = 0.8 if val >= 0 else -0.8
        ha   = 'left' if val >= 0 else 'right'
        ax2.text(val + xoff, lab, f'{val:+.1f}%', va='center', ha=ha, fontsize=12)
    
    xmin, xmax = ax2.get_xlim()
    ax2.set_xlim(min(-60, xmin), max(60, xmax))
    ax2.set_xlabel('% difference  (NLDAS vs AORC)')

    # Panel C - seasonal wet-day fraction
    ax3 = axes[1,0]
    if 'season' in asea.columns and 'wet_day_frac' in asea.columns:
        seasons = asea['season'].tolist()
        x3 = np.arange(len(seasons))
        ax3.bar(x3 - w/2, asea['wet_day_frac'].values, w,
                color=CA, label='AORC', **BAR_KW)
        ax3.bar(x3 + w/2, nsea['wet_day_frac'].values, w,
                color=CN, label='NLDAS', **BAR_KW)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(seasons)
        ax3.set_ylabel('Wet-day Fraction')
    else:
        ax3.axis('off')

    # Panel D - seasonal mean wet-day intensity (same layout as panel C)
    ax4 = axes[1,1]
    if 'season' in asea.columns and 'wet_mean_mm' in asea.columns:
        x4 = np.arange(len(asea))
        ax4.bar(x4 - w/2, asea['wet_mean_mm'].values, w,
                color=CA, label='AORC', **BAR_KW)
        ax4.bar(x4 + w/2, nsea['wet_mean_mm'].values, w,
                color=CN, label='NLDAS', **BAR_KW)
        ax4.set_xticks(x4)
        ax4.set_xticklabels(asea['season'].tolist())
        ax4.set_ylabel('Mean wet-day intensity (mm day$^{-1}$)')
    else:
        ax4.axis('off')
    plt.tight_layout(h_pad=2.0, w_pad=1.8)
    _save('phase9_intermittency.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10a  ─  Spectral power ratio vs period
# ══════════════════════════════════════════════════════════════════════════════
def fig_spectral_ratio():
    df = _tbl('spectral_ratios.csv')
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    pal = {
        'precip':    VAR_COLORS['precip'],
        'temp':      VAR_COLORS['temp'],
        'shortwave': VAR_COLORS['shortwave'],
    }
    for var, color in pal.items():
        sub = (df[(df['variable'] == var)
                  & df['period_months'].apply(np.isfinite)
                  & (df['period_months'] > 0)
                  & (df['period_months'] < 60)]
               .sort_values('period_months'))
        if sub.empty:
            continue
        ax.plot(sub['period_months'], sub['spectral_ratio'],
                color=color, lw=2.5, label=vlab(var))

    ax.axhline(1.0, color='k', lw=1.0, ls='--', alpha=0.7)
    ax.set_xlabel('Period (months)')
    ax.set_ylabel('NLDAS / AORC spectral power ratio')
    ax.legend(fontsize=14, frameon=False)
    ax.text(0.98, 0.97, '>1 = NLDAS more energetic',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, color='0.4')
    plt.tight_layout()
    _save('phase10_spectral_ratio.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10b  ─  Dominant spectral peak relative power
# ══════════════════════════════════════════════════════════════════════════════
def fig_spectral_peaks():
    df   = _tbl('spectral_peaks.csv')
    idx  = df.groupby(['variable', 'dataset'])['relative_power'].idxmax()
    top  = df.loc[idx].reset_index(drop=True)

    vars_ = [v for v in VAR_ORDER if v in top['variable'].values]
    x = np.arange(len(vars_))
    w = BAR_WIDTH
    aorc_t = top[top['dataset'] == 'AORC'].set_index('variable')
    nld_t  = top[top['dataset'] == 'NLDAS'].set_index('variable')

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rpa = [aorc_t.loc[v, 'relative_power'] if v in aorc_t.index else np.nan for v in vars_]
    rpn = [nld_t.loc[v,  'relative_power'] if v in nld_t.index  else np.nan for v in vars_]
    ax.bar(x - w/2, rpa, w, color=CA, label='AORC',  **BAR_KW)
    ax.bar(x + w/2, rpn, w, color=CN, label='NLDAS', **BAR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels([vlab(v) for v in vars_], rotation=20, ha='right')
    ax.set_ylabel('Relative power at dominant period')
    ax.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    _save('phase10_spectra.png')


# ══════════════════════════════════════════════════════════════════════════════
# Synthesis  ─  4-panel cross-pipeline overview
# ══════════════════════════════════════════════════════════════════════════════
def fig_synthesis():
    bm  = _tbl('bias_metrics.csv')
    et  = _tbl('et0_sensitivity.csv')
    rat = _tbl('spectral_ratios.csv')
    trd = _tbl('trend_tests.csv')

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    # ── A: Pearson r per variable ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ann  = bm[bm['season'] == 'All'].set_index('variable')
    vars_a = [v for v in VAR_ORDER if v in ann.index]
    rs   = [ann.loc[v, 'r'] for v in vars_a]
    ylbs = [vlab(v) for v in vars_a]
    bars = ax_a.barh(ylbs, rs, color=CA, height=0.55,
                     edgecolor='black', linewidth=1.5, alpha=0.88)
    ax_a.axvline(0.75, color=PALETTE['green_3'], lw=1.2, ls='--', alpha=0.65, label='$r$=0.75')
    for bar, r_val in zip(bars, rs):
        ax_a.text(r_val + 0.01, bar.get_y() + bar.get_height() / 2,
                  f'{r_val:.2f}', va='center', fontsize=12)
    ax_a.set_xlabel('Pearson $r$  (NLDAS vs AORC, annual)')
    ax_a.set_ylabel('Variable')
    ax_a.legend(fontsize=12, frameon=False)

    # ── B: ET0 attribution ───────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    et_s = et.sort_values('mean_contribution_mm_d')
    cbs  = [CA if v > 0 else CN for v in et_s['mean_contribution_mm_d']]
    ylbs_b = et_s['variable'].map(VAR_LABELS).fillna(et_s['variable'])
    ax_b.barh(ylbs_b, et_s['mean_contribution_mm_d'], color=cbs,
              alpha=0.88, height=0.55, edgecolor='black', linewidth=1.5)
    ax_b.axvline(0, color='k', lw=0.9)
    ax_b.set_xlabel(r'ET$_0$ contribution  (mm day$^{-1}$)')
    ax_b.set_ylabel('Forcing variable')

    # ── C: Spectral ratio ─────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    pal = {
        'precip':    VAR_COLORS['precip'],
        'temp':      VAR_COLORS['temp'],
        'shortwave': VAR_COLORS['shortwave'],
    }
    for var, color in pal.items():
        sub = (rat[(rat['variable'] == var)
                   & rat['period_months'].apply(np.isfinite)
                   & (rat['period_months'] > 0)
                   & (rat['period_months'] < 50)]
               .sort_values('period_months'))
        if sub.empty:
            continue
        ax_c.plot(sub['period_months'], sub['spectral_ratio'],
                  color=color, lw=2.5, label=vlab(var))
    ax_c.axhline(1.0, color='k', lw=0.8, ls='--')
    ax_c.set_xlabel('Period (months)')
    ax_c.set_ylabel('NLDAS / AORC spectral power ratio')
    ax_c.legend(fontsize=12, frameon=False)

    # ── D: Trend significance heatmap ─────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    vars_d = [v for v in VAR_ORDER if v in trd['variable'].values]
    dsets  = ['AORC', 'NLDAS']
    pmat   = np.full((len(dsets), len(vars_d)), np.nan)
    tmap   = {}
    for i, ds in enumerate(dsets):
        for j, v in enumerate(vars_d):
            row = trd[(trd['dataset'] == ds) & (trd['variable'] == v)]
            if row.empty:
                continue
            p = float(row['mk_p'].values[0])
            pmat[i, j] = -np.log10(max(p, 1e-4))
            tmap[(i, j)] = (row['mk_trend'].values[0], p)

    im = ax_d.imshow(pmat, cmap='YlOrRd', vmin=0, vmax=3, aspect='auto')
    plt.colorbar(im, ax=ax_d, label=r'$-\log_{10}(p)$', shrink=0.85,
                 fraction=0.03)
    ax_d.set_xticks(range(len(vars_d)))
    ax_d.set_xticklabels([vlab(v) for v in vars_d], rotation=22,
                          ha='right')
    ax_d.set_yticks(range(len(dsets)))
    ax_d.set_yticklabels(dsets)
    for (i, j), (trend, p) in tmap.items():
        arrow = '\u2191' if trend == 'increasing' else ('\u2193' if trend == 'decreasing' else '\u2014')
        tc    = 'white' if pmat[i, j] > 2 else 'black'
        ax_d.text(j, i, f'{arrow}\np={p:.2f}',
                  ha='center', va='center', fontsize=12, color=tc)
    ax_d.grid(False)

    # Panel labels (A B C D)
    for ax, lbl in [(ax_a, '(A)'), (ax_b, '(B)'), (ax_c, '(C)'), (ax_d, '(D)')]:
        ax.text(-0.10, 1.04, lbl, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom')

    _save('synthesis_figure.png')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Generating clean figures -> {FIG_DIR}\n')
    fig_climatology();        print()
    fig_taylor();             print()
    fig_bias_heatmap();       print()
    fig_et0_sensitivity();    print()
    fig_gev();                print()
    fig_circular();           print()
    fig_bias_correction();    print()
    fig_trends();             print()
    fig_scale_sensitivity();  print()
    fig_intermittency();      print()
    fig_spectral_ratio();     print()
    fig_spectral_peaks();     print()
    fig_synthesis();          print()
    print('\nDone.')
