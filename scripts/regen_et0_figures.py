"""
regen_et0_figures.py
====================
Regenerates ET0 station-validation figures.
Style: figures4papers (ChenLiu-1996/figures4papers)

  et01_bias_map.png    - Spatial distribution of HS-ET0 bias
  et02_taylor_all.png  - Station-level Taylor diagram

Data source: results/et0_validation/<period>/tables/et0_station_metrics.csv
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

# Shared style
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pipeline.fig_style import (
    apply_publication_style, finalize_figure,
    CA, CN, PALETTE, vlab,
)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from ext_config import EXT
TBL_DIR = os.path.join(EXT.RESULTS_ROOT, 'et0_validation', 'full_2001_2020', 'tables')
FIG_DIR = os.path.join(EXT.RESULTS_ROOT, 'full_2001_2020', 'final_figures')
os.makedirs(FIG_DIR, exist_ok=True)

apply_publication_style(font_size=16, axes_linewidth=2.5)


def _save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close('all')
    print(f'  saved  {name}')


def fig_et01_bias_map():
    csv = os.path.join(TBL_DIR, 'et0_station_metrics.csv')
    if not os.path.exists(csv):
        print(f'  SKIP et01 - {csv} not found'); return

    df = pd.read_csv(csv)
    products = [('AORC', CA), ('NLDAS', CN)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (prod, color) in zip(axes, products):
        sub = df[df['product'] == prod].dropna(subset=['lat', 'lon', 'bias', 'n'])
        if sub.empty:
            ax.text(0.5, 0.5, f'No data for {prod}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            continue

        s_min, s_max = 25, 220
        n_norm = (sub['n'] - sub['n'].min()) / (sub['n'].max() - sub['n'].min() + 1e-9)
        sizes  = s_min + n_norm * (s_max - s_min)

        v_abs  = max(abs(sub['bias'].quantile(0.02)), abs(sub['bias'].quantile(0.98)))
        v_abs  = max(v_abs, 1.0)

        sc = ax.scatter(sub['lon'], sub['lat'],
                        c=sub['bias'], s=sizes,
                        cmap='RdBu_r', vmin=-v_abs, vmax=v_abs,
                        edgecolors='k', linewidths=0.25, alpha=0.85,
                        zorder=3)
        cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=13)
        cb.set_label('ET$_0$ Bias (mm month$^{-1}$)', fontsize=14)

        for lat_line in [29, 33]:
            ax.axhline(lat_line, color=PALETTE['blue_secondary'], lw=0.6, ls='--', alpha=0.5)
        for lon_line in [-95, -90]:
            ax.axvline(lon_line, color=PALETTE['blue_secondary'], lw=0.6, ls='--', alpha=0.5)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        n_sta  = len(sub)
        b_mean = sub['bias'].mean()
        b_std  = sub['bias'].std()
        ax.set_title(f'{prod}  (n = {n_sta},  bias = {b_mean:.1f} \u00b1 {b_std:.1f} mm mo$^{{-1}}$)',
                     fontsize=14, pad=8)

    plt.tight_layout()
    _save('et01_bias_map.png')


def fig_et02_taylor_all():
    csv = os.path.join(TBL_DIR, 'et0_station_metrics.csv')
    if not os.path.exists(csv):
        print(f'  SKIP et02 - {csv} not found'); return

    df  = pd.read_csv(csv)
    req = ['r', 'obs_std', 'sim_std', 'rmse', 'product', 'n']
    df  = df.dropna(subset=req)
    df  = df[df['n'] >= 12]

    df['sigma_r'] = df['sim_std'] / df['obs_std']
    df['rmse_n']  = df['rmse']   / df['obs_std']

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='polar')

    colors_map = {'AORC': CA, 'NLDAS': CN}
    markers    = {'AORC': 'o', 'NLDAS': 's'}

    for prod in ['AORC', 'NLDAS']:
        sub = df[df['product'] == prod]
        if sub.empty: continue
        color  = colors_map[prod]
        marker = markers[prod]

        theta = np.arccos(np.clip(sub['r'].values, -1, 1))
        r_val = sub['sigma_r'].values
        ax.scatter(theta, r_val, color=color, marker=marker,
                   s=18, alpha=0.35, zorder=2,
                   label=f'{prod} (n={len(sub)})')

        theta_m = np.arccos(np.clip(sub['r'].mean(), -1, 1))
        r_m     = sub['sigma_r'].mean()
        ax.scatter(theta_m, r_m, color=color, marker=marker,
                   s=220, edgecolors='k', linewidths=2.0, zorder=5,
                   label=f'{prod} mean')

    ax.scatter(0, 1.0, marker='*', color='k', s=280, zorder=6,
               label='GHCN ET$_0$ (reference)')
    # the star is identified in the legend; an on-plot label covered the markers

    thetas  = np.linspace(0, np.pi / 2, 200)

    for r_label in [0.8, 0.9, 0.95, 0.99]:
        t = np.arccos(r_label)
        ax.plot(t, 1.75, '.', color='grey', ms=4, alpha=0.6)
        ax.text(t, 1.80, f'r={r_label}', fontsize=11,
                ha='center', va='bottom', color='grey')
        ax.plot([t, t], [0, 1.75], lw=0.5, ls=':', color='grey', alpha=0.4)

    for r_circ in [0.5, 1.0, 1.5]:
        circle_t = np.linspace(0, np.pi / 2, 300)
        ax.plot(circle_t, np.full_like(circle_t, r_circ),
                lw=0.7, ls='--', color='#aaaaaa', alpha=0.6, zorder=1)
        ax.text(np.pi / 2, r_circ, f'{r_circ:.1f}\u03c3',
                fontsize=11, ha='left', va='center', color='#aaaaaa')

    for rmse_c in [0.25, 0.50, 0.75]:
        t_rmse = np.linspace(0, np.pi / 2, 400)
        a, b_c = 1, -2 * np.cos(t_rmse)
        c_c    = 1 - rmse_c ** 2
        disc   = b_c ** 2 - 4 * a * c_c
        mask   = disc >= 0
        r_sol  = (-b_c[mask] + np.sqrt(disc[mask])) / 2
        valid  = (r_sol >= 0) & (r_sol <= 1.75)
        if valid.sum() > 1:
            ax.plot(t_rmse[mask][valid], r_sol[valid],
                    lw=0.6, ls=':', color='#888888', alpha=0.5, zorder=1)
            # label at the middle of the arc, clear of the station cloud and the axis
            vi = valid.nonzero()[0]
            idx = vi[int(len(vi) * (0.6 if rmse_c > 0.7 else 0.4))]
            ax.text(t_rmse[mask][idx], r_sol[idx],
                    f'RMSE={rmse_c}', fontsize=9, color='#888888', ha='left', va='center')

    ax.set_thetamax(90)
    ax.set_rlim(0, 1.75)
    ax.set_xlabel('Normalised standard deviation ($\\sigma_{sim}$ / $\\sigma_{obs}$)',
                  fontsize=14, labelpad=15)
    ax.set_rlabel_position(45)
    ax.tick_params(labelsize=13)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right',
              bbox_to_anchor=(1.42, 1.05), fontsize=13, frameon=False)

    plt.tight_layout()
    _save('et02_taylor_all.png')


if __name__ == '__main__':
    print(f'Regenerating ET0 station-validation figures -> {FIG_DIR}\n')
    fig_et01_bias_map()
    fig_et02_taylor_all()
    print('\nDone.')
