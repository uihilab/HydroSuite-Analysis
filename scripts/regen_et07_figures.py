"""
fix_three_figures.py
====================
Targeted fixes for specific figures without re-running any pipeline:

  1. et07_method_gap_climatology.png  — rebuild from D-drive CSV, strip all titles
  2. phase7_spatial_coherence_precip.png — clip AORC panel to Southern US domain,
                                          remove all panel titles and suptitle

Run from project root:
    python scripts/fix_three_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from ext_config import EXT
FIG_DIR = os.path.join(EXT.RESULTS_ROOT, 'full_2001_2020', 'final_figures')
DPI = 300
ET07_PERIOD = 'full_2001_2020'      # results/comparisons/<period>/tables

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline.fig_style import apply_publication_style, CA, CN, PALETTE

apply_publication_style(font_size=16, axes_linewidth=2.5)

GHCN_C = PALETTE['green_3']
ML     = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MOS    = np.arange(1, 13)


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — et07_method_gap_climatology.png
#   Rebuild from the pre-computed D-drive CSV — no titles, consistent style.
# ══════════════════════════════════════════════════════════════════════════════
def fix_et07():
    from ext_config import EXT          # HYDRO_DATA_DIR → results/comparisons/<period>/tables
    csv = os.path.join(EXT.RESULTS_ROOT, 'comparisons', ET07_PERIOD, 'tables',
                       'et07_method_comparison.csv')
    if not os.path.exists(csv):
        print(f'  [SKIP] et07 CSV not found: {csv}')
        return

    tbl = pd.read_csv(csv)

    def _a(col):
        return tbl[col].values.astype(float) if col in tbl.columns else np.full(12, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) All five estimates
    ax = axes[0, 0]
    ax.plot(MOS, _a('ghcn_hs'),  color=GHCN_C, lw=2.5, marker='D', ms=8,
            label='GHCN HS-ET$_0$ (obs)')
    ax.plot(MOS, _a('aorc_hs'),  color=CA,     lw=2.5, marker='o', ms=8, ls='--',
            label='AORC HS-ET$_0$')
    ax.plot(MOS, _a('nldas_hs'), color=CN,     lw=2.5, marker='s', ms=8, ls='--',
            label='NLDAS HS-ET$_0$')
    ax.plot(MOS, _a('aorc_pm'),  color=CA,     lw=3.0, marker='o', ms=8,
            label='AORC PM-ET$_0$')
    ax.plot(MOS, _a('nldas_pm'), color=CN,     lw=3.0, marker='s', ms=8,
            label='NLDAS PM-ET$_0$')
    ax.set_xticks(MOS); ax.set_xticklabels(ML, rotation=30)
    ax.set_ylabel('ET$_0$ (mm day$^{-1}$)')
    ax.legend(fontsize=13, frameon=False)

    # (0,1) Method gap bars (PM - HS, same dataset)
    ax = axes[0, 1]
    ax.bar(MOS - 0.2, _a('aorc_method_gap'),  0.35, color=CA, alpha=0.82,
           edgecolor='black', linewidth=1.5, label='AORC: PM \u2212 HS')
    ax.bar(MOS + 0.2, _a('nldas_method_gap'), 0.35, color=CN, alpha=0.82,
           edgecolor='black', linewidth=1.5, label='NLDAS: PM \u2212 HS')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(MOS); ax.set_xticklabels(ML, rotation=30)
    ax.set_ylabel('\u0394 ET$_0$ (mm day$^{-1}$)')
    ax.legend(fontsize=13, frameon=False)

    # (0,2) Forcing gap (AORC - NLDAS, same method)
    ax = axes[0, 2]
    ax.bar(MOS - 0.2, _a('forcing_gap_hs'), 0.35, color=PALETTE['blue_secondary'], alpha=0.82,
           edgecolor='black', linewidth=1.5, label='via HS')
    ax.bar(MOS + 0.2, _a('forcing_gap_pm'), 0.35, color=PALETTE['violet'], alpha=0.82,
           edgecolor='black', linewidth=1.5, label='via PM')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(MOS); ax.set_xticklabels(ML, rotation=30)
    ax.set_ylabel('AORC \u2212 NLDAS ET$_0$ (mm day$^{-1}$)')
    ax.legend(fontsize=13, frameon=False)

    # (1,0) AORC: PM vs HS vs GHCN
    ax = axes[1, 0]
    ax.plot(MOS, _a('aorc_pm'),  color=CA,     lw=3.0, marker='o', ms=8,
            label='AORC PM-ET$_0$')
    ax.plot(MOS, _a('aorc_hs'),  color=CA,     lw=2.5, marker='o', ms=8, ls='--',
            label='AORC HS-ET$_0$')
    ax.plot(MOS, _a('ghcn_hs'),  color=GHCN_C, lw=2.5, marker='D', ms=8,
            label='GHCN HS-ET$_0$ (obs)')
    ax.fill_between(MOS, _a('aorc_pm'), _a('aorc_hs'), alpha=0.12, color=CA)
    ax.set_xticks(MOS); ax.set_xticklabels(ML, rotation=30)
    ax.set_ylabel('ET$_0$ (mm day$^{-1}$)')
    ax.set_xlabel('AORC forcing')
    ax.legend(fontsize=13, frameon=False)

    # (1,1) NLDAS: PM vs HS vs GHCN
    ax = axes[1, 1]
    ax.plot(MOS, _a('nldas_pm'), color=CN,     lw=3.0, marker='s', ms=8,
            label='NLDAS PM-ET$_0$')
    ax.plot(MOS, _a('nldas_hs'), color=CN,     lw=2.5, marker='s', ms=8, ls='--',
            label='NLDAS HS-ET$_0$')
    ax.plot(MOS, _a('ghcn_hs'),  color=GHCN_C, lw=2.5, marker='D', ms=8,
            label='GHCN HS-ET$_0$ (obs)')
    ax.fill_between(MOS, _a('nldas_pm'), _a('nldas_hs'), alpha=0.12, color=CN)
    ax.set_xticks(MOS); ax.set_xticklabels(ML, rotation=30)
    ax.set_ylabel('ET$_0$ (mm day$^{-1}$)')
    ax.set_xlabel('NLDAS forcing')
    ax.legend(fontsize=13, frameon=False)

    # (1,2) PM/HS ratio
    ax = axes[1, 2]
    a_safe = np.where(_a('aorc_hs')  > 0.01, _a('aorc_hs'),  np.nan)
    n_safe = np.where(_a('nldas_hs') > 0.01, _a('nldas_hs'), np.nan)
    ax.plot(MOS, _a('aorc_pm')  / a_safe, color=CA, lw=2.5, marker='o', ms=8,
            label='AORC PM/HS')
    ax.plot(MOS, _a('nldas_pm') / n_safe, color=CN, lw=2.5, marker='s', ms=8,
            label='NLDAS PM/HS')
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.6)
    ax.set_xticks(MOS); ax.set_xticklabels(ML, rotation=30)
    ax.set_ylabel('PM-ET$_0$ / HS-ET$_0$')
    ax.set_ylim(0, 3)
    ax.set_xlabel('Month')
    ax.legend(fontsize=13, frameon=False)

    for a in axes.flat:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    out = os.path.join(FIG_DIR, 'et07_method_gap_climatology.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print('  saved  et07_method_gap_climatology.png')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Fixing figures -> {FIG_DIR}\n')
    fix_et07()
    print('\nDone.')
