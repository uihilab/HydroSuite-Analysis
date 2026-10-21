"""
regen_phase7.py — Phase 7 spatial coherence maps (means and bias) on the common footprint.

Reads the full-period, common-footprint maps written by run_pipeline.py
(results/<period>/gridpass_v2/spatial.npz). The earlier version averaged every 20th
file of each product separately, read NLDAS-3 precipitation without unit
conversion. Layout, colour maps, scaling and the AORC − NLDAS bias convention are
unchanged from the original figure.
"""
import os, sys, glob, re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pipeline.fig_style import apply_publication_style, CA, CN, PALETTE, VAR_LABELS
apply_publication_style(font_size=16, axes_linewidth=2.5)

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from ext_config import EXT
FIG_DIR   = os.path.join(EXT.RESULTS_ROOT, 'full_2001_2020', 'final_figures')
STATES_PATH = os.path.join(ROOT, 'data', 'raw', 'basins', 'us_states_simple.geojson')

# Map extent: the common footprint spans 106.5-79.0 W, 25.8-36.5 N
GULF_LAT = (25.0, 37.0)
GULF_LON = (-107.0, -78.5)


def _save(name):
    tmp = os.path.join(FIG_DIR, '_tmp_' + name)
    plt.savefig(tmp, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close('all')
    final = os.path.join(FIG_DIR, name)
    try: os.remove(final)
    except: pass
    os.rename(tmp, final)
    print(f'  saved  {name}')


def compute_and_cache(var='precip', period='full_2001_2020'):
    from ext_config import EXT
    from pipeline.gridpass import load_gridpass
    gp = load_gridpass(os.path.join(EXT.RESULTS_ROOT, period, 'gridpass_v2'))
    if gp is None or var not in gp.get('maps', {}):
        print(f'  ERROR: no gridpass maps for {var} — run scripts/run_pipeline.py --full first')
        return None
    st = gp['maps'][var]
    off = 273.15 if var == 'temp' else 0.0
    # gridpass stores NLDAS − AORC; the figure keeps its original AORC − NLDAS convention
    return gp['map_lats'], gp['map_lons'], st['mean_aorc'] - off, st['mean_nldas'] - off, -st['bias']


def plot_gulf(var, tgt_lats, tgt_lons, aorc, nldas, bias):
    var_label = VAR_LABELS.get(var, var)

    # Clip to Gulf States
    lat_mask = (tgt_lats >= GULF_LAT[0]) & (tgt_lats <= GULF_LAT[1])
    lon_mask = (tgt_lons >= GULF_LON[0]) & (tgt_lons <= GULF_LON[1])
    lats_c = tgt_lats[lat_mask]
    lons_c = tgt_lons[lon_mask]
    aorc_c  = aorc[np.ix_(lat_mask, lon_mask)]
    nldas_c = nldas[np.ix_(lat_mask, lon_mask)]
    bias_c  = bias[np.ix_(lat_mask, lon_mask)]

    lon_g, lat_g = np.meshgrid(lons_c, lats_c)
    fin_b = bias_c[np.isfinite(bias_c)]
    vabs = np.nanpercentile(np.abs(fin_b), 95) if len(fin_b) > 0 else 1.0

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 4, hspace=0.15, wspace=0.15)
    ax0 = fig.add_subplot(gs[0, 0:2])   # AORC top-left
    ax1 = fig.add_subplot(gs[0, 2:4])   # NLDAS top-right
    ax2 = fig.add_subplot(gs[1, 1:3])   # Bias bottom-center

    for ax, data, title, cmap in [
        (ax0, aorc_c,  'AORC',                   'viridis'),
        (ax1, nldas_c, 'NLDAS',                  'viridis'),
        (ax2, bias_c,  'Bias (AORC \u2212 NLDAS)', 'RdBu_r'),
    ]:
        fin = data[np.isfinite(data)]
        vmin, vmax = ((-vabs, vabs) if 'Bias' in title
                      else (float(np.nanmin(fin)), float(np.nanmax(fin))) if fin.size else (0, 1))
        im = ax.pcolormesh(lon_g, lat_g, data, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading='auto')
        cb = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.10, aspect=30)
        cb.ax.tick_params(labelsize=9)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=6)
        ax.tick_params(labelsize=10)

        # State borders
        try:
            if os.path.exists(STATES_PATH):
                with open(STATES_PATH) as f:
                    gj = json.load(f)
                for feat in gj['features']:
                    geom = feat['geometry']
                    coords_list = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
                    for poly in coords_list:
                        for ring in poly:
                            xs = [c[0] for c in ring]
                            ys = [c[1] for c in ring]
                            ax.plot(xs, ys, 'k-', lw=0.6, alpha=0.65)
        except Exception:
            pass

        ax.set_xlim(GULF_LON)
        ax.set_ylim(GULF_LAT)

    fig.savefig(os.path.join(FIG_DIR, '_tmp_phase7_spatial_coherence_' + var + '.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close('all')
    final = os.path.join(FIG_DIR, f'phase7_spatial_coherence_{var}.png')
    try: os.remove(final)
    except: pass
    os.rename(os.path.join(FIG_DIR, '_tmp_phase7_spatial_coherence_' + var + '.png'), final)
    print(f'  saved  phase7_spatial_coherence_{var}.png')


if __name__ == '__main__':
    for var in sys.argv[1:] or ['precip']:
        result = compute_and_cache(var)
        if result is not None:
            plot_gulf(var, *result)
        print('\nDone.')
