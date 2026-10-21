"""
stats.py — Statistical tools for NLDAS vs AORC comparison
==========================================================
Includes:
  - Taylor diagram
  - Error PDFs (KDE)
  - Bias metrics (RMSE, r, MAE, PBias)
  - Seasonal splits (DJF/MAM/JJA/SON)
  - Mann-Kendall trend test + Sen's slope
  - Pettitt change-point detection
  - Circular statistics for seasonal composition analysis
    (circular mean DOY, mean resultant length R, circular variance)
  - Polar climatology rose plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as spstats
from scipy.stats import gaussian_kde


# ══════════════════════════════════════════════════════════════════════════════
# Basic metrics
# ══════════════════════════════════════════════════════════════════════════════

def bias_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute RMSE, Pearson r, MAE, and PBias.
    obs  = reference (AORC), pred = evaluated (NLDAS).
    """
    mask = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[mask], pred[mask]
    if len(o) < 3:
        return dict(n=0, rmse=np.nan, r=np.nan, mae=np.nan, pbias=np.nan, bias=np.nan)
    rmse  = float(np.sqrt(np.mean((p - o)**2)))
    r     = float(np.corrcoef(o, p)[0, 1])
    mae   = float(np.mean(np.abs(p - o)))
    bias  = float(np.mean(p - o))
    pbias = float(bias / np.mean(np.abs(o)) * 100) if np.mean(np.abs(o)) != 0 else np.nan
    return dict(n=int(mask.sum()), rmse=rmse, r=r, mae=mae, pbias=pbias, bias=bias)


def deseasonalize(series: pd.Series, window_days: int = 31) -> pd.Series:
    """
    Daily anomalies relative to a smoothed day-of-year climatology.

    Correlations and Taylor statistics of raw daily series are dominated by the
    shared annual cycle (temperature r ≈ 1.00 regardless of day-to-day skill);
    anomalies measure agreement in the weather itself.
    """
    s = series.astype(float)
    doy = np.minimum(np.asarray(pd.DatetimeIndex(s.index).dayofyear), 365)
    clim = s.groupby(doy).mean().reindex(range(1, 366))
    # circular smoothing of the climatology
    pad = window_days // 2
    ext = pd.concat([clim.iloc[-pad:], clim, clim.iloc[:pad]])
    smooth = ext.rolling(window_days, center=True, min_periods=1).mean().iloc[pad:pad + 365]
    smooth.index = range(1, 366)
    return s - smooth.reindex(doy).values


def anomaly_correlation(obs: pd.Series, pred: pd.Series) -> float:
    """Pearson r of deseasonalized daily anomalies on common dates."""
    both = pd.DataFrame({'o': obs, 'p': pred}).dropna()
    if len(both) < 30:
        return np.nan
    a = deseasonalize(both['o']); b = deseasonalize(both['p'])
    m = np.isfinite(a.values) & np.isfinite(b.values)
    return float(np.corrcoef(a.values[m], b.values[m])[0, 1]) if m.sum() > 3 else np.nan


def taylor_stats(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Correlation and normalized std / centered RMSE of pred against ITS OWN reference."""
    m = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[m], pred[m]
    if len(o) < 3 or np.std(o) == 0:
        return dict(r=np.nan, std_ratio=np.nan, crmse_norm=np.nan)
    r = float(np.corrcoef(o, p)[0, 1])
    sd = float(np.std(p) / np.std(o))
    crmse = float(np.sqrt(np.mean(((p - p.mean()) - (o - o.mean())) ** 2)) / np.std(o))
    return dict(r=r, std_ratio=sd, crmse_norm=crmse)


def taylor_diagram_normalized(ax, points: list[tuple], colors: list | None = None):
    """
    Normalized Taylor diagram: each point is (label, r, std_ratio) computed against its
    own reference, so variables with different units share one diagram. The reference
    sits at (r = 1, σ_n = 1).
    """
    colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
    smax = max([1.6] + [p[2] * 1.15 for p in points if np.isfinite(p[2])])
    for crmse in [0.25, 0.5, 0.75, 1.0]:
        th = np.linspace(0, np.pi, 300)
        x = 1 + crmse * np.cos(th); y = crmse * np.sin(th)
        rr = np.hypot(x, y); tt = np.arctan2(y, x)
        ok = (tt >= 0) & (tt <= np.pi / 2) & (rr <= smax)
        ax.plot(tt[ok], rr[ok], color='grey', ls='--', lw=0.6, alpha=0.6)
    ax.plot(0, 1, 'k*', markersize=14, label='Reference (AORC)')
    for i, (lbl, r, sd) in enumerate(points):
        if not (np.isfinite(r) and np.isfinite(sd)):
            continue
        ax.plot(np.arccos(np.clip(r, -1, 1)), sd, 'o', color=colors[i % len(colors)],
                markersize=9, label=f'{lbl}  r={r:.2f}  σn={sd:.2f}')
    ax.set_thetamax(90)
    ax.set_rlim(0, smax)
    ax.set_thetagrids([0, 15, 30, 45, 60, 75, 90],
                      labels=['1.0', '0.97', '0.87', '0.71', '0.50', '0.26', '0.0'])
    ax.set_xlabel('Correlation coefficient', labelpad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), fontsize=8)


def seasonal_split(df: pd.DataFrame) -> dict:
    """Return dict with keys DJF/MAM/JJA/SON, values = sub-DataFrames."""
    m = df.index.month
    return {
        'DJF': df[m.isin([12, 1, 2])],
        'MAM': df[m.isin([3,  4, 5])],
        'JJA': df[m.isin([6,  7, 8])],
        'SON': df[m.isin([9, 10, 11])],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Taylor Diagram
# ══════════════════════════════════════════════════════════════════════════════

def taylor_diagram(ax, ref: np.ndarray, sims: list[tuple], title: str = '',
                   max_std: float | None = None, colors: list | None = None):
    """
    Draw a Taylor diagram on a polar axes.

    Parameters
    ----------
    ax    : matplotlib polar Axes
    ref   : 1-D reference array (AORC)
    sims  : list of (label, array) tuples (e.g. [('NLDAS', nldas_arr)])
    title : plot title
    max_std : max std dev for axis limit (auto if None)
    """
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    std_ref = float(np.std(ref))

    std_max = max_std or (std_ref * 1.6)

    colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Draw arcs for RMSE contours
    for rmse_val in [0.25, 0.5, 0.75, 1.0, 1.25]:
        rmse_abs = rmse_val * std_ref
        theta = np.linspace(0, np.pi / 2, 200)
        # arc centered on (std_ref, 0) in Cartesian = (0, std_ref) in polar...
        # Use Cartesian overlay
        xc = std_ref + rmse_abs * np.cos(theta + np.pi)
        yc =           rmse_abs * np.sin(theta + np.pi)
        r_arc   = np.sqrt(xc**2 + yc**2)
        th_arc  = np.arctan2(yc, xc)
        mask    = (th_arc >= 0) & (th_arc <= np.pi / 2) & (r_arc <= std_max)
        if mask.any():
            ax.plot(th_arc[mask], r_arc[mask], 'k--', lw=0.4, alpha=0.4)

    # std circles
    for sf in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        c = plt.Circle((0, 0), sf * std_ref, transform=ax.transData._b,
                        fill=False, color='grey', lw=0.4, linestyle=':')

    # Reference point (AORC)
    ax.plot(0, std_ref, 'k*', markersize=12, label='AORC (ref)')

    # Evaluated points
    for i, (lbl, arr) in enumerate(sims):
        arr = np.asarray(arr, dtype=float)
        both = np.isfinite(ref) & np.isfinite(arr[:len(ref)]) if len(arr) >= len(ref) else None
        if both is None or both.sum() < 3:
            continue
        r_val  = float(np.corrcoef(ref[both], arr[:len(ref)][both])[0, 1])
        std_s  = float(np.std(arr[:len(ref)][both]))
        theta_ = np.arccos(np.clip(r_val, -1, 1))
        ax.plot(theta_, std_s, 'o', color=colors[i % len(colors)],
                markersize=9, label=f'{lbl}  r={r_val:.2f}  σ={std_s:.2f}')

    # Reference std line
    ax.axvline(0, color='k', lw=0.8)
    ax.plot([0, np.pi / 2], [std_ref, std_ref], 'k--', lw=0.6)

    ax.set_thetamax(90)
    ax.set_rlim(0, std_max)
    ax.set_thetagrids([0, 15, 30, 45, 60, 75, 90],
                      labels=['1.0','0.97','0.87','0.71','0.50','0.26','0.0'])
    ax.set_xlabel('Correlation coefficient', labelpad=20)
    # removed set_title
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)


# ══════════════════════════════════════════════════════════════════════════════
# Error PDF
# ══════════════════════════════════════════════════════════════════════════════

def error_pdf(ax, errors: np.ndarray, label: str = '', color: str = 'steelblue'):
    """
    Plot KDE + Gaussian overlay of error distribution.
    errors = pred − ref (or ref − pred; just be consistent).
    """
    errors = errors[np.isfinite(errors)]
    if len(errors) < 10:
        return
    if np.std(errors) < 1e-9 * max(1.0, abs(float(np.mean(errors)))):   # constant offset: no density
        ax.axvline(float(np.mean(errors)), color=color, lw=2, label=f'{label} (constant)')
        ax.axvline(0, color='k', lw=0.8, linestyle=':')
        return
    kde = gaussian_kde(errors, bw_method='scott')
    x   = np.linspace(errors.min(), errors.max(), 300)
    ax.plot(x, kde(x), color=color, lw=2, label=f'{label} KDE')
    # Gaussian fit
    mu, sigma = float(errors.mean()), float(errors.std())
    gauss = spstats.norm.pdf(x, mu, sigma)
    ax.plot(x, gauss, color=color, lw=1, linestyle='--', alpha=0.7, label=f'Gaussian $\mu$={mu:.3f}')
    ax.axvline(0, color='k', lw=0.8, linestyle=':')


# ══════════════════════════════════════════════════════════════════════════════
# Trend analysis
# ══════════════════════════════════════════════════════════════════════════════

def mann_kendall(series: pd.Series) -> dict:
    """
    Mann-Kendall trend test + Sen's slope.
    Returns: trend ('increasing'/'decreasing'/'no trend'), p_value, slope, intercept.
    """
    x = series.dropna().values
    n = len(x)
    if n < 4:
        return dict(trend='insufficient data', p_value=np.nan, slope=np.nan, intercept=np.nan)

    # S statistic
    S = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            S += np.sign(x[j] - x[k])

    # Variance of S
    unique, counts = np.unique(x, return_counts=True)
    tie_sum = sum(c * (c - 1) * (2 * c + 5) for c in counts if c > 1)
    var_S = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18.0

    if S > 0:
        z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_S)
    else:
        z = 0.0

    p_val = 2 * (1 - spstats.norm.cdf(abs(z)))

    # Sen's slope
    slopes = []
    for k in range(n - 1):
        for j in range(k + 1, n):
            slopes.append((x[j] - x[k]) / (j - k))
    slope = float(np.median(slopes))

    # Intercept
    t = np.arange(n)
    intercept = float(np.median(x) - slope * np.median(t))

    trend = 'no trend'
    if p_val < 0.05:
        trend = 'increasing' if S > 0 else 'decreasing'

    return dict(trend=trend, p_value=round(p_val, 4),
                slope=round(slope, 6), intercept=round(intercept, 4), z=round(z, 3))


def pettitt_test(series: pd.Series) -> dict:
    """
    Pettitt change-point test.
    Returns: change_point_index, change_point_date, p_value, U_statistic.
    """
    x = series.dropna()
    n = len(x)
    if n < 10:
        return dict(change_point_index=None, change_point_date=None, p_value=np.nan)
    vals = x.values.astype(float)
    # Pettitt (1979): U_t = Σ_{i≤t} Σ_{j>t} sgn(x_j − x_i).
    # The earlier version summed signs of consecutive differences instead, so K stayed
    # tiny relative to n³ and every series returned p ≈ 1.
    D = np.sign(vals[None, :] - vals[:, None])            # D[i, j] = sgn(x_j − x_i)
    upper = np.triu(D, k=1)                               # keep j > i
    # U_t = Σ_{i=0..t} Σ_{j=t+1..n-1} D[i, j]
    row_cum = np.cumsum(upper, axis=0)                    # Σ_{i≤t} D[i, j]
    U = np.array([row_cum[t, t + 1:].sum() for t in range(n - 1)])
    K   = float(np.max(np.abs(U)))
    idx = int(np.argmax(np.abs(U)))                       # last index of the first segment
    p   = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
    return dict(change_point_index=idx,
                change_point_date=x.index[idx] if hasattr(x.index[idx], 'date') else idx,
                p_value=round(float(np.clip(p, 0, 1)), 4),
                K=round(K, 2))


# ══════════════════════════════════════════════════════════════════════════════
# Circular Statistics — Seasonal Composition
# ══════════════════════════════════════════════════════════════════════════════

def to_circular_angle(doy: np.ndarray) -> np.ndarray:
    """Convert day of year (1–365/366) to angle in radians [0, 2π]."""
    return 2 * np.pi * (doy - 1) / 365.0


def circular_mean_doy(weights: np.ndarray, doys: np.ndarray) -> dict:
    """
    Compute circular mean day of year, mean resultant length R, and circular variance.

    Parameters
    ----------
    weights : daily values (e.g. precipitation amounts) used as weights
    doys    : corresponding day-of-year values

    Returns
    -------
    dict with: mean_doy, R (concentration, 0=uniform, 1=highly concentrated),
               circular_variance, circular_std_days
    """
    weights = np.asarray(weights, dtype=float)
    doys    = np.asarray(doys,    dtype=float)
    mask    = np.isfinite(weights) & np.isfinite(doys) & (weights > 0)
    if mask.sum() < 5:
        return dict(mean_doy=np.nan, R=np.nan, circular_variance=np.nan, circular_std_days=np.nan)

    w = weights[mask]
    theta = to_circular_angle(doys[mask])

    # Weighted circular mean
    C = np.sum(w * np.cos(theta)) / np.sum(w)
    S = np.sum(w * np.sin(theta)) / np.sum(w)
    R = float(np.sqrt(C**2 + S**2))           # mean resultant length
    mean_angle = float(np.arctan2(S, C))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    mean_doy = float(mean_angle / (2 * np.pi) * 365 + 1)

    circ_var = float(1 - R)
    circ_std_days = float(np.sqrt(-2 * np.log(R)) * 365 / (2 * np.pi)) if R > 0 else np.nan

    return dict(
        mean_doy         = round(mean_doy, 1),
        R                = round(R, 4),
        circular_variance= round(circ_var, 4),
        circular_std_days= round(circ_std_days, 1),
    )


def circular_stats_by_variable(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """
    Compute circular seasonal statistics for each variable × dataset (AORC/NLDAS).

    For precipitation: raw values are used as weights (zeros on dry days are meaningful).
    For all other variables: positive anomalies from the series mean are used as weights,
    so the circular mean identifies the season when the variable is above its annual mean.
    Using raw values for temperature (≈290 K) or humidity would make winter nearly as
    'heavy' as summer, producing an arbitrary mean that depends on unit offsets, not on
    where the seasonal peak actually falls.

    Returns a DataFrame summarizing mean_doy, R, circular_variance per variable per dataset.
    """
    doys = df.index.day_of_year.values
    records = []
    for var in variables:
        for ds in ['aorc', 'nldas']:
            col = f'{var}_{ds}'
            if col not in df.columns:
                continue
            vals = df[col].values.astype(float)
            if var == 'precip':
                # For precip, raw values are the correct weights (0 = dry day)
                weights = np.maximum(vals, 0)
            else:
                # For all other variables, use positive anomalies so the mean
                # reflects where in the year the variable is elevated
                weights = np.maximum(vals - np.nanmean(vals), 0)
            cs = circular_mean_doy(weights, doys)
            records.append({
                'variable': var,
                'dataset' : ds.upper(),
                **cs,
            })
    return pd.DataFrame(records)


def polar_climatology_plot(ax, df: pd.DataFrame, col: str, label: str,
                            color: str = 'steelblue', bins: int = 12):
    """
    Draw a polar rose/bar plot showing the seasonal distribution of a variable.

    ax    : matplotlib polar Axes
    df    : DataFrame indexed by date
    col   : column name (e.g. 'precip_aorc')
    label : legend label
    bins  : number of angular bins (12 = monthly)
    """
    if col not in df.columns:
        return
    vals = df[col].dropna()
    
    # Calculate anomaly mass as per circular_stats logic
    if col.startswith('precip'):
        weights = np.maximum(vals, 0)
    else:
        weights = np.maximum(vals - np.nanmean(vals), 0)
        
    months = vals.index.month.values
    month_sums = np.array([
        weights[months == m].sum() if (months == m).any() else 0
        for m in range(1, 13)
    ])

    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    width  = 2 * np.pi / 12

    bars = ax.bar(angles, month_sums, width=width * 0.8,
                  color=color, alpha=0.75, label=label, align='edge')

    ax.set_xticks(angles)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=14)
    ax.tick_params(axis='x', pad=10)
    ax.set_yticklabels([]) # Hide radial values to prevent overlapping with bars
    ax.set_theta_zero_location('N')   # January at top
    ax.set_theta_direction(-1)        # clockwise

    # Annotate circular mean DOY — same weights as circular_stats_by_variable
    # (raw precipitation; positive anomalies for every other variable)
    cs = circular_mean_doy(np.asarray(weights, dtype=float), vals.index.day_of_year.values)
    if not np.isnan(cs['mean_doy']):
        mean_angle = (cs['mean_doy'] - 1) / 365.0 * 2 * np.pi
        ax.plot([0, mean_angle], [0, ax.get_ylim()[1] * 1.05],
                color=color, lw=2.5, linestyle='--', alpha=0.9)
        # Add DOY text to the bottom corners to prevent overlapping!
        if 'aorc' in col.lower():
            ax.annotate(f"DOY: {cs['mean_doy']:.0f}", xy=(0.0, -0.05), xycoords='axes fraction', 
                        color=color, fontsize=12, ha='left', va='top', fontweight='bold')
        else:
            ax.annotate(f"DOY: {cs['mean_doy']:.0f}", xy=(1.0, -0.05), xycoords='axes fraction', 
                        color=color, fontsize=12, ha='right', va='top', fontweight='bold')
