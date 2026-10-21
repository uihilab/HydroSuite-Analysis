"""
extremes.py — Pixel-scale precipitation extremes and intensity statistics
==========================================================================
Annual maxima of a domain-mean series describe a 25° × 11° spatial average, which
smooths away the convective extremes of interest. These helpers work on the
daily series of individual pixels (sampled by pipeline.gridpass) and summarize the
distribution of pixel-level results across the domain.

  * GEV on annual block maxima per pixel (maximum likelihood, scipy)
  * Peaks-over-threshold with a Generalized Pareto fit (98th percentile threshold,
    runs declustering) — the POT sensitivity check described in the methods
  * per-pixel wet-day intensity percentiles and seasonal intermittency
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import genextreme, genpareto

warnings.filterwarnings('ignore', category=RuntimeWarning)
RETURN_PERIODS = (2, 5, 10, 25)          # 20 annual maxima do not support longer ones


def annual_maxima(daily: pd.DataFrame, min_days: int = 300) -> pd.DataFrame:
    """Annual maxima per column; years with fewer than ``min_days`` valid days → NaN."""
    g = daily.groupby(daily.index.year)
    amax = g.max()
    amax[g.count() < min_days] = np.nan
    return amax


def gev_fit_lmom(x) -> tuple[float, float, float]:
    """
    GEV parameters by L-moments (Hosking, 1990; Hosking & Wallis, 1997), returned in
    scipy's genextreme convention (c, loc, scale); c > 0 is a bounded upper tail.

    With ~20 annual maxima, maximum likelihood (scipy genextreme.fit) is unstable: on
    the NLDAS-3 domain-mean maxima it put the upper bound at the sample maximum and
    gave a 2-yr level (24.4 mm) above 19 of the 20 observed maxima. L-moments are the
    standard estimator for short hydrologic records and cannot degenerate that way.
    """
    from math import gamma, log
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    n = len(x)
    if n < 5 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    t3 = l3 / l2
    z = 2 / (3 + t3) - log(2) / log(3)
    k = 7.8590 * z + 2.9554 * z ** 2                    # Hosking's k == scipy's c
    if abs(k) < 1e-6:                                   # Gumbel limit
        scale = l2 / log(2); loc = l1 - 0.5772156649 * scale
        return 0.0, loc, scale
    scale = l2 * k / ((1 - 2 ** (-k)) * gamma(1 + k))
    loc = l1 - scale * (1 - gamma(1 + k)) / k
    return k, loc, scale


def gev_quantiles(x: np.ndarray, T=RETURN_PERIODS) -> np.ndarray:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 10 or np.ptp(x) == 0:
        return np.full(len(T), np.nan)
    c, loc, scale = gev_fit_lmom(x)
    if not np.isfinite(scale) or scale <= 0:
        return np.full(len(T), np.nan)
    return genextreme.ppf(1 - 1 / np.asarray(T, float), c, loc, scale)


def pot_gpd(x: pd.Series, q: float = 0.98, run_length: int = 1, T=RETURN_PERIODS) -> dict:
    """
    Peaks over the ``q`` quantile of the full daily record, declustered with runs of
    ``run_length`` dry-of-threshold days, fitted with a GPD (location fixed at 0).
    Return level: x_T = u + σ/ξ · ((λT)^ξ − 1), λ = clusters per year.
    """
    s = x.dropna()
    if len(s) < 365 * 5:
        return {}
    u = float(np.quantile(s.values, q))
    exc = s > u
    peaks, cur, gap = [], None, 0
    for val, e in zip(s.values, exc.values):
        if e:
            cur = val if cur is None else max(cur, val); gap = 0
        elif cur is not None:
            gap += 1
            if gap >= run_length:
                peaks.append(cur); cur = None
    if cur is not None:
        peaks.append(cur)
    peaks = np.asarray(peaks) - u
    if len(peaks) < 15:
        return dict(threshold=u, n_clusters=len(peaks))
    xi, _, sigma = genpareto.fit(peaks, floc=0)
    years = len(s) / 365.25
    lam = len(peaks) / years
    out = dict(threshold=round(u, 3), n_clusters=len(peaks), rate_per_year=round(lam, 3),
               xi=round(float(xi), 4), sigma=round(float(sigma), 4))
    for t in T:
        m = lam * t
        rl = u + (sigma / xi * (m ** xi - 1) if abs(xi) > 1e-6 else sigma * np.log(m))
        out[f'T{t}yr_mm'] = round(float(rl), 3)
    return out


def pixel_gev_summary(samples: dict, T=RETURN_PERIODS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit GEV per sampled pixel for both products.
    Returns (per-pixel table, summary table with median / IQR across pixels and the
    median of the paired NLDAS/AORC ratio).
    """
    rows = []
    am = {p: annual_maxima(samples[p]) for p in ('aorc', 'nldas')}
    for px in samples['aorc'].columns:
        qa = gev_quantiles(am['aorc'][px].values, T)
        qn = gev_quantiles(am['nldas'][px].values, T)
        rec = dict(pixel=px, n_years=int(am['aorc'][px].notna().sum()))
        for t, a, n in zip(T, qa, qn):
            rec[f'aorc_T{t}'] = a; rec[f'nldas_T{t}'] = n
        rows.append(rec)
    per_px = pd.DataFrame(rows).merge(samples['coords'], on='pixel', how='left')

    summ = []
    for t in T:
        a, n = per_px[f'aorc_T{t}'], per_px[f'nldas_T{t}']
        ok = a.notna() & n.notna() & (a > 0)
        ratio = (n[ok] / a[ok])
        for ds, s in (('AORC', a), ('NLDAS', n)):
            summ.append(dict(dataset=ds, T_yr=t, n_pixels=int(s.notna().sum()),
                             median_mm=round(float(s.median()), 2),
                             p25_mm=round(float(s.quantile(.25)), 2),
                             p75_mm=round(float(s.quantile(.75)), 2),
                             median_ratio_nldas_over_aorc=round(float(ratio.median()), 3)))
    return per_px, pd.DataFrame(summ)


def pixel_intensity_summary(samples: dict, threshold: float = 1.0) -> pd.DataFrame:
    """Per-pixel P90 wet-day intensity, fraction from intense days and total; medians across pixels."""
    rows = []
    for p in ('aorc', 'nldas'):
        d = samples[p]
        wet = d.where(d >= threshold)
        p90 = wet.quantile(0.90)
        frac = d.where(d.ge(p90, axis=1)).sum() / d.sum()
        tot = d.sum(min_count=300)
        rows.append(dict(dataset=p, P90_intensity_mm=float(p90.median()),
                         frac_intense=float(frac.median()), total_mm=float(tot.median())))
    return pd.DataFrame(rows).set_index('dataset')


def pixel_seasonal_intermittency(samples: dict, threshold: float = 1.0) -> dict[str, pd.DataFrame]:
    """Seasonal intermittency per pixel, summarized as the median across sampled pixels."""
    from pipeline.spectral import intermittency_stats
    smap = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
            6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
    out = {}
    for p in ('aorc', 'nldas'):
        d = samples[p]
        season = d.index.month.map(smap)
        rows = []
        for s in ('DJF', 'MAM', 'JJA', 'SON'):
            sub = d[season == s]
            per = [intermittency_stats(sub[c], threshold) for c in sub.columns]
            per = pd.DataFrame([x for x in per if x])
            if per.empty:
                continue
            med = per.median(numeric_only=True); med['season'] = s
            rows.append(med)
        out[p] = pd.DataFrame(rows).set_index('season') if rows else pd.DataFrame()
    return out
