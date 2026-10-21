"""
structure.py — Temporal-structure diagnostics
==============================================
* STL decomposition of monthly series (statsmodels when installed; otherwise a
  classical moving-average decomposition, flagged in the output).
* Temporal-structure metrics used to test the claim that statistical adjustment
  aligns distributions without recovering event sequencing or cross-variable
  coupling. Each metric is computed for the reference (AORC), raw NLDAS-3, and the
  adjusted series, so "not recovered" is a number, not an assertion.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from pipeline.stats import deseasonalize


def stl_decompose(monthly: pd.Series, period: int = 12) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame[trend, seasonal, resid], method)."""
    s = monthly.dropna().astype(float)
    try:
        from statsmodels.tsa.seasonal import STL
        r = STL(s, period=period, robust=True).fit()
        return pd.DataFrame({'trend': r.trend, 'seasonal': r.seasonal, 'resid': r.resid}), 'STL (statsmodels, robust)'
    except ImportError:
        trend = s.rolling(period, center=True, min_periods=period).mean()
        trend = trend.rolling(2, center=True).mean() if period % 2 == 0 else trend
        detr = s - trend
        mm = detr.groupby(s.index.month).mean()
        mm = mm - mm.mean()                       # seasonal component sums to zero over a year
        seas = pd.Series(mm.reindex(s.index.month).values, index=s.index)
        return (pd.DataFrame({'trend': trend, 'seasonal': seas, 'resid': s - trend - seas}),
                'classical moving-average (statsmodels not installed)')


def _spells(wet: np.ndarray) -> float:
    lengths, c = [], 0
    for w in wet:
        if w: c += 1
        elif c: lengths.append(c); c = 0
    if c: lengths.append(c)
    return float(np.mean(lengths)) if lengths else np.nan


def structure_metrics(p: pd.Series, sw: pd.Series | None = None, threshold: float = 1.0,
                      event_ref: pd.Series | None = None) -> dict:
    """
    Temporal-structure metrics of a daily precipitation series:
      lag1_autocorr, wet_day_frac, mean_wet_spell, p_wet_given_wet, p_wet_given_dry,
      precip_sw_anom_corr  (coupling with the shortwave anomaly of the same product),
      event_hit_rate       (share of event_ref's top-5% days that are also top-5% here).
    """
    x = p.dropna()
    wet = (x >= threshold).values
    out = dict(
        lag1_autocorr=float(x.autocorr(1)),
        wet_day_frac=float(wet.mean()),
        mean_wet_spell=_spells(wet),
        p_wet_given_wet=float(wet[1:][wet[:-1]].mean()) if wet[:-1].any() else np.nan,
        p_wet_given_dry=float(wet[1:][~wet[:-1]].mean()) if (~wet[:-1]).any() else np.nan,
    )
    if sw is not None:
        both = pd.DataFrame({'p': deseasonalize(x), 's': deseasonalize(sw.reindex(x.index))}).dropna()
        out['precip_sw_anom_corr'] = float(both['p'].corr(both['s'])) if len(both) > 30 else np.nan
    if event_ref is not None:
        ref = event_ref.reindex(x.index)
        top_ref = ref >= ref.quantile(0.95); top_x = x >= x.quantile(0.95)
        out['event_hit_rate'] = float((top_ref & top_x).sum() / max(top_ref.sum(), 1))
    return out
