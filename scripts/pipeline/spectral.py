"""
spectral.py — Spectral & Intermittency Analysis
================================================
Two complementary analyses of gridded forcing time series:

1. SPECTRAL ANALYSIS
   FFT-based power spectra of monthly time series.
   Highlights the annual (f=1/12 mo) and semi-annual (f=1/6 mo) peaks.
   Returns relative power so different variables/datasets are comparable.

2. INTERMITTENCY ANALYSIS (precipitation-focused)
   Wet/dry day statistics and spell-length distributions.
   Metric           | Definition
   ─────────────────|────────────────────────────────────────────────────
   wet_day_frac     | fraction of days P > threshold
   dry_day_frac     | fraction of days P ≤ threshold
   wet_mean         | mean precip on wet days (intensity)
   dry_spell_mean   | mean dry-spell length (consecutive dry days)
   wet_spell_mean   | mean wet-spell length
   dry_spell_max    | longest dry spell in period
   wet_spell_max    | longest wet spell in period
   P90_intensity    | 90th-percentile wet-day intensity
   frac_intense     | fraction of total precip from days > P90

   These are the statistics most sensitive to forcing products with
   different spatial origins — both AORC and NLDAS-3 are at 1km, but
   they differ in input data sources and processing chains.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal


WET_THRESHOLD_MM = 1.0   # mm/day; standard for daily precip intermittency


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def power_spectrum(series: pd.Series, detrend: bool = True) -> pd.DataFrame:
    """
    Compute one-sided FFT power spectrum of a time series.

    Parameters
    ----------
    series  : pd.Series indexed by date (ideally monthly-resampled)
    detrend : remove linear trend before FFT (avoids spectral leakage)

    Returns
    -------
    DataFrame with columns:
      period_months, frequency_per_month, power, relative_power (sum=1)
    """
    x = series.dropna().values.astype(float)
    n = len(x)
    if n < 12:
        return pd.DataFrame()

    if detrend:
        x = signal.detrend(x, type='linear')

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    x = x * window

    fft_vals  = np.fft.rfft(x)
    power     = (np.abs(fft_vals) ** 2) / n
    freqs     = np.fft.rfftfreq(n)            # cycles per sample (1 sample = 1 month)
    periods   = np.where(freqs > 0, 1.0 / freqs, np.inf)

    total = power.sum()
    return pd.DataFrame({
        'period_months'   : np.round(periods, 2),
        'frequency'       : np.round(freqs, 6),
        'power'           : np.round(power, 6),
        'relative_power'  : np.round(power / total, 6) if total > 0 else np.zeros_like(power),
    }).sort_values('period_months')


def spectral_peaks(spec_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top-N spectral peaks sorted by power."""
    if spec_df.empty:
        return pd.DataFrame()
    # Exclude DC component (infinite period)
    finite = spec_df[np.isfinite(spec_df['period_months'])].copy()
    return finite.nlargest(top_n, 'power').reset_index(drop=True)


def compare_spectra(
    ref_series: pd.Series,
    eval_series: pd.Series,
    label_ref: str = 'AORC',
    label_eval: str = 'NLDAS',
) -> dict:
    """
    Compare power spectra of two series. Returns spectral ratio (eval/ref)
    and the dominant period for each dataset.

    Fix: align both series to their COMMON non-NaN date range BEFORE FFT
    so both always have the same n → identical rfftfreq grids → the merge
    always returns the full spectral table instead of a few coincidental rows.
    """
    # Align to common non-NaN dates
    both = pd.DataFrame({'ref': ref_series, 'eval': eval_series}).dropna()
    if len(both) < 12:
        return {}

    spec_ref  = power_spectrum(both['ref'])
    spec_eval = power_spectrum(both['eval'])
    if spec_ref.empty or spec_eval.empty:
        return {}

    # Merge on frequency (exact match; avoids period_months rounding issues)
    merged = spec_ref[['frequency', 'period_months', 'power']].rename(
        columns={'power': f'power_{label_ref}'}
    ).merge(
        spec_eval[['frequency', 'power']].rename(
            columns={'power': f'power_{label_eval}'}),
        on='frequency', how='inner'
    )
    merged['spectral_ratio'] = (
        merged[f'power_{label_eval}'] / (merged[f'power_{label_ref}'] + 1e-12)
    )

    finite = merged[np.isfinite(merged['period_months'])].copy()
    if finite.empty:
        return {'table': merged}

    dom_ref  = finite.loc[finite[f'power_{label_ref}'].idxmax(),  'period_months']
    dom_eval = finite.loc[finite[f'power_{label_eval}'].idxmax(), 'period_months']

    # Closest spectral bin to 12-month and 6-month periods
    annual_row   = finite.iloc[(finite['period_months'] - 12).abs().argsort().iloc[0]]
    semi_ann_row = finite.iloc[(finite['period_months'] -  6).abs().argsort().iloc[0]]

    return {
        'table'                         : merged,
        f'dominant_period_{label_ref}'  : float(dom_ref),
        f'dominant_period_{label_eval}' : float(dom_eval),
        'annual_ratio'                  : float(annual_row['spectral_ratio']),
        'semi_annual_ratio'             : float(semi_ann_row['spectral_ratio']),
    }



# ══════════════════════════════════════════════════════════════════════════════
# INTERMITTENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _spell_lengths(binary: np.ndarray) -> list[int]:
    """Return list of consecutive run lengths where binary==True."""
    lengths = []
    count   = 0
    for v in binary:
        if v:
            count += 1
        elif count > 0:
            lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)
    return lengths if lengths else [0]


def intermittency_stats(
    series: pd.Series,
    threshold: float = WET_THRESHOLD_MM,
) -> dict:
    """
    Compute intermittency statistics for a precipitation time series.

    Parameters
    ----------
    series    : daily precip (mm/day)
    threshold : wet-day threshold in mm/day (default 1.0)

    Returns
    -------
    dict of scalar statistics
    """
    vals = series.dropna().values
    if len(vals) < 10:
        return {}

    wet  = vals >= threshold
    dry  = ~wet

    wet_vals = vals[wet]
    p90 = float(np.percentile(wet_vals, 90)) if wet_vals.size > 0 else np.nan

    wet_spells = _spell_lengths(wet)
    dry_spells = _spell_lengths(dry)

    return {
        'n_days'          : int(len(vals)),
        'wet_day_frac'    : round(float(wet.mean()), 4),
        'dry_day_frac'    : round(float(dry.mean()), 4),
        'wet_mean_mm'     : round(float(wet_vals.mean()), 3) if wet_vals.size else np.nan,
        'wet_spell_mean'  : round(float(np.mean(wet_spells)), 2),
        'wet_spell_max'   : int(max(wet_spells)),
        'dry_spell_mean'  : round(float(np.mean(dry_spells)), 2),
        'dry_spell_max'   : int(max(dry_spells)),
        'P90_intensity_mm': round(float(p90), 3),
        'frac_intense'    : round(float((vals[vals >= p90]).sum() / vals.sum()), 4) if vals.sum() > 0 else np.nan,
        'total_mm'        : round(float(vals.sum()), 1),
    }


def seasonal_intermittency(series: pd.Series, threshold: float = WET_THRESHOLD_MM) -> pd.DataFrame:
    """
    Compute intermittency stats broken down by season (DJF/MAM/JJA/SON).
    Returns a DataFrame with season as index.
    """
    season_map = {12:'DJF', 1:'DJF', 2:'DJF',
                  3:'MAM', 4:'MAM', 5:'MAM',
                  6:'JJA', 7:'JJA', 8:'JJA',
                  9:'SON', 10:'SON', 11:'SON'}
    series = series.dropna()
    rows = []
    for season in ['DJF','MAM','JJA','SON']:
        mask   = series.index.month.map(season_map) == season
        sub    = series[mask]
        stats  = intermittency_stats(sub, threshold)
        if stats:
            stats['season'] = season
            rows.append(stats)
    return pd.DataFrame(rows).set_index('season') if rows else pd.DataFrame()


def compare_intermittency(
    ref_series: pd.Series,
    eval_series: pd.Series,
    threshold: float = WET_THRESHOLD_MM,
) -> pd.DataFrame:
    """
    Side-by-side intermittency comparison between two datasets.
    Returns a summary DataFrame (metric × dataset).
    """
    stats_ref  = intermittency_stats(ref_series,  threshold)
    stats_eval = intermittency_stats(eval_series, threshold)
    if not stats_ref or not stats_eval:
        return pd.DataFrame()

    keys = [k for k in stats_ref if k != 'n_days']
    rows = []
    for k in keys:
        r, e = stats_ref.get(k, np.nan), stats_eval.get(k, np.nan)
        diff = e - r if isinstance(r, (int, float)) and isinstance(e, (int, float)) else np.nan
        pct  = (diff / abs(r) * 100) if r and r != 0 else np.nan
        rows.append(dict(metric=k, aorc=r, nldas=e,
                         diff=round(float(diff), 4),
                         pct_diff=round(float(pct), 2)))
    return pd.DataFrame(rows)
