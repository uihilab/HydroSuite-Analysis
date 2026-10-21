"""
et_calc.py — Reference Evapotranspiration (FAO-56 Penman-Monteith and Hargreaves-Samani)
=========================================================================================
Computes daily ET₀ (mm/day) from meteorological forcing, plus the one-at-a-time (OAT)
substitution decomposition of the inter-dataset ET₀ difference.

Corrections relative to the earlier version
-------------------------------------------
* Net radiation now uses the forcing's downward longwave:
      Rn = (1 − α)·Rs + ε·LW↓ − ε·σ·T⁴
  The earlier code accepted LW↓ but never used it (net longwave came from the FAO
  empirical clear-sky formula), so the longwave OAT contribution was identically 0.
  The FAO empirical form is still available with ``rnl_method='fao'``.
* Latitude may be an array (per pixel / per station) instead of a fixed 31°N.
* Saturation vapour pressure uses (es(Tmax) + es(Tmin)) / 2 when Tmax/Tmin are
  supplied (FAO-56 eq. 12), otherwise es(Tmean).
* Hargreaves-Samani converts Ra to equivalent evaporation (× 0.408, FAO-56 eq. 52).
  Using Ra in MJ m⁻² d⁻¹ directly inflates HS-ET₀ by 2.45×.

Reference: Allen et al. (1998), FAO Irrigation and Drainage Paper No. 56.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

SIGMA_MJ = 4.903e-9          # Stefan-Boltzmann [MJ K⁻⁴ m⁻² day⁻¹]
W_TO_MJ_DAY = 0.0864         # W m⁻² (daily mean) → MJ m⁻² day⁻¹
MJ_TO_MM = 0.408             # MJ m⁻² day⁻¹ → mm day⁻¹ equivalent evaporation
SURFACE_EMISSIVITY = 0.98


# ── Thermodynamics ─────────────────────────────────────────────────────────────

def sat_vap_pressure(T_C):
    """Saturation vapor pressure [kPa] from temperature [°C]. FAO eq. 11."""
    return 0.6108 * np.exp((17.27 * T_C) / (T_C + 237.3))


def slope_sat_vap(T_C):
    """Slope of saturation vapor pressure curve Δ [kPa/°C]. FAO eq. 13."""
    return (4098.0 * sat_vap_pressure(T_C)) / (T_C + 237.3) ** 2


def psychrometric_constant(P_kPa):
    """Psychrometric constant γ [kPa/°C]. FAO eq. 8."""
    return 0.000665 * P_kPa


def actual_vap_pressure_from_q(q, P_kPa):
    """Actual vapor pressure eₐ [kPa] from specific humidity [kg/kg] and pressure [kPa]."""
    return (q * P_kPa) / (0.622 + 0.378 * q)


# ── Radiation ──────────────────────────────────────────────────────────────────

def extraterrestrial_radiation(lat_deg, doy):
    """
    Extraterrestrial radiation Ra [MJ m⁻² day⁻¹], FAO eq. 21.
    lat_deg and doy may be scalars or arrays (broadcast together).
    """
    phi   = np.radians(np.asarray(lat_deg, dtype=float))
    doy   = np.asarray(doy, dtype=float)
    dr    = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws    = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def net_radiation(Rs, Rl_down, T_C, ea, Ra, alpha: float = 0.23,
                  rnl_method: str = 'forcing', elevation_m: float = 50.0):
    """
    Net radiation Rn [MJ m⁻² day⁻¹].

      Rs, Rl_down : daily-mean downward shortwave / longwave [W m⁻²]
      T_C         : mean air temperature [°C]
      ea          : actual vapour pressure [kPa]   (used by 'fao' only)
      Ra          : extraterrestrial radiation [MJ m⁻² day⁻¹] (used by 'fao' only)
      rnl_method  : 'forcing' — net longwave from the forcing's LW↓ (default):
                        Rnl = ε·σ·T⁴ − ε·LW↓
                    'fao'     — FAO-56 eq. 39 empirical clear-sky estimate,
                                which ignores LW↓ entirely.
    """
    Rs_mj = np.asarray(Rs, float) * W_TO_MJ_DAY
    Rns   = (1 - alpha) * Rs_mj
    T_K   = np.asarray(T_C, float) + 273.15

    if rnl_method == 'forcing':
        Rl_mj = np.asarray(Rl_down, float) * W_TO_MJ_DAY
        Rnl   = SURFACE_EMISSIVITY * SIGMA_MJ * T_K ** 4 - SURFACE_EMISSIVITY * Rl_mj
    elif rnl_method == 'fao':
        Rso   = (0.75 + 2e-5 * elevation_m) * Ra
        fcd   = np.clip(1.35 * (Rs_mj / np.maximum(Rso, 1e-6)) - 0.35, 0.05, 1.0)
        Rnl   = SIGMA_MJ * T_K ** 4 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.01))) * fcd
    else:
        raise ValueError(f'rnl_method must be "forcing" or "fao", got {rnl_method!r}')
    return Rns - Rnl


# ── Penman-Monteith ────────────────────────────────────────────────────────────

def pm_et0(T_K, Rs, Rl, P_Pa, q, u2, doy, lat_deg=31.0, G: float = 0.0,
           Tmax_K=None, Tmin_K=None, rnl_method: str = 'forcing',
           return_components: bool = False):
    """
    FAO-56 Penman-Monteith reference ET₀ [mm/day]. All inputs broadcast, so this
    works for scalars, time series, or (lat × lon) grids with a lat_deg grid.
    """
    T_C   = np.asarray(T_K, float) - 273.15
    P_kPa = np.asarray(P_Pa, float) / 1000.0

    if Tmax_K is not None and Tmin_K is not None:
        es = 0.5 * (sat_vap_pressure(np.asarray(Tmax_K, float) - 273.15) +
                    sat_vap_pressure(np.asarray(Tmin_K, float) - 273.15))
    else:
        es = sat_vap_pressure(T_C)
    ea    = actual_vap_pressure_from_q(q, P_kPa)
    VPD   = np.maximum(es - ea, 0.0)
    Delta = slope_sat_vap(T_C)
    gamma = psychrometric_constant(P_kPa)

    Ra = extraterrestrial_radiation(lat_deg, doy)
    Rn = net_radiation(Rs, Rl, T_C, ea, Ra, rnl_method=rnl_method)

    numer_rad  = 0.408 * Delta * (Rn - G)
    numer_aero = gamma * (900.0 / (T_C + 273.0)) * u2 * VPD
    denom      = Delta + gamma * (1.0 + 0.34 * u2)
    ET0 = np.maximum((numer_rad + numer_aero) / denom, 0.0)

    if return_components:
        return dict(et0=ET0, rad_term=numer_rad / denom, aero_term=numer_aero / denom,
                    VPD_kPa=VPD, Delta=Delta, gamma=gamma, Rn_MJm2d=Rn, ea_kPa=ea, es_kPa=es)
    return ET0


# ── Hargreaves-Samani ──────────────────────────────────────────────────────────

def hs_et0(Tmean_C, Tmax_C, Tmin_C, lat_deg, doy):
    """
    Hargreaves-Samani reference ET₀ [mm/day], FAO-56 eq. 52:
        ET₀ = 0.0023 · (Tmean + 17.8) · √(Tmax − Tmin) · 0.408 · Ra
    Ra [MJ m⁻² day⁻¹] is converted to mm/day with 0.408. A negative diurnal range
    (bad data) gives NaN rather than a silent zero.
    """
    dtr = np.asarray(Tmax_C, float) - np.asarray(Tmin_C, float)
    Ra  = extraterrestrial_radiation(lat_deg, doy)
    et  = 0.0023 * (np.asarray(Tmean_C, float) + 17.8) * np.sqrt(np.where(dtr >= 0, dtr, np.nan)) \
          * MJ_TO_MM * Ra
    return np.maximum(et, 0.0)


# ── DataFrame helpers (domain-mean series) ─────────────────────────────────────

_PM_REQUIRED = ['temp', 'shortwave', 'longwave', 'pressure', 'specific_humidity', 'wind_2m']


def compute_et0_series(df: pd.DataFrame, suffix: str, lat_deg=31.0,
                       rnl_method: str = 'forcing') -> pd.Series:
    """
    PM-ET₀ from domain-mean columns ``<var>_<suffix>``.

    Prefer the per-pixel ET₀ columns produced by pipeline.gridpass (``et0_<suffix>``):
    PM is nonlinear, so ET₀ of the domain-mean inputs is not the domain-mean ET₀.
    This function remains for station / single-point use.
    """
    cols = {v: f'{v}_{suffix}' for v in _PM_REQUIRED}
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        print(f'    [et_calc] Missing for {suffix}: {missing}')
        return pd.Series(np.nan, index=df.index)
    tmax = df.get(f'temp_max_{suffix}')
    tmin = df.get(f'temp_min_{suffix}')
    et0 = pm_et0(
        T_K=df[cols['temp']].values, Rs=df[cols['shortwave']].values,
        Rl=df[cols['longwave']].values, P_Pa=df[cols['pressure']].values,
        q=df[cols['specific_humidity']].values, u2=df[cols['wind_2m']].values,
        doy=df.index.day_of_year.values, lat_deg=lat_deg,
        Tmax_K=None if tmax is None else tmax.values,
        Tmin_K=None if tmin is None else tmin.values,
        rnl_method=rnl_method,
    )
    return pd.Series(et0, index=df.index, name=f'et0_{suffix}')


SWAP_VARS = ['temp', 'shortwave', 'longwave', 'pressure', 'specific_humidity', 'wind_2m']


def et0_sensitivity(df: pd.DataFrame, ref: str = 'aorc', eval_: str = 'nldas',
                    lat_deg=31.0, rnl_method: str = 'forcing') -> pd.DataFrame:
    """
    OAT substitution on domain-mean inputs: swap one variable from ``ref`` to ``eval_``
    and record the mean ET₀ change. Also returns the interaction residual
    (total difference minus the sum of single-variable swaps).

    If ``df`` holds per-pixel OAT results from pipeline.gridpass (columns
    ``et0_swap_<var>``), those are used instead — they are exact per pixel.
    """
    et0_ref  = df[f'et0_{ref}']  if f'et0_{ref}'  in df else compute_et0_series(df, ref,  lat_deg, rnl_method)
    et0_eval = df[f'et0_{eval_}'] if f'et0_{eval_}' in df else compute_et0_series(df, eval_, lat_deg, rnl_method)
    total = float((et0_eval - et0_ref).mean())

    records = []
    for var in SWAP_VARS:
        swap_col = f'et0_swap_{var}'
        if swap_col in df:
            swapped = df[swap_col]
        else:
            kw = {v: df[f'{v}_{eval_ if v == var else ref}'].values for v in SWAP_VARS}
            swapped = pd.Series(pm_et0(
                T_K=kw['temp'], Rs=kw['shortwave'], Rl=kw['longwave'], P_Pa=kw['pressure'],
                q=kw['specific_humidity'], u2=kw['wind_2m'],
                doy=df.index.day_of_year.values, lat_deg=lat_deg, rnl_method=rnl_method,
            ), index=df.index)
        contrib = float((swapped - et0_ref).mean())
        records.append(dict(variable=var, mean_contribution_mm_d=round(contrib, 4),
                            fraction_of_bias=round(contrib / total, 4) if total else 0.0))
    out = pd.DataFrame(records).sort_values('mean_contribution_mm_d', key=abs, ascending=False)
    swaps = float(out['mean_contribution_mm_d'].sum())
    # Closure kept out of the variable rows (figure scripts plot every row)
    out.attrs['closure'] = dict(total_difference_mm_d=round(total, 4),
                                sum_of_swaps_mm_d=round(swaps, 4),
                                interaction_residual_mm_d=round(total - swaps, 4))
    return out.reset_index(drop=True)
