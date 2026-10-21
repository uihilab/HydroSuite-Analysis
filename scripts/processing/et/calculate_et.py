
import numpy as np
import xarray as xr

# FAO-56 Constants
GSC = 0.0820  # Solar constant [MJ m-2 min-1]
SIGMA = 4.903e-9  # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
CP = 1.013e-3  # Specific heat capacity [MJ kg-1 degC-1]
EPSILON = 0.622  # Ratio of molecular weight of water vapour/dry air

def calculate_et0(
    t_mean,  # Mean/Daily Temp [K] or [C] (function handles conversion if needed)
    pres,    # Pressure [Pa]
    u2,      # Wind speed at 2m [m/s]
    rs,      # Shortwave Radiation [W/m2] -> [MJ m-2 day-1]
    rl_in,   # Longwave Radiation Down [W/m2] -> Needed for Rnl? No, FAO uses Rs.
             # Actually FAO uses Rs (solar) to compute Rns. And Tmax/Tmin/ea for Rnl.
             # But we have explicit RL_DOWN (Longwave Down).
             # Rnet = (Rs_down - Rs_up) + (Rl_down - Rl_up)
             # If strictly FAO-56 method with limited data (Tmax, Tmin, Rs), we use empirical Rnl.
             # BUT since we have forcing data (RL_DOWN), we should use it for better accuracy!
             # Rnet = (1 - albedo)*Rs + (Rl_down - Rl_up)
             # Rl_up = epsilon_s * sigma * T^4.
             # This is "Data-Rich Penman-Monteith".
    q_spec,  # Specific Humidity [kg/kg] -> e_a [kPa]
    elevation=0, # Elevation [m] (not used if Pres is provided)
    lat_rad=None, # Latitude [rad] (needed for Ra if Rs missing, but we have Rs)
    doy=None     # Day of Year (needed for Ra if Rs missing)
):
    """
    Calculates Reference Evapotranspiration (ET0) using FAO-56 Penman-Monteith.
    Using full forcing data where available.
    """
    
    # 1. Convert Units
    # Temperature: K -> C
    t_c = t_mean - 273.15 if np.any(t_mean > 100) else t_mean
    t_k = t_c + 273.15
    
    # Pressure: Pa -> kPa
    p_kpa = pres / 1000.0
    
    # Radiation: W/m2 -> MJ/m2/day. (Mean daily W/m2 * 0.0864)
    # 1 W/m2 = 1 J/s/m2 = 86400 J/day/m2 = 0.0864 MJ/day/m2
    rs_mj = rs * 0.0864 
    rl_in_mj = rl_in * 0.0864
    
    # 2. Net Radiation (Rn)
    # Albedo for grass reference crop = 0.23
    alpha = 0.23
    rns = (1 - alpha) * rs_mj
    
    # Rnl (Net Longwave)
    # Rnl = Rl_down - Rl_up
    # Rl_up = epsilon_surf * sigma * T^4
    # epsilon_surf for grass ~ 0.98 (FAO usually assumes net directly or uses T)
    # FAO Eq 39 estimates Rnl from Tmax, Tmin, ea, Rs/Rso.
    # But we have Rl_down directly from AORC/NLDAS (DLWRF).
    # So Rnl = Rl_down - (0.98 * 4.903e-9 * T_k^4)
    # Wait, simple physics is better than empirical FAO Rnl equation here.
    # Validate with standard: typical usage with NLDAS uses standard Stefan-Boltzmann for up.
    
    # Using simple Rnl calculation based on T and Rl_down
    # Stefan-Boltzmann for daily step is tricky with T_mean^4 vs mean(T^4).
    # But error is small.
    epsilon_s = 0.98
    rl_up_mj = epsilon_s * SIGMA * (t_k**4)
    rnl = rl_in_mj - rl_up_mj
    
    rn = rns + rnl
    
    # 3. Soil Heat Flux (G)
    # For daily time step, G ≈ 0
    g = 0
    
    # 4. Vapor Pressure Deficit (es - ea)
    # es: Saturation Vapor Pressure [kPa]
    # es = 0.6108 * exp(17.27 * T / (T + 237.3))
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    
    # ea: Actual Vapor Pressure [kPa] derived from Specific Humidity (q) & Pressure
    # ea = (q * P) / (0.622 + 0.378 * q) -> standard meteorology
    # FAO usually uses RH.
    ea = (q_spec * p_kpa) / (EPSILON + 0.378 * q_spec)
    
    vpd = es - ea
    vpd = np.maximum(vpd, 0.0) # Ensure non-negative
    
    # 5. Slope of Saturation Vapor Pressure Curve (delta) [kPa/degC]
    delta = (4098 * es) / ((t_c + 237.3)**2)
    
    # 6. Psychrometric Constant (gamma) [kPa/degC]
    # gamma = (Cp * P) / (epsilon * lambda)
    # lambda = Latent Heat of Vaporization [MJ/kg]
    # lambda = 2.501 - (0.002361 * T)
    lam = 2.501 - (0.002361 * t_c)
    gamma = (CP * p_kpa) / (EPSILON * lam)
    
    # 7. Computation
    # Num = 0.408 * delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * vpd
    # Denom = delta + gamma * (1 + 0.34 * u2)
    # Note: 900 coeff is for Daily time step (FAO-56 Table 2.2 for Reference Grass).
    # Hourly is 37.
    
    # Ensure u2 is not zero (avoid singularity? No, only in denom if negative?)
    # u2 >= 0.
    
    numerator = (0.408 * delta * (rn - g)) + (gamma * (900 / (t_c + 273)) * u2 * vpd)
    denominator = delta + (gamma * (1.0 + 0.34 * u2))
    
    et0 = numerator / denominator
    
    return np.maximum(et0, 0.0)

def convert_wind_height(u_z, z_meas, z_ref=2.0):
    """
    Logarithmic wind profile adjustment.
    u2 = u_z * 4.87 / ln(67.8 * z - 5.42)
    FAO-56 Eq 47.
    """
    return u_z * (4.87 / np.log(67.8 * z_meas - 5.42))
