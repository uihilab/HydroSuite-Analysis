
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

# Robust absolute path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'aorc')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'aorc_louisiana')
BBOX_LA = [-94.1, 28.8, -88.7, 33.1] # West, South, East, North

def process_clip(zarr_path):
    day_str = os.path.basename(zarr_path).split('_')[1] # aorc_20120101_south.zarr
    outfile = os.path.join(OUTPUT_DIR, f"aorc_{day_str}_la.nc")
    
    if os.path.exists(outfile):
        print(f"Skipping {day_str} (Exists)")
        return
        
    print(f"Clipping {day_str}...")
    
    try:
        ds = xr.open_zarr(zarr_path)
        
        # Clip
        ds_clip = ds.sel(
            latitude=slice(BBOX_LA[1], BBOX_LA[3]),
            longitude=slice(BBOX_LA[0], BBOX_LA[2])
        )
        
        # --- DATA SANITIZATION ---
        # Filter out corrupt/fill values in pressure field (fixes 2020 AORC anomaly)
        if 'surface_pressure' in ds_clip.data_vars:
            ds_clip['surface_pressure'] = ds_clip['surface_pressure'].where(
                (ds_clip['surface_pressure'] >= 50000) & (ds_clip['surface_pressure'] <= 110000)
            )
        elif 'PRES_surface' in ds_clip.data_vars:
            ds_clip['PRES_surface'] = ds_clip['PRES_surface'].where(
                (ds_clip['PRES_surface'] >= 50000) & (ds_clip['PRES_surface'] <= 110000)
            )
        # -------------------------
        
        # Save as NetCDF (smaller, single file, easier for downstream)
        ds_clip.to_netcdf(outfile)
        print("  Saved NetCDF.")
        
    except Exception as e:
        print(f"  Failed {day_str}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process all Zarr dirs
    # glob returns dirs ending in .zarr
    files = sorted(glob.glob(os.path.join(INPUT_DIR, '*_south.zarr')))
    
    if not files:
        print("No Zarr files found yet.")
        return
        
    for f in files:
        process_clip(f)

if __name__ == "__main__":
    main()
