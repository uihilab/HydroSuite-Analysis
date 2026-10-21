
import os
import s3fs
import xarray as xr
import pandas as pd
import numpy as np
# remove dask import if not explicitly used, xarray handles it internally if installed
from datetime import datetime, timedelta

# Configuration
YEAR = 2000
# Robust absolute path based on script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'aorc')
BBOX_SOUTH = [-109.0, 25.0, -79.0, 38.0] # West, South, East, North

VARS_MAP = {
    'APCP_surface': ('precip', 'sum'),
    'TMP_2maboveground': ('temp', 'mean'),
    'SPFH_2maboveground': ('specific_humidity', 'mean'),
    'DLWRF_surface': ('longwave', 'mean'),
    'DSWRF_surface': ('shortwave', 'mean'),
    'PRES_surface': ('pressure', 'mean'),
    'UGRD_10maboveground': ('u_wind', 'mean'),
    'VGRD_10maboveground': ('v_wind', 'mean')
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Connecting to AORC S3 for {YEAR}...")
    fs = s3fs.S3FileSystem(anon=True)
    url = f"s3://noaa-nws-aorc-v1-1-1km/{YEAR}.zarr"
    
    ds_remote = xr.open_zarr(s3fs.S3Map(url, s3=fs))
    
    print("Slicing to Southern US...")
    ds_subset = ds_remote.sel(
        latitude=slice(BBOX_SOUTH[1], BBOX_SOUTH[3]),
        longitude=slice(BBOX_SOUTH[0], BBOX_SOUTH[2])
    )
    
    print(f"Subset Size: {ds_subset.dims}")
    
    start_date = datetime(YEAR, 1, 1)
    end_date = datetime(YEAR, 12, 31)
    delta = end_date - start_date
    
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        day_str = day.strftime("%Y%m%d")
        
        # Use .zarr extension
        outfile = os.path.join(OUTPUT_DIR, f"aorc_{day_str}_south.zarr")
        abs_outfile = os.path.abspath(outfile)
        
        # Debugging path
        # print(f"Checking: {abs_outfile}")
        
        # Check if exists (Simple check)
        if os.path.exists(outfile):
             print(f"SKIPPING {day_str} (Found at {abs_outfile})")
             continue
            
        print(f"STARTING_DOWNLOAD: {day_str}...")
        print(f"  Target: {abs_outfile}")
        
        try:
            t_start = day
            t_end = day + timedelta(hours=23)
            
            ds_day = ds_subset.sel(time=slice(t_start, t_end))
            
            if ds_day.time.size == 0:
                print(f"  No data for {day_str}")
                continue
            
            # Create Dataset using the slice structure directly to preserve coords
            # But we need to handle time dimension replacement
            
            # Create Dataset using the slice structure directly to preserve coords
            # But we need to handle time dimension replacement
            
            # Better approach:
            # Explicitly load coordinates (Fixes empty Zarr issue)
            try:
                lats = ds_day.latitude.values
                lons = ds_day.longitude.values
            except Exception as e:
                print(f"  Failed to load coords: {e}")
                continue
                
            ds_out = xr.Dataset(
                coords={
                    'time': [pd.to_datetime(day.strftime("%Y-%m-%d"))],
                    'latitude': lats,
                    'longitude': lons
                }
            )
            
            data_vars = {}
            for zarr_var, (out_name, method) in VARS_MAP.items():
                print(f"  Fetching {zarr_var}...")
                
                # Use .load() to bring to memory (numpy)
                da = ds_day[zarr_var].load()
                
                if method == 'sum':
                    res = da.sum(dim='time', keep_attrs=True)
                else:
                    res = da.mean(dim='time', keep_attrs=True)
                
                res.attrs = da.attrs
                if method == 'sum' and 'precip' in out_name:
                    res.attrs['units'] = 'kg/m^2/day'
                
                data_vars[out_name] = res
                
            for name, da in data_vars.items():
                ds_out[name] = (('time', 'latitude', 'longitude'), [da.values])
                ds_out[name].attrs = da.attrs
            
            # Debug: Print ds_out info
            print("  Dataset to write:")
            print(ds_out)
            
            # Save to Zarr
            ds_out.to_zarr(outfile, mode='w')
            print("  Saved Zarr.")
            
        except Exception as e:
            print(f"  Failed {day_str}: {e}")

if __name__ == "__main__":
    main()
