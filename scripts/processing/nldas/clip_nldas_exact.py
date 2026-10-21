import os
import sys
import glob
import re
import argparse
import xarray as xr

_EXT_ROOT = os.environ.get('HYDRO_DATA_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
RAW_DIR       = os.path.join(_EXT_ROOT, 'nldas')
PROCESSED_SOUTH = os.path.join(_EXT_ROOT, 'data', 'processed', 'nldas_southern')
TEMPLATE_FILE   = os.path.join(PROCESSED_SOUTH, '20120101_southern.nc')

VARS_TO_KEEP = ['Rainf', 'Tair', 'PSurf', 'Qair', 'Wind_E', 'Wind_N', 'LWdown', 'SWdown']

# Pre-load the exact grid coordinates and mask from the 2012 reference
if not os.path.exists(TEMPLATE_FILE):
    raise FileNotFoundError(f"Template {TEMPLATE_FILE} missing!")

template_ds = xr.open_dataset(TEMPLATE_FILE)
# Get the exact bounding coordinates used in the older correct clip
LAT_SLICE = slice(template_ds.lat.values.min(), template_ds.lat.values.max())
LON_SLICE = slice(template_ds.lon.values.min(), template_ds.lon.values.max())
# Extract a binary mask from any variable (PSurf) to get the exact 1,487,777 ocean/mountain cutouts
# Drop the time dimension so the mask is 2D
SPATIAL_MASK = template_ds['PSurf'].isel(time=0).notnull().load()
template_ds.close()

def process_file(input_path):
    filename = os.path.basename(input_path)
    match = re.search(r'\.A(\d{8})\.', filename)
    if not match: return False
    
    date_str = match.group(1)
    out_south = os.path.join(PROCESSED_SOUTH, f"{date_str}_southern.nc")
    
    if os.path.exists(out_south) and os.path.getsize(out_south) > 5 * 1024 * 1024:
        print(f"Skipping {date_str}: exists")
        return True
        
    print(f"Processing EXACT MATCH {date_str}...", end='', flush=True)
    
    try:
        ds = xr.open_dataset(input_path)
        
        # 1. Exact geographic slice matching 2012 bounding box
        ds_sub = ds.sel(lat=LAT_SLICE, lon=LON_SLICE)
        
        # 2. Filter artifacts (just like the other script)
        if 'PSurf' in ds_sub.data_vars:
            ds_sub['PSurf'] = ds_sub['PSurf'].where((ds_sub['PSurf'] >= 50000) & (ds_sub['PSurf'] <= 110000))
            
        # 3. Apply the exact True/False spatial ocean mask from 2012
        ds_sub = ds_sub.where(SPATIAL_MASK)
        
        avail_vars = [v for v in VARS_TO_KEEP if v in ds_sub.data_vars]
        final_south = ds_sub[avail_vars]
        
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in final_south.data_vars}
        
        final_south.to_netcdf(out_south, encoding=encoding)
        ds.close()
        final_south.close()
        
        print(" [OK]")
        return True
    except Exception as e:
        print(f" [ERR: {e}]")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int)
    args = parser.parse_args()
    
    pattern = f"NLDAS_FOR0010_D.A{args.year}*.nc"
    files = sorted(glob.glob(os.path.join(RAW_DIR, pattern)))
    
    print(f"Found {len(files)} raw files for {args.year}")
    
    for f in files:
        process_file(f)

if __name__ == '__main__':
    main()
