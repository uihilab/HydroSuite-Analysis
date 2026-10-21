
import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr

_EXT_ROOT = os.environ.get('HYDRO_DATA_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
INPUT_DIR = os.path.join(_EXT_ROOT, 'aorc')
OUTPUT_DIR = r'data\processed\aorc_louisiana'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_daily_json(json_file):
    print(f"Processing {os.path.basename(json_file)}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    # Metadata
    date_str = data['date'] # YYYY-MM-DD
    bbox = data['bbox'] # [W, S, E, N]
    vars_data = data['variables']
    
    # Grid dimensions (Infer from first variable)
    # The JSON array structure is [row][col] or [lat][lon]?
    # Usually [lat_idx][lon_idx].
    # Need to check dimensions from one array.
    
    first_var = list(vars_data.keys())[0]
    grid = np.array(vars_data[first_var])
    nlat, nlon = grid.shape
    
    # Construct Coordinates
    # AORC is 30 arc-sec ~ 0.008333 deg?
    # Bbox: [-94.1, 28.8, -88.7, 33.1]
    # Use linspace to be safe
    lats = np.linspace(bbox[3], bbox[1], nlat) # North to South? or South to North?
    # Usually images are Row 0 = North. Let's assume North to South (descending).
    # Wait, in test_aorc.json, dimensions were [160, 201].
    # 33.1 - 28.8 = 4.3. 4.3 / 160 = 0.026.
    # If 0 is North, then lats should descend.
    # Let's verify with NLDAS standard (usually South-North).
    # But standard image array [0][0] is Top-Left.
    # So Lat should comprise [North, ..., South].
    
    # Using descending lats for now.
    lats = np.linspace(bbox[3], bbox[1], nlat)
    lons = np.linspace(bbox[0], bbox[2], nlon)
    
    # Create Dataset
    ds = xr.Dataset(coords={'lat': lats, 'lon': lons, 'time': pd.to_datetime(date_str)})
    
    for v_name, v_data in vars_data.items():
        if v_data is None: continue
        arr = np.array(v_data)
        
        # Add to dataset
        # Dimensions: (lat, lon). Add time.
        ds[v_name] = (('time', 'lat', 'lon'), [arr])
        
        # Attributes
        if v_name.startswith('APCP'): ds[v_name].attrs['units'] = 'kg/m^2' # Sum
        elif v_name.startswith('TMP'): ds[v_name].attrs['units'] = 'K'     # Mean
        
    # Save Daily NetCDF
    out_name = f"aorc_{date_str.replace('-','')}.nc"
    ds.to_netcdf(os.path.join(OUTPUT_DIR, out_name))

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, '*_daily.json')))
    if not files:
        print("No JSON files found yet.")
        return
        
    for f in files:
        # Check if already processed?
        # Maybe overwriting is fine for now.
        process_daily_json(f)

if __name__ == "__main__":
    main()
