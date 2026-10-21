
import os
import glob
import re
import zipfile
import tempfile
import xarray as xr
import rioxarray
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import mapping

# Config
_EXT_ROOT = os.environ.get('HYDRO_DATA_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
RAW_PRISM_DIR = os.path.join(_EXT_ROOT, 'prism')
PROCESSED_DIR = r'data\processed\prism'
BASIN_FILE = r'data\raw\basins\Louisiana_State.geojson'
VARS = ['ppt', 'tmean']

os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_year(year, variable, geom):
    print(f"\nProcessing {variable} for {year}...")
    
    # Pattern: PRISM_ppt_stable_4kmD2_20120101_bil.zip
    search_path = os.path.join(RAW_PRISM_DIR, variable, f"PRISM_{variable}_stable_4kmD2_{year}*_bil.zip")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No files found for {year} {variable}")
        return

    daily_datasets = []
    dates = []
    
    # We will process day by day and stack in memory (or chunk)
    # PRISM 4km is small enough for 1 year in memory if clipped.
    
    for f in files:
        # Extract date
        match = re.search(r'_(\d{8})_', f)
        if not match: continue
        date_str = match.group(1)
        date = pd.to_datetime(date_str)
        
        try:
            with zipfile.ZipFile(f, 'r') as z:
                # Find .bil file
                bil_name = [n for n in z.namelist() if n.endswith('.bil')][0]
                
                # Extract to temp to read with rasterio/rioxarray
                with tempfile.TemporaryDirectory() as tmpdir:
                    z.extractall(tmpdir)
                    bil_path = os.path.join(tmpdir, bil_name)
                    
                    # Open with rioxarray
                    da = rioxarray.open_rasterio(bil_path).isel(band=0)
                    da = da.drop_vars('band')
                    
                    # Write CRS if missing (PRISM is usually EPSG:4269 NAD83)
                    if da.rio.crs is None:
                        da.rio.write_crs("EPSG:4269", inplace=True)
                    
                    # Reproject to 4326 (WGS84) to match NLDAS
                    da = da.rio.reproject("EPSG:4326")
                    
                    # Clip
                    clipped = da.rio.clip(geom.geometry.apply(mapping), geom.crs, drop=True)
                    
                    # Add time dim
                    clipped = clipped.expand_dims(time=[date])
                    
                    # Name variable
                    clipped.name = variable
                    
                    daily_datasets.append(clipped)
                    dates.append(date)
                    
                    # Progress
                    if len(dates) % 30 == 0:
                        print('.', end='', flush=True)
                        
        except Exception as e:
            print(f"Error {date_str}: {e}")
            
    if not daily_datasets:
        return

    print(" Concatenating...")
    ds = xr.concat(daily_datasets, dim='time')
    
    # Save
    out_file = os.path.join(PROCESSED_DIR, f"prism_{variable}_{year}_louisiana.nc")
    
    # Encoding for compression
    comp = dict(zlib=True, complevel=5)
    ds.to_netcdf(out_file, encoding={variable: comp})
    print(f"Saved {out_file} ({ds.nbytes / 1e6:.1f} MB raw data)")

def main():
    # Load geometry
    gdf = gpd.read_file(BASIN_FILE)
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
        
    for var in VARS:
        process_year(2012, var, gdf)

if __name__ == '__main__':
    main()
