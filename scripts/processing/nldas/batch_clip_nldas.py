#!/usr/bin/env python3
"""
Batch process NLDAS NetCDF files: clip to basin and save by date
"""
import xarray as xr
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
import os
import glob
from pathlib import Path
import re

# Configuration
BASIN_SHAPE = 'data/raw/basins/Amite_DenhamSprings.geojson'
RAW_DIR = 'data/raw/nldas'
OUTPUT_DIR = 'data/processed/nldas'
DELETE_RAW = False  # Set to True ONLY AFTER verifying clipped files are correct

# Variables to keep
VARS_TO_KEEP = ['Rainf', 'Tair', 'PSurf', 'Qair', 'Wind_E', 'Wind_N', 'LWdown', 'SWdown']

def extract_date_from_filename(filename):
    """Extract YYYYMMDD from NLDAS filename"""
    # Pattern: NLDAS_FOR0010_D.A20120101.030.beta.nc
    match = re.search(r'\.A(\d{8})\.', filename)
    if match:
        return match.group(1)
    return None

def clip_nldas_file(input_path, basin, output_path):
    """Clip a single NLDAS file to basin geometry"""
    try:
        # Load NetCDF
        ds = xr.open_dataset(input_path)
        
        # Set spatial dimensions
        if 'lat' in ds.dims:
            ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        elif 'latitude' in ds.dims:
            ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        
        ds.rio.write_crs("EPSG:4326", inplace=True)
        
        # Clip to basin
        clipped = ds.rio.clip(basin.geometry.apply(mapping), basin.crs, drop=True)
        
        # Select only required variables
        available_vars = [v for v in VARS_TO_KEEP if v in clipped.data_vars]
        final_ds = clipped[available_vars]
        
        # Save with compression
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in final_ds.data_vars}
        final_ds.to_netcdf(output_path, encoding=encoding)
        
        # Close datasets
        ds.close()
        final_ds.close()
        
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    # Load basin geometry
    print("Loading basin geometry...")
    basin = gpd.read_file(BASIN_SHAPE)
    if basin.crs != 'EPSG:4326':
        basin = basin.to_crs('EPSG:4326')
    print(f"  Basin bounds: {basin.total_bounds}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all NLDAS files
    pattern = os.path.join(RAW_DIR, 'NLDAS_FOR0010_D.A*.nc')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No NLDAS files found in {RAW_DIR}")
        return
    
    print(f"\nFound {len(files)} NLDAS files to process")
    
    # Process each file
    processed = 0
    failed = 0
    total_size_saved = 0
    
    for i, input_file in enumerate(files, 1):
        filename = os.path.basename(input_file)
        date_str = extract_date_from_filename(filename)
        
        if not date_str:
            print(f"[{i}/{len(files)}] SKIP {filename} - cannot extract date")
            failed += 1
            continue
        
        output_file = os.path.join(OUTPUT_DIR, f"{date_str}_clipped.nc")
        
        # Skip if already processed
        if os.path.exists(output_file):
            print(f"[{i}/{len(files)}] SKIP {date_str} - already processed")
            continue
        
        input_size = os.path.getsize(input_file) / (1024**2)  # MB
        print(f"[{i}/{len(files)}] Processing {date_str} ({input_size:.0f}MB)...")
        
        success = clip_nldas_file(input_file, basin, output_file)
        
        if success:
            output_size = os.path.getsize(output_file) / (1024**2)  # MB
            size_saved = input_size - output_size
            total_size_saved += size_saved
            
            print(f"  SUCCESS: {output_size:.1f}MB saved (reduced by {size_saved:.0f}MB)")
            processed += 1
            
            # Delete original if requested
            if DELETE_RAW:
                os.remove(input_file)
                print(f"  Deleted original file")
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Processed: {processed}/{len(files)}")
    print(f"  Failed: {failed}")
    print(f"  Total space saved: {total_size_saved/1024:.1f} GB")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
