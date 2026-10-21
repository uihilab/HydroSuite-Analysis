
import os
import sys
import glob
import re
import argparse
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray

# Configuration
_EXT_ROOT = os.environ.get('HYDRO_DATA_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
RAW_DIR       = os.path.join(_EXT_ROOT, 'nldas')
PROCESSED_LA  = os.path.join(_EXT_ROOT, 'data', 'processed', 'nldas_louisiana')
PROCESSED_SOUTH = os.path.join(_EXT_ROOT, 'data', 'processed', 'nldas_southern')

# Basin GeoJSON files (local repo)
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASIN_LA    = os.path.join(_SCRIPT_DIR, 'data', 'raw', 'basins', 'Louisiana_State.geojson')
BASIN_SOUTH = os.path.join(_SCRIPT_DIR, 'data', 'raw', 'basins', 'Southern_US.geojson')

# Variables to keep
VARS_TO_KEEP = ['Rainf', 'Tair', 'PSurf', 'Qair', 'Wind_E', 'Wind_N', 'LWdown', 'SWdown']

def setup_directories():
    os.makedirs(PROCESSED_LA, exist_ok=True)
    os.makedirs(PROCESSED_SOUTH, exist_ok=True)

def load_geometries():
    geoms = {}
    
    # Louisiana Basin
    if os.path.exists(BASIN_LA):
        gdf = gpd.read_file(BASIN_LA)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        geoms['louisiana'] = gdf
    else:
        print(f"Warning: {BASIN_LA} not found")

    # Southern US
    if os.path.exists(BASIN_SOUTH):
        gdf = gpd.read_file(BASIN_SOUTH)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        geoms['southern'] = gdf
    else:
        print(f"Warning: {BASIN_SOUTH} not found")
        
    return geoms

def process_file(input_path, geoms, dry_run=False):
    filename = os.path.basename(input_path)
    
    match = re.search(r'\.A(\d{8})\.', filename)
    if not match:
        # print(f"Skipping {filename}: Cannot parse date")
        return False
    
    date_str = match.group(1)
    
    # Define outputs
    out_la = os.path.join(PROCESSED_LA, f"{date_str}_louisiana.nc")
    out_south = os.path.join(PROCESSED_SOUTH, f"{date_str}_southern.nc")
    
    # SMART CHECK: Only skip if BOTH exist AND appear valid
    la_exists = os.path.exists(out_la)
    south_exists = os.path.exists(out_south)
    
    # Validation thresholds for NLDAS-3 (0.01° / ~1km grid).
    # NLDAS-3 LA files are ~3-4MB (416×528 pixels); Southern US ~6-8MB.
    # These were previously set for NLDAS-2 (0.125°) which was ~30KB for LA.
    MIN_SIZE_LA    = 3 * 1024 * 1024   # 3 MB
    MIN_SIZE_SOUTH = 5 * 1024 * 1024   # 5 MB

    # Check if existing files are valid
    la_valid = la_exists and os.path.getsize(out_la) > MIN_SIZE_LA
    south_valid = south_exists and os.path.getsize(out_south) > MIN_SIZE_SOUTH
    
    if la_valid and south_valid:
        print(f"Skipping {date_str}: Already processed and valid")
        return True # Considered success (no action needed)
    
    # If invalid, we re-process
    processing_types = []
    if not la_valid:
        processing_types.append("LA")
    if not south_valid:
        processing_types.append("SOUTH")
        
    # Only print if we are actually doing something
    print(f"Processing {date_str} {processing_types}...", end='', flush=True)
    
    if dry_run:
        print(" [DRY RUN]")
        return True
        
    try:
        # Load Dataset - STRICT CHECK
        try:
            ds = xr.open_dataset(input_path)
        except Exception as e:
            print(f" ❌ RAW CORRUPT: {e}")
            return False
        
        # Set spatial dims
        if 'lat' in ds.dims:
            ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        ds.rio.write_crs("EPSG:4326", inplace=True)
        
        # --- DATA SANITIZATION ---
        # Filter out corrupt/fill values in pressure field (fixes 2017-2018 NLDAS anomalies)
        if 'PSurf' in ds.data_vars:
            # Valid sea-level pressure ranges from roughly 87,000 to 108,000 Pa.
            # Values like 0, -9999, or >110,000 are artifacts.
            ds['PSurf'] = ds['PSurf'].where((ds['PSurf'] >= 50000) & (ds['PSurf'] <= 110000))
        # -------------------------
        
        # Compression encoding
        comp = dict(zlib=True, complevel=5)
        
        # 1. Process Louisiana
        if 'louisiana' in geoms and not la_valid:
            try:
                # FIX: all_touched=True ensures we catch pixels even if center is outside
                clipped = ds.rio.clip(geoms['louisiana'].geometry.apply(mapping), geoms['louisiana'].crs, drop=True, all_touched=True)
                
                # Check for empty result
                if clipped.dims['lat'] == 0 or clipped.dims['lon'] == 0:
                     print(" [LA: EMPTY]", end='')
                else:
                    avail_vars = [v for v in VARS_TO_KEEP if v in clipped.data_vars]
                    final_la = clipped[avail_vars]
                    
                    # DEBUG: Print bounds
                    # print(f" [LA Bounds: {final_la.rio.bounds()}]", end='')
                    
                    encoding = {var: comp for var in final_la.data_vars}
                    final_la.to_netcdf(out_la, encoding=encoding)
                    final_la.close()
                    
                    # Verify output size
                    size_mb = os.path.getsize(out_la) / (1024 * 1024)
                    if size_mb < 3.0:  # NLDAS-3 LA files should be ~3-4MB
                         print(f" [LA: SMALL ({size_mb:.1f}MB) - DELETED]", end='')
                    else:
                        print(f" [LA: OK ({size_mb:.1f}MB)]", end='')
            except Exception as e:
                print(f" [LA: ERR {str(e)[:10]}]", end='')

        # 2. Process Southern US
        if 'southern' in geoms and not south_valid:
            try:
                clipped = ds.rio.clip(geoms['southern'].geometry.apply(mapping), geoms['southern'].crs, drop=True, all_touched=True)
                
                # Check for empty result
                if clipped.dims['lat'] == 0 or clipped.dims['lon'] == 0:
                     print(" [SOUTH: EMPTY]", end='')
                else:
                    avail_vars = [v for v in VARS_TO_KEEP if v in clipped.data_vars]
                    final_south = clipped[avail_vars]
                    encoding = {var: comp for var in final_south.data_vars}
                    final_south.to_netcdf(out_south, encoding=encoding)
                    final_south.close()

                    # Verify output size
                    if os.path.getsize(out_south) < MIN_SIZE_SOUTH:
                         print(" [SOUTH: TOO SMALL - DELETED]", end='')
                         os.remove(out_south)
                    else:
                        print(" [SOUTH: OK]", end='')
            except Exception as e:
                print(f" [SOUTH: ERR {str(e)[:10]}]", end='')
            
        ds.close()
        print(" Done.")
        return True
        
    except Exception as e:
        print(f" Failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch clip NLDAS files')
    # Make arguments optional with default=0
    parser.add_argument('year', type=int, nargs='?', default=0, help='Year (0 for all)')
    parser.add_argument('month', type=int, nargs='?', default=0, help='Month (0 for all)')
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    setup_directories()
    geoms = load_geometries()
    
    if args.year == 0:
        print(f"Scanning ALL files in {RAW_DIR}...")
        pattern = "NLDAS_FOR0010_D.A*.nc"
    elif args.month == 0:
        print(f"Scanning ALL months for {args.year}...")
        pattern = f"NLDAS_FOR0010_D.A{args.year}*.nc"
    else:
        month_str = f"{args.month:02d}"
        pattern = f"NLDAS_FOR0010_D.A{args.year}{month_str}*.nc"
    
    search_path = os.path.join(RAW_DIR, pattern)
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No raw files found in {RAW_DIR}")
        print(f"Pattern used: {pattern}")
        # Not a hard error, just nothing to do
        return
        
    print(f"Found {len(files)} raw files to check.")
    print(f"  LA Output:    {PROCESSED_LA}")
    print(f"  SOUTH Output: {PROCESSED_SOUTH}")
    
    count = 0
    errors = 0
    
    for i, f in enumerate(files):
        # Progress indication
        if i % 10 == 0:
             # Just a newline for clarity periodically
             pass 
            
        try:
            if process_file(f, geoms, args.dry_run):
                count += 1
            else:
                errors += 1
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f" Error details: {e}")
            errors += 1
            
    print(f"\n\nDone! Successfully processed/verified {count} files.")

if __name__ == '__main__':
    main()
