
import xarray as xr
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
import sys
import os
import json

def clip_netcdf(nc_path, geojson_path, output_path):
    """
    Clips a NetCDF file to the geometry defined in a GeoJSON file.
    """
    try:
        # Load Basin Geometry
        basin = gpd.read_file(geojson_path)
        
        # Ensure CRS match (NLDAS is usually EPSG:4326)
        if basin.crs != 'EPSG:4326':
            basin = basin.to_crs('EPSG:4326')

        # Load NetCDF
        # NLDAS files often have (lat, lon) or (latitude, longitude) dimensions
        ds = xr.open_dataset(nc_path)

        # Set CRS and Dimensions for rioxarray
        # Verify dimension names
        if 'lat' in ds.dims:
            ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        elif 'latitude' in ds.dims:
            ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        
        ds = ds.rio.write_crs("EPSG:4326")

        # Clip
        clipped = ds.rio.clip(basin.geometry.apply(mapping), basin.crs, drop=True)

        # Save
        # comp = dict(zlib=True, complevel=5)
        # encoding = {var: comp for var in clipped.data_vars}
        clipped.to_netcdf(output_path)
        
        print(f"Success: Clipped {os.path.basename(nc_path)} -> {os.path.basename(output_path)}")
        
        # Close datasets
        ds.close()
        clipped.close()
        
        return True

    except Exception as e:
        print(f"Error clipping {nc_path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python clip_nldas.py <input_nc> <geojson> <output_nc>")
        sys.exit(1)

    input_file = sys.argv[1]
    geojson_file = sys.argv[2]
    output_file = sys.argv[3]

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)
        
    if not os.path.exists(geojson_file):
        print(f"GeoJSON file not found: {geojson_file}")
        sys.exit(1)

    clip_netcdf(input_file, geojson_file, output_file)
