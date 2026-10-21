#!/usr/bin/env python3
"""
Download the Louisiana state outline (US Census cartographic boundary, 1:500k) and
write data/raw/basins/Louisiana_Outline.geojson, which ext_config uses for the
'louisiana' domain in place of the Louisiana_State.geojson box.
"""
import geopandas as gpd
import os

# Output directory (absolute, so the script works from any working directory)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                          'data', 'raw', 'basins')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Louisiana FIPS code is 22
# Using Census TIGER/Line data via geopandas read_file with URL
# cartographic boundary file: coastline clipped to land, unlike TIGER/Line which extends into water
url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_500k.zip"

print("Downloading US states...")
states = gpd.read_file(url)

# Filter for Louisiana (FIPS code: 22, STATEFP: '22')
louisiana = states[states['STATEFP'] == '22']

if len(louisiana) == 0:
    print("Louisiana not found! Trying alternative...")
    # Try by name
    louisiana = states[states['NAME'] == 'Louisiana']

# Save as GeoJSON
output_path = os.path.join(OUTPUT_DIR, 'Louisiana_Outline.geojson')
louisiana.to_crs(4326).to_file(output_path, driver='GeoJSON')

print(f"Louisiana boundary saved to: {output_path}")
print(f"Bounds: {louisiana.total_bounds}")
print(f"CRS: {louisiana.crs}")
