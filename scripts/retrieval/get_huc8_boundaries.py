"""
get_huc8_boundaries.py — Download HUC8 polygons from the USGS Watershed Boundary Dataset
=======================================================================================
Writes data/raw/basins/<Name>_HUC8.geojson, which ext_config.OPTIONAL_DOMAINS picks up
automatically (amite → Amite_HUC8.geojson, calcasieu → Calcasieu_HUC8.geojson,
ouachita → UpperOuachita_HUC8.geojson).

The HUC8 codes are REQUIRED arguments — look them up (e.g. USGS WBD viewer or
https://water.usgs.gov/wsc/map_index.html) and confirm before running; the script
prints the name WBD returns for each code so you can check it.

    py -3.11 scripts/retrieval/get_huc8_boundaries.py Amite=<huc8> Calcasieu=<huc8> UpperOuachita=<huc8>
"""

import os, sys, json, argparse
import urllib.parse, urllib.request

WBD_HUC8 = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', 'basins')


def fetch(huc8: str) -> dict:
    q = urllib.parse.urlencode({'where': f"huc8='{huc8}'", 'outFields': 'huc8,name,areasqkm',
                                'outSR': 4326, 'f': 'geojson'})
    with urllib.request.urlopen(f'{WBD_HUC8}?{q}', timeout=120) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pairs', nargs='+', help='Name=HUC8code (8 digits)')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for pair in a.pairs:
        name, _, code = pair.partition('=')
        if not (code.isdigit() and len(code) == 8):
            sys.exit(f'{pair!r}: expected Name=8-digit HUC8 code')
        gj = fetch(code)
        feats = gj.get('features', [])
        if not feats:
            sys.exit(f'HUC8 {code} not found in WBD')
        props = feats[0]['properties']
        out = os.path.join(OUT_DIR, f'{name}_HUC8.geojson')
        with open(out, 'w') as fh:
            json.dump(gj, fh)
        print(f'{name}: HUC8 {code} = "{props.get("name")}" ({props.get("areasqkm")} km²) → {out}')


if __name__ == '__main__':
    main()
