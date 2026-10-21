import xarray as xr
import glob
import numpy as np

def check_year(year, dataset='nldas'):
    _EXT_ROOT = os.environ.get('HYDRO_DATA_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    if dataset == 'nldas':
        files = glob.glob(fr'{_EXT_ROOT}\data\processed\nldas_southern\{year}*_southern.nc')
    else:
        files = glob.glob(fr'{_EXT_ROOT}\data\processed\aorc_southern\{year}*_southern.nc')
        
    bad_files = []
    print(f"Checking {dataset.upper()} for {year} ({len(files)} files found)")
    for f in files:
        try:
            ds = xr.open_dataset(f)
            if dataset == 'nldas':
                p = ds['PSurf'].values
            else:
                p = ds['surface_pressure'].values if 'surface_pressure' in ds else ds['PRES_surface'].values
                
            mean_p = np.nanmean(p)
            std_p = np.nanstd(p)
            
            # Normal pressure is around 96000-100000 Pa, std around 300-600 Pa
            if std_p > 1000 or mean_p < 90000 or mean_p > 110000:
                bad_files.append((f, mean_p, std_p))
            ds.close()
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print(f"Found {len(bad_files)} bad files in {year}")
    for b in bad_files[:5]:
        print(f"  {b[0].split('\\')[-1]}: mean={b[1]:.1f}, std={b[2]:.1f}")
        
check_year(2017, 'nldas')
check_year(2018, 'nldas')
check_year(2020, 'aorc')
