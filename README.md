# Hydrometeorological Consistency and Structural Differences in High-Resolution Atmospheric Forcing over the Gulf Coast

A reproducible diagnostic framework for comparing the **AORC** and **NLDAS-3** meteorological forcing products over the U.S. Gulf states, 2001–2020, daily. It is part of the [HydroSuite](https://github.com/uihilab/hydrosuite) ecosystem (see [Attribution](#attribution)).

This repository holds code only. Data, results and figures are written outside it (see [Data and outputs](#data-and-outputs)).

---

## What This Project Does

The pipeline quantifies how two widely used gridded forcing products differ in precipitation, temperature, shortwave and longwave radiation, specific humidity, wind and pressure, and what those differences do to derived quantities such as reference evapotranspiration (ET₀). It covers:

- **Gridded comparison (10 phases):** distributions, extremes, ET₀ attribution, seasonality, bias correction and trends, spatial coherence, scale sensitivity, intermittency and spectra.
- **Station ET₀ check:** Hargreaves–Samani and Penman–Monteith ET₀ from both products against GHCN-Daily stations.
- **Rain-gauge check:** daily precipitation structure (wet-day frequency, drizzle, intensity, spells, totals) from both products against GHCN-Daily gauges.

The practical question: **when you swap one forcing product for another, what changes, what doesn't, and what can statistical correction recover?**

---

## Study Domain

| Item | Detail |
|---|---|
| **Domain** | Gulf states (TX, LA, MS, AL, FL, GA, SC, NC), clipped from NLDAS-3 |
| **Grid** | Common footprint on the NLDAS-3 0.01° grid; AORC sampled by nearest neighbour |
| **Sub-domain** | Louisiana |
| **Period** | 2001–2020, daily, strict date pairing |
| **Stations** | GHCN-Daily (TMAX/TMIN for ET₀, PRCP for gauges) |

---

## Key Research Questions

1. **Systematic forcing differences:** Where do the two records diverge, and are the differences consistent across seasons, decades and spatial scales?
2. **Precipitation structure:** Does forcing construction change wet-day frequency, intensity and the upper tail?
3. **Atmospheric demand:** How do differences in radiation, humidity, temperature and wind translate into ET₀?
4. **Distributional vs. structural equivalence:** Can statistical adjustment align the distributions without recovering event-scale and cross-variable structure?

---

## Repository Structure

```
HydroSuiteAnalysis/
├── scripts/
│   ├── run_pipeline.py                 # Gridded analysis, Phases 1-10
│   ├── run_et_station_comparison.py    # ET0 (HS, PM) at GHCN-Daily stations
│   ├── run_precip_gauge_comparison.py  # Precipitation structure at GHCN-Daily gauges
│   ├── make_all_figures.py             # Every figure from one period's result tables
│   ├── make_clean_figures.py           #   figure modules called by make_all_figures
│   ├── regen_phase7.py                 #   spatial coherence maps
│   ├── regen_pipeline_figures.py       #   error PDFs, rolling anomalies
│   ├── regen_et0_figures.py            #   station ET0 bias map and Taylor diagram
│   ├── regen_et07_figures.py           #   PM vs HS method gap
│   ├── diagnose_inputs.py              # Checks units, time steps and footprints of the inputs
│   ├── survey_grids.py                 # Lists the distinct grids among the processed files
│   ├── ext_config.py                   # Paths (data directory, results, domains)
│   ├── pipeline/                       # Reusable modules
│   │   ├── gridpass.py                 #   one pass over all days on the common footprint
│   │   ├── freshread.py, repair.py     #   uncached reads; rebuild damaged files from raw
│   │   ├── stats.py, extremes.py       #   statistics; L-moment GEV, POT
│   │   ├── et_calc.py                  #   PM and HS ET0
│   │   ├── structure.py, spectral.py   #   intermittency, spells, spectra
│   │   ├── ghcn.py, loaders.py, regrid.py, units.py, domains.py
│   │   └── fig_style.py                #   figure style
│   ├── processing/                     # Clipping and conversion of raw files
│   └── retrieval/                      # Download scripts (HydroLang via Node.js, and Python)
├── tests/                              # Tests on synthetic data (no downloads needed)
├── data/raw/{basins,boundaries}/       # Small domain outlines (GeoJSON)
├── requirements.txt
└── README.md
```

---

## Environment Setup

**Python 3.10+** (tested with 3.11):

```bash
pip install -r requirements.txt
```

**Node.js 18+** and **HydroLang** for the NLDAS-3 download. The Node.js scripts import HydroLang from a `hydrolang/` folder at the repository root (ignored by git):

```bash
git clone https://github.com/uihilab/HydroLang.git ../HydroLang
cp -r ../HydroLang/hydrolang ./hydrolang
cd scripts/retrieval && npm install
```

**Data directory:** the scripts read and write under one directory, set with `HYDRO_DATA_DIR` (defaults to the repository root). Downloads, processed files and results all go there:

```bash
# PowerShell
$env:HYDRO_DATA_DIR="C:\path\to\data"

# Bash
export HYDRO_DATA_DIR="/path/to/data"
```

---

## Data Retrieval & Preprocessing

All inputs come from public archives. Every step writes under `HYDRO_DATA_DIR`; the analysis expects this layout:

```
$HYDRO_DATA_DIR/
├── aorc/raw/                      AORC hourly Zarr subsets (kept to rebuild damaged files)
├── aorc/processed/                aorc_YYYYMMDD_south_states.nc   (daily, clipped)
├── nldas/                         NLDAS_FOR0010_D.AYYYYMMDD.030.beta.nc   (raw daily NLDAS-3)
├── data/processed/nldas_southern/ YYYYMMDD_southern.nc   (daily, clipped)
└── ghcn/                          inventory.csv and daily/<station_id>.parquet
```

**1. AORC v1.1** (NOAA AWS Open Data, Zarr). Downloads and clips each day:

```bash
python scripts/retrieval/aorc/download_aorc_20yr_external.py              # 2001-2020
python scripts/retrieval/aorc/download_aorc_20yr_external.py 2010 2015    # selected years
```

**2. NLDAS-3** (NASA, daily beta forcing; about 650 MB per day). Download one month with HydroLang, then clip it to the study domain:

```bash
node scripts/retrieval/nldas/get_nldas_monthly.js 2016 8
python scripts/processing/nldas/clip_nldas_batch.py 2016 8
```

`scripts/run_nldas_pipeline.ps1 -Year 2016 -Month 8` runs both steps for a month; `scripts/process_all_nldas.ps1` clips every downloaded file. Repeat the download for each month of 2001-2020.

**3. GHCN-Daily** (NOAA NCEI). Station inventory and daily records for the study states:

```bash
python scripts/retrieval/ghcn/get_ghcn_retrieval.py
```

Then check the processed files before running the analysis:

```bash
python scripts/diagnose_inputs.py     # units, time steps, Tmax/Tmin, footprints
python scripts/survey_grids.py        # distinct grids among the processed files
```

---

## Running the Analysis

```bash
# 0. Tests (synthetic data, no downloads needed)
python tests/test_pipeline_fixes.py
python tests/test_run_pipeline_e2e.py

# 1. Gridded analysis, Phases 1-10, on the common footprint
#    The first run builds results/<period>/gridpass_v2/ (about 13 h on a laptop for
#    2001-2020); it checkpoints every 180 days and resumes after an interruption.
#    Later runs reuse it.
python scripts/run_pipeline.py --full

# 2. ET0 at GHCN-Daily stations (about 7 h)
python scripts/run_et_station_comparison.py --full

# 3. Precipitation at GHCN-Daily gauges (about 1.5 h)
python scripts/run_precip_gauge_comparison.py --full

# 4. All figures for the period (written to results/<period>/final_figures)
python scripts/make_all_figures.py --period full_2001_2020
```

Useful `run_pipeline.py` flags: `--phases 2,7`, `--year 2016`, `--years 2001-2010`, `--decades`, `--rebuild-cache`, `--no-resume`, `--dry-run`.

Reading is robust to unreliable storage: each file is read without the OS cache, retried, and, if it is truly damaged, rebuilt from the raw download (the damaged copy is moved to `_corrupt/`). Days that still cannot be read are listed in the run summary.

---

## Analysis Pipeline

| Phase | Analysis | Figure written |
|---|---|---|
| 1 | Monthly climatology | `phase1_monthly_climatology.png` |
| 2 | Distributions: error PDFs, anomaly Taylor diagram, bias heatmap | `phase2_*.png` |
| 3 | ET₀ (PM) and one-at-a-time attribution to each forcing variable | `phase3_et0.png` |
| 4 | Precipitation extremes: L-moment GEV (domain mean and per pixel), POT | `phase4_precip_extremes.png` |
| 5 | Circular seasonal statistics | `phase5_*.png` |
| 6 | Bias correction (leave-one-year-out QM), robust STL, Mann–Kendall, Pettitt | `phase6_*.png` |
| 7 | Spatial coherence: means, bias, daily correlation | `phase7_spatial_coherence_*.png` |
| 8 | Scale sensitivity (Gulf footprint vs. Louisiana) | `phase8_scale_sensitivity.png` |
| 9 | Intermittency: wet-day frequency, intensity, spells (domain mean and per pixel) | `phase9_intermittency.png` |
| 10 | Power spectra | `phase10_*.png` |

---

## Data and Outputs

Nothing below is tracked in git.

| Path (under `$HYDRO_DATA_DIR/results`) | Contents |
|---|---|
| `<period>/gridpass_v2/` | Daily domain series and per-pixel accumulations on the common footprint (parquet + checkpoint) |
| `<period>/tables/`, `<period>/figures/` | Phase 1-10 tables and figures |
| `<period>/final_figures/` | Output of `make_all_figures.py` |
| `et0_validation/<period>/` | Station ET₀ metrics |
| `precip_gauges/<period>/` | Gauge precipitation metrics, extracts and figure |
| `comparisons/<period>/` | PM vs HS method comparison |

---

## What's Next

- [ ] **Other forcing products:** Extend the framework to ERA5-Land, HRRR, and CPC to evaluate consistency across a broader family of gridded atmospheric products
- [ ] **Other domains:** Apply the diagnostic pipeline to climatologically distinct regions (e.g., Pacific Northwest, Great Plains, Northeast) to test whether structural differences are region-dependent
- [ ] **Other hydrologic proxies:** Go beyond ET₀ to evaluate forcing sensitivity through soil moisture dynamics, snowmelt partitioning, runoff generation, and baseflow recession signatures
- [ ] **Downstream hydrologic simulation:** Couple forcing differences to NWM or SUMMA outputs to quantify how distributional vs. structural divergence propagates into streamflow prediction
- [ ] **Sub-daily resolution:** Extend spectral and intermittency diagnostics to hourly timesteps where forcing products are known to diverge most sharply

---

## Attribution

This workflow is part of [**HydroSuite**](https://github.com/uihilab/hydrosuite), an open, web-native ecosystem of hydroinformatics tools. Data retrieval uses HydroLang, HydroSuite's data-access library. The analysis and evaluation code in this repository was developed within HydroSuite to show that multi-decade gridded and station records can be retrieved from public archives, harmonized, and evaluated reproducibly.

Figure styling follows the [**figures4papers**](https://github.com/ChenLiu-1996/figures4papers) conventions, implemented in [`scripts/pipeline/fig_style.py`](scripts/pipeline/fig_style.py).

---

## Citation

If you use this code, please cite the preprint that describes the analysis, along with this repository and HydroSuite:

Erazo Ramirez, C., & Demir, I. (2026). *Hydrometeorological consistency and structural differences in high-resolution atmospheric forcing products over the Gulf Coast* [Preprint]. EarthArXiv. https://doi.org/10.31223/X5NF96

```bibtex
@misc{erazoramirez2026hydrometeorological,
  author    = {Erazo Ramirez, Carlos and Demir, Ibrahim},
  title     = {Hydrometeorological Consistency and Structural Differences in High-Resolution Atmospheric Forcing Products over the Gulf Coast},
  year      = {2026},
  publisher = {EarthArXiv},
  note      = {Preprint},
  doi       = {10.31223/X5NF96},
  url       = {https://doi.org/10.31223/X5NF96}
}
```

---

## Contact

Carlos Erazo Ramirez, Ibrahim Demir  
Department of River-Coastal Science and Engineering, Tulane University  
Email: [cerazoramirez@tulane.edu](mailto:cerazoramirez@tulane.edu)  

---

## License

This project is distributed under the MIT License (see [LICENSE](LICENSE)).
