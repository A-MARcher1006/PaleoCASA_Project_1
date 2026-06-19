# A Laboratory for Tracing SSA–EWAIS Teleconnections via EPEs

This repository contains the scripts used in the study:

*"Extreme Precipitation Events at Pine Island Glacier Linked to Temperature Variability in Southern South America"*

---

# 1. Description

This study investigates teleconnections between southern South America (SSA) and the eastern West Antarctic Ice Sheet (EWAIS) using ice-core-derived extreme precipitation event (EPE) windows.

---

# 2. Main Analyses Included

## Analysis of EPE Climatology

- Extreme precipitation event (EPE) identification
- Seasonal and annual time-series and trend analyses
- Pearson, Spearman, and Kendall correlation analyses
- Decadal analysis of frequency distributions

## Identification of Ice-Core Regions Sharing Similar EPE Climatology

- Ward’s hierarchical clustering analysis
- Creation of event-date lists by group

## Preprocessing of ERA5 Climate Fields

### Single-Level Variables

- Preprocessing of `tp` fields

### Pressure-Level Variables

- Preprocessing of `z500` fields
- Preprocessing of `t850` fields
- Computation and preprocessing of wind (`v + u`) fields at 850 hPa
- Computation and preprocessing of `VT850` fields

## Composite Analysis Using EPE-Filtered Dates

- Composite maps of `tp` + winds at 850 hPa
- Composite maps of `t850` + `z500`
- Composite maps of `VT850` + winds at 850 hPa

## Spatiotemporal Analysis

- Difference and trend maps + `t850` anomaly time series over SSA
- Mean maps by decade over SSA
- Decadal composite maps in the Western Hemisphere
- Climate-mode time series

---

# 3. Data

Data are not distributed in this repository.
ERA5 datasets must be downloaded from the Copernicus Climate Data Store.
Climate indices must be downloaded from their respective online sources. Users may reproduce the analyses using the same indices employed in this study or substitute equivalent datasets.

## Ice Core Positions

| Site | Latitude | Longitude |
|---|---:|---:|
| TT01 | -84.00000 | -79.50000 |
| TT02 | -83.45375 | -85.16708 |
| TT03 | -82.85000 | -88.99000 |
| TT04 | -82.17178 | -92.83136 |
| TT05 | -81.37192 | -94.69267 |
| TT06 | -80.41525 | -94.68422 |
| TT07 | -79.92000 | -94.39000 |
| TT08 | -79.12314 | -96.37517 |
| BR-IC-5 | -82.50856 | -79.46742 |
| BR-IC-6 | -81.05281 | -79.83586 |
| PIG | -78.50444 | -97.49333 |
| ITASE01-3 | -78.12020 | -95.64630 |
| ITASE01-4 | -77.61160 | -92.24830 |
| ITASE01-5 | -77.05930 | -89.13760 |
| PASO-1 | -79.63352 | -85.00625 |
| ST22 | -79.74170 | -78.54500 |
| Ferrigno | -74.57000 | -86.90000 |
| Gomez | -73.59000 | -70.36000 |
| Palmer | -73.86000 | -65.46000 |
| Rendezvous | -74.45000 | -78.16000 |
| Jurassic | -74.33000 | -73.06000 |

## ERA5 Reanalysis

- https://cds.climate.copernicus.eu/

## Climate Indices

### Southern Annular Mode (SAM)

- https://legacy.bas.ac.uk/met/gjma/sam.html

### ENSO, IOD, and AMO Indices

- https://meteorologia.unifei.edu.br/teleconexoes/indices

---

# 4. Repository Structure

This repository contains scripts only.
Figures are generated directly from the scripts.

```text
/epe_climatology/
├── EPE_identification
├── time_series_trend
├── correlation_analysis
└── distribution

/clustering/
├── HC_ward
└── EPE_date_list

/preprocessing/
├── preprocessing_tp
├── preprocessing_z500
├── preprocessing_t850
├── preprocessing_wind850
└── preprocessing_VT850

/composites/
└── composite_maps

/spatiotemporal_analysis/
├── trend_difference_maps_t850_timeseries
├── decadal_mean_maps
├── decadal_composite_maps
└── climate_modes_timeseries
```

---

# 5. Python Libraries and Packages Used

## Data Manipulation and Analysis

### xarray
- `xarray`

### pandas
- `pandas`

### numpy
- `numpy`

---

## Data Visualization and Plotting

### matplotlib

#### matplotlib.pyplot
- `matplotlib.pyplot`

#### matplotlib.colors
- `LinearSegmentedColormap`
- `TwoSlopeNorm`

---

## Statistics and Scientific Analysis

### SciPy

#### scipy.stats
- `linregress`
- `pearsonr`
- `spearmanr`
- `kendalltau`
- `stats`

#### scipy.signal
- `signal`

#### scipy.interpolate
- `RegularGridInterpolator`
- `griddata`
- `make_interp_spline`

#### scipy.cluster.hierarchy
- `linkage`
- `dendrogram`
- `fcluster`

#### scipy.spatial.distance
- `squareform`

#### scipy.ndimage
- `gaussian_filter1d`

---

## Geospatial Processing and Mapping

### cartopy

#### cartopy.crs
- `ccrs`

#### cartopy.feature
- `cfeature`

### shapely

#### shapely.geometry
- `Point`

#### shapely.prepared
- `prep`

---

## System Utilities and File Handling

### Python Standard Library

#### os
- `os`

#### glob
- `glob`

#### collections
- `Counter`

---

## Progress Monitoring

### tqdm
- `tqdm`

---
# 6. Installation

This repository was developed and tested using Python 3.12.4

Install the required packages with:

```bash
pip install -r requirements.txt
```

Alternatively, install the required packages manually:

```bash
pip install numpy
pip install pandas
pip install xarray
pip install scipy
pip install matplotlib
pip install cartopy
pip install netCDF4
pip install tqdm
pip install shapely
```
---
# 7. Code Developer

Andressa Marcher

---