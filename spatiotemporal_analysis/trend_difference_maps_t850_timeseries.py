"""
@author: andressa_marcher

@reviewers: write_here

"""

# =============================================================================
# ANALYSIS OF STANDARDIZED T850 ANOMALIES OVER SOUTH AMERICA
# =============================================================================
#
# Description
# -----------
# This script performs multiple analyses of standardized 850 hPa temperature
# (T850) anomalies over South America using ERA5-derived NetCDF data and
# Extreme Precipitation Event (EPE) dates.
#
# The workflow includes:
#
# 1. Annual mean anomaly analysis;
# 2. Comparison between all available days and EPE-only days;
# 3. Linear trend estimation for annual mean anomalies;
# 4. Spatial composite differences between two user-defined periods;
# 5. Spatial trend analysis at each grid point;
# 6. Identification of statistically significant trends.
#
# The script is fully adaptable and can be applied to:
#     - Any atmospheric variable;
#     - Any EPE dataset;
#     - Any season;
#     - Any geographic domain;
#     - Any temporal interval.
#
# Users must define:
#     - Input NetCDF file;
#     - Variable name;
#     - EPE date files;
#     - Analysis periods;
#     - Geographic domain;
#     - Output directory.
#
# Main libraries
# --------------
# xarray
# numpy
# pandas
# matplotlib
# cartopy
# scipy
# shapely
#
# Tested with:
# Python 3.12+ using Spyder IDE
#
# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from scipy.stats import linregress
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely.prepared import prep
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# NetCDF configuration
# -----------------------------------------------------------------------------

NETCDF_FILE = r"/path/to/netcdf_file.nc"

VARIABLE_NAME = "t850"

# -----------------------------------------------------------------------------
# EPE date files
# -----------------------------------------------------------------------------

EPE_DATE_FILES = [
    r"/path/to/epe_dates_1.txt",
    r"/path/to/epe_dates_2.txt",
]

# -----------------------------------------------------------------------------
# Time periods for comparison
# -----------------------------------------------------------------------------

PERIOD_1_START = "1979-01-01"
PERIOD_1_END   = "2010-12-31"

PERIOD_2_START = "2011-01-01"
PERIOD_2_END   = "2024-12-31"

# -----------------------------------------------------------------------------
# Spatial domain
# -----------------------------------------------------------------------------

LAT_MIN = -60
LAT_MAX = -20

LON_MIN = -80
LON_MAX = -30

# -----------------------------------------------------------------------------
# Color scale limits
# -----------------------------------------------------------------------------

COMPOSITE_VMIN = -0.5
COMPOSITE_VMAX = 0.5

TREND_VMIN = -0.02
TREND_VMAX = 0.02

# -----------------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------------

OUTPUT_DIR = r"/path/to/output_directory"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# LOAD NETCDF DATA
# =============================================================================

print("Loading NetCDF dataset...")

ds = xr.open_dataset(NETCDF_FILE)

da = ds[VARIABLE_NAME]

# Convert time coordinate to datetime
time_values = pd.to_datetime(ds["time"].values)

print("Dataset loaded successfully.")


# =============================================================================
# LOAD EPE DATES
# =============================================================================

print("Loading EPE dates...")

epe_dates = []

for file in EPE_DATE_FILES:

    df_dates = pd.read_csv(
        file,
        sep=r"\s+|,|\t",
        engine="python"
    )

    df_dates["date"] = pd.to_datetime(df_dates["date"])

    epe_dates.extend(df_dates["date"].tolist())

# Remove duplicated dates
epe_dates = pd.to_datetime(epe_dates).drop_duplicates()

print(f"Total EPE dates loaded: {len(epe_dates)}")


# =============================================================================
# SELECT EPE DAYS FROM NETCDF
# =============================================================================

mask_epe = time_values.isin(epe_dates)

da_epe = da.isel(time=mask_epe)

print("EPE days successfully selected.")


# =============================================================================
# DEFINE SPATIAL DOMAIN
# =============================================================================

lat_vals = ds["latitude"].values
lon_vals = ds["longitude"].values

lat_mask = (lat_vals >= LAT_MIN) & (lat_vals <= LAT_MAX)
lon_mask = (lon_vals >= LON_MIN) & (lon_vals <= LON_MAX)

lat_sel = lat_vals[lat_mask]
lon_sel = lon_vals[lon_mask]

lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)


# =============================================================================
# ANNUAL MEAN TIME SERIES
# =============================================================================

print("Computing annual mean anomalies...")

# Convert DataArray to pandas Series
temp_series = pd.Series(
    da.values.flatten(),
    index=time_values.repeat(len(lat_vals) * len(lon_vals))
)

# NOTE:
# This section assumes a pre-computed regional mean time series.
# Users may replace this section with their preferred regional averaging method.


# =============================================================================
# PERIOD COMPARISON
# =============================================================================

print("Computing spatial composites for two periods...")

mask_period1 = (
    (time_values >= PERIOD_1_START) &
    (time_values <= PERIOD_1_END) &
    mask_epe
)

mask_period2 = (
    (time_values >= PERIOD_2_START) &
    (time_values <= PERIOD_2_END) &
    mask_epe
)

da_period1 = da.isel(time=mask_period1)
da_period2 = da.isel(time=mask_period2)

# Compute mean difference
mean_difference = (
    da_period2[:, lat_mask, :][:, :, lon_mask].mean(dim="time")
    -
    da_period1[:, lat_mask, :][:, :, lon_mask].mean(dim="time")
)

mean_difference_values = mean_difference.values


# =============================================================================
# SPATIAL INTERPOLATION
# =============================================================================

print("Applying spatial interpolation...")

points = np.vstack([
    lon_grid.ravel(),
    lat_grid.ravel()
]).T

values = mean_difference_values.ravel()

grid_difference = griddata(
    points,
    values,
    (lon_grid, lat_grid),
    method="linear"
)


# =============================================================================
# CREATE CUSTOM COLORMAP
# =============================================================================

cmap_custom = LinearSegmentedColormap.from_list(
    "custom_cmap",
    ["blue", "white", "crimson"]
)


# =============================================================================
# PLOT PERIOD DIFFERENCE MAP
# =============================================================================

print("Generating composite difference map...")

fig, ax = plt.subplots(
    figsize=(10, 10),
    dpi=200,
    subplot_kw={"projection": ccrs.PlateCarree()}
)

ax.set_extent(
    [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    crs=ccrs.PlateCarree()
)

im = ax.pcolormesh(
    lon_grid,
    lat_grid,
    grid_difference,
    cmap=cmap_custom,
    shading="gouraud",
    vmin=COMPOSITE_VMIN,
    vmax=COMPOSITE_VMAX,
    transform=ccrs.PlateCarree()
)

# Map features
ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=1)

ax.add_feature(
    cfeature.LAND,
    facecolor="none",
    edgecolor="black",
    linewidth=1.2,
    zorder=2
)

ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=3)

ax.add_feature(
    cfeature.BORDERS,
    linestyle=":",
    zorder=3
)

# Colorbar
cbar = plt.colorbar(
    im,
    ax=ax,
    orientation="vertical",
    pad=0.02,
    fraction=0.05
)

cbar.set_label(
    f"{VARIABLE_NAME} anomaly difference "
    f"({PERIOD_2_START[:4]}–{PERIOD_2_END[:4]} minus "
    f"{PERIOD_1_START[:4]}–{PERIOD_1_END[:4]})"
)

plt.title(
    f"Difference in standardized {VARIABLE_NAME} anomalies "
    f"during EPE days",
    fontsize=12
)

# Save figure
output_map = os.path.join(
    OUTPUT_DIR,
    "spatial_difference_map.png"
)

fig.savefig(
    output_map,
    dpi=500,
    bbox_inches="tight"
)

plt.show()

print(f"Composite map saved to:\n{output_map}")


# =============================================================================
# GRID-POINT TREND ANALYSIS
# =============================================================================

print("Computing grid-point trends...")

da_sub = da_epe[:, lat_mask, :][:, :, lon_mask]

# Decimal years
years = (
    time_values[mask_epe].year +
    (time_values[mask_epe].dayofyear - 1) / 365.25
)

x = years.values

slopes = np.zeros((len(lat_sel), len(lon_sel)))
pvals  = np.ones((len(lat_sel), len(lon_sel)))

for i in range(len(lat_sel)):

    for j in range(len(lon_sel)):

        y = da_sub[:, i, j].values

        if np.isfinite(y).all() and len(y) > 1:

            slope, intercept, r, p, stderr = stats.linregress(x, y)

            slopes[i, j] = slope
            pvals[i, j] = p

        else:

            slopes[i, j] = np.nan
            pvals[i, j] = np.nan


# =============================================================================
# INTERPOLATE TREND FIELD
# =============================================================================

grid_slope = griddata(
    points,
    slopes.ravel(),
    (lon_grid, lat_grid),
    method="linear"
)


# =============================================================================
# LAND MASK FOR SIGNIFICANT POINTS
# =============================================================================

print("Applying land mask to significant trends...")

land_geom = cfeature.NaturalEarthFeature(
    "physical",
    "land",
    "50m"
).geometries()

land_polygons = [prep(poly) for poly in land_geom]

sig_points = np.where(pvals < 0.05)

lon_sig = lon_grid[sig_points]
lat_sig = lat_grid[sig_points]

mask_land = [

    any(poly.contains(Point(lon, lat)) for poly in land_polygons)

    for lon, lat in zip(lon_sig, lat_sig)

]

lon_sig_land = lon_sig[mask_land]
lat_sig_land = lat_sig[mask_land]


# =============================================================================
# PLOT TREND MAP
# =============================================================================

print("Generating trend map...")

fig, ax = plt.subplots(
    figsize=(10, 10),
    dpi=200,
    subplot_kw={"projection": ccrs.PlateCarree()}
)

ax.set_extent(
    [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    crs=ccrs.PlateCarree()
)

im = ax.pcolormesh(
    lon_grid,
    lat_grid,
    grid_slope,
    cmap=cmap_custom,
    shading="gouraud",
    vmin=TREND_VMIN,
    vmax=TREND_VMAX,
    transform=ccrs.PlateCarree(),
    zorder=1
)

# Map features
ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=2)

ax.add_feature(
    cfeature.LAND,
    facecolor="none",
    edgecolor="black",
    linewidth=1.2,
    zorder=3
)

ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=4)

ax.add_feature(
    cfeature.BORDERS,
    linestyle=":",
    zorder=4
)

# Significant points
ax.scatter(
    lon_sig_land,
    lat_sig_land,
    marker="x",
    color="black",
    s=30,
    transform=ccrs.PlateCarree(),
    zorder=5
)

# Colorbar
cbar = plt.colorbar(
    im,
    ax=ax,
    orientation="vertical",
    pad=0.02,
    fraction=0.05
)

cbar.set_label(
    f"Linear trend in standardized {VARIABLE_NAME} anomalies "
    "(per year)"
)

plt.title(
    f"Linear trend of standardized {VARIABLE_NAME} anomalies "
    f"during EPE days",
    fontsize=12
)

# Save figure
output_trend = os.path.join(
    OUTPUT_DIR,
    "trend_map.png"
)

fig.savefig(
    output_trend,
    dpi=500,
    bbox_inches="tight"
)

plt.show()

print(f"Trend map saved to:\n{output_trend}")


# =============================================================================
# ANNUAL REGIONAL TIME SERIES FOR EPE DAYS
# =============================================================================

print("Computing annual regional means for EPE days...")

# Compute regional mean
regional_mean = da_epe.mean(dim=["latitude", "longitude"])

# Convert to pandas Series
regional_series = pd.Series(
    regional_mean.values,
    index=time_values[mask_epe]
)

# Annual mean
annual_mean = regional_series.resample("Y").mean()

annual_mean.index = annual_mean.index.year


# =============================================================================
# LINEAR REGRESSION FUNCTION
# =============================================================================

def add_trendline(ax, x, y, color="black"):

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    trend = slope * x + intercept

    ax.plot(
        x,
        trend,
        color=color,
        linestyle="--",
        linewidth=1.5
    )

    if p_value < 0.001:
        p_text = f"{p_value:.2e}"
    else:
        p_text = f"{p_value:.3f}"

    stats_text = (
        f"y = {slope:.3f}x + {intercept:.2f}\n"
        f"R² = {r_value**2:.3f}, p = {p_text}"
    )

    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6)
    )


# =============================================================================
# PLOT ANNUAL TIME SERIES
# =============================================================================

print("Generating annual anomaly time series...")

fig, ax = plt.subplots(
    figsize=(6, 3),
    facecolor="none"
)

ax.bar(
    annual_mean.index,
    annual_mean.values,
    color="firebrick"
)

ax.axhline(0, color="black", linewidth=0.8)

ax.set_ylabel(f"Standardized {VARIABLE_NAME} anomaly")

ax.set_xlabel("Year")

ax.grid(axis="y", linestyle="--", alpha=0.6)

# Add regression line
add_trendline(
    ax,
    annual_mean.index.values,
    annual_mean.values
)

plt.tight_layout()

# Save figure
output_timeseries = os.path.join(
    OUTPUT_DIR,
    "annual_timeseries.png"
)

plt.savefig(
    output_timeseries,
    dpi=500,
    transparent=True
)

plt.show()

print(f"Annual time series saved to:\n{output_timeseries}")

print("\nAnalysis completed successfully.")