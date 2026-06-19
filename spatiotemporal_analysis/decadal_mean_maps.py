"""
@author: andressa_marcher

@reviewers: write_here

"""

# ==================================================================================================
# Mean Standardized T850 Anomaly Maps over South America
# ==================================================================================================
#
# DESCRIPTION
# --------------------------------------------------------------------------------------------------
# This script generates spatial composites of standardized 850 hPa temperature anomalies (T850)
# over South America using ERA5-derived NetCDF data.
#
# Two types of analyses are included:
#
# 1. Climatological Mean Composite
#    - Computes the mean T850 anomaly for user-defined seasons and periods.
#
# 2. Event-Based Composite
#    - Computes the mean T850 anomaly using externally provided event-date files
#      (e.g., EPE groups, SOM clusters, lagged-event dates, etc.).
#
# The script:
#   - Loads standardized T850 fields from NetCDF;
#   - Filters by season and/or event dates;
#   - Computes temporal composites;
#   - Applies spatial smoothing using scipy.griddata;
#   - Produces high-resolution South America maps.
#
# INPUT REQUIREMENTS
# --------------------------------------------------------------------------------------------------
# NetCDF file:
#   - Daily standardized T850 anomalies
#   - Expected dimensions:
#         time × latitude × longitude
#
# Event-date files:
#   - Plain text or CSV files containing one date per line
#   - Dates must be readable by pandas.to_datetime()
#
# USER-DEFINED PARAMETERS
# --------------------------------------------------------------------------------------------------
# The following sections MUST be configured by the user:
#
#   • NetCDF input paths
#   • Event-date file paths
#   • Output directories
#   • Seasons to include
#   • Time intervals
#   • Spatial domain
#   • Plot titles
#   • Colorbar ranges
#
# OUTPUT
# --------------------------------------------------------------------------------------------------
# High-resolution PNG figures (500 dpi) showing:
#   - Mean standardized T850 anomalies
#   - Spatial composites over South America
#
# ==================================================================================================
# Tested with:
# Python 3.12+ using Spyder IDE
#
# ==================================================================================================
# Import Libraries
# ==================================================================================================

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap


# ==================================================================================================
# USER INPUT SECTION
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# NetCDF file containing standardized T850 anomalies
# --------------------------------------------------------------------------------------------------

NETCDF_FILE = r"PATH_TO_T850_STANDARDIZED_NETCDF.nc"


# --------------------------------------------------------------------------------------------------
# Output directory for generated figures
# --------------------------------------------------------------------------------------------------

OUTPUT_DIR = r"PATH_TO_OUTPUT_DIRECTORY"


# --------------------------------------------------------------------------------------------------
# Seasons to include in climatological analysis
#
# Examples:
#   ["DJF"]
#   ["SON"]
#   ["DJF", "SON"]
#   ["MAM", "JJA"]
# --------------------------------------------------------------------------------------------------

SEASONS = []


# --------------------------------------------------------------------------------------------------
# Climatological analysis period
#
# Example:
#   PERIOD_START = "1979"
#   PERIOD_END   = "1990"
# --------------------------------------------------------------------------------------------------

PERIOD_START = ""
PERIOD_END   = ""


# --------------------------------------------------------------------------------------------------
# Spatial domain over South America
# --------------------------------------------------------------------------------------------------

LAT_MIN = -60
LAT_MAX = -20

LON_MIN = -80
LON_MAX = -30


# --------------------------------------------------------------------------------------------------
# Plot settings
# --------------------------------------------------------------------------------------------------

COLORBAR_MIN = -1.0
COLORBAR_MAX =  1.0

FIGURE_DPI = 500


# ==================================================================================================
# STEP 1 — LOAD DATASET
# ==================================================================================================

ds = xr.open_dataset(NETCDF_FILE)

# Variable name expected in dataset
t850 = ds["t850"]


# ==================================================================================================
# STEP 2 — SELECT SEASONS
# ==================================================================================================

season_mask = False

for season in SEASONS:
    season_mask = season_mask | (ds["season"] == season)

t850_selected = t850.where(season_mask, drop=True)


# ==================================================================================================
# STEP 3 — SELECT CLIMATOLOGICAL PERIOD
# ==================================================================================================

t850_period = t850_selected.sel(
    time=slice(PERIOD_START, PERIOD_END)
)


# ==================================================================================================
# STEP 4 — DEFINE SOUTH AMERICA DOMAIN
# ==================================================================================================

lat_values = ds["latitude"].values
lon_values = ds["longitude"].values

lat_mask = (lat_values >= LAT_MIN) & (lat_values <= LAT_MAX)
lon_mask = (lon_values >= LON_MIN) & (lon_values <= LON_MAX)

lat_selected = lat_values[lat_mask]
lon_selected = lon_values[lon_mask]

lon_grid, lat_grid = np.meshgrid(lon_selected, lat_selected)


# ==================================================================================================
# STEP 5 — COMPUTE MEAN COMPOSITE
# ==================================================================================================

mean_composite = (
    t850_period[:, lat_mask, :][:, :, lon_mask]
    .mean(dim="time")
)

mean_values = mean_composite.values


# ==================================================================================================
# STEP 6 — SPATIAL SMOOTHING
# ==================================================================================================
#
# The griddata interpolation is used only for visual smoothing.
# Original data resolution is preserved in the analysis itself.
# ==================================================================================================

points = np.vstack([
    lon_grid.ravel(),
    lat_grid.ravel()
]).T

values = mean_values.ravel()

smoothed_grid = griddata(
    points,
    values,
    (lon_grid, lat_grid),
    method="linear"
)


# ==================================================================================================
# STEP 7 — GENERATE MAP
# ==================================================================================================

fig, ax = plt.subplots(
    figsize=(10, 10),
    dpi=200,
    subplot_kw={"projection": ccrs.PlateCarree()}
)

ax.set_extent(
    [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    crs=ccrs.PlateCarree()
)


# --------------------------------------------------------------------------------------------------
# Colormap
# --------------------------------------------------------------------------------------------------

cmap_custom = LinearSegmentedColormap.from_list(
    "custom_cmap",
    ["blue", "white", "crimson"]
)


# --------------------------------------------------------------------------------------------------
# Filled contours
# --------------------------------------------------------------------------------------------------

im = ax.pcolormesh(
    lon_grid,
    lat_grid,
    smoothed_grid,
    cmap=cmap_custom,
    shading="gouraud",
    vmin=COLORBAR_MIN,
    vmax=COLORBAR_MAX,
    transform=ccrs.PlateCarree()
)


# --------------------------------------------------------------------------------------------------
# Geographic features
# --------------------------------------------------------------------------------------------------

ax.add_feature(
    cfeature.OCEAN,
    facecolor="white",
    zorder=1
)

ax.add_feature(
    cfeature.LAND,
    facecolor="none",
    edgecolor="black",
    linewidth=1.2,
    zorder=2
)

ax.add_feature(
    cfeature.COASTLINE,
    linewidth=1,
    zorder=3
)

ax.add_feature(
    cfeature.BORDERS,
    linestyle=":",
    zorder=3
)


# --------------------------------------------------------------------------------------------------
# Colorbar
# --------------------------------------------------------------------------------------------------

cbar = plt.colorbar(
    im,
    ax=ax,
    orientation="vertical",
    pad=0.02,
    fraction=0.05
)

cbar.set_label(
    f"Mean standardized T850 anomaly ({PERIOD_START}–{PERIOD_END})"
)


# --------------------------------------------------------------------------------------------------
# Title
# --------------------------------------------------------------------------------------------------

plt.title(
    f"Mean standardized T850 anomalies\n"
    f"{' + '.join(SEASONS)} ({PERIOD_START}–{PERIOD_END})",
    fontsize=12
)


# ==================================================================================================
# STEP 8 — SAVE FIGURE
# ==================================================================================================

output_file = (
    f"{OUTPUT_DIR}/"
    f"mean_t850_{PERIOD_START}_{PERIOD_END}.png"
)

fig.savefig(
    output_file,
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.show()


# ==================================================================================================
# EVENT-BASED COMPOSITE ANALYSIS
# ==================================================================================================
#
# DESCRIPTION
# --------------------------------------------------------------------------------------------------
# This section computes composites using externally supplied event dates.
#
# Event files may represent:
#   - Extreme precipitation events
#   - SOM classifications
#   - Lagged events
#   - Synoptic groups
#   - Teleconnection phases
#   - Any user-defined subset of dates
#
# Multiple files can be combined into a single composite.
#
# ==================================================================================================


# ==================================================================================================
# USER INPUT — EVENT FILES
# ==================================================================================================

EVENT_DATE_FILES = [
    r"PATH_TO_EVENT_FILE_01.txt",
    r"PATH_TO_EVENT_FILE_02.txt",
    r"PATH_TO_EVENT_FILE_03.txt",
]


# --------------------------------------------------------------------------------------------------
# Analysis period for event-based composites
#
# Example:
#   EVENT_PERIOD_START = 2013
#   EVENT_PERIOD_END   = 2024
# --------------------------------------------------------------------------------------------------

EVENT_PERIOD_START = None
EVENT_PERIOD_END   = None


# ==================================================================================================
# STEP 1 — READ EVENT DATES
# ==================================================================================================

all_dates = []

for file in EVENT_DATE_FILES:

    with open(file, "r") as f:

        # Skip optional header
        lines = f.readlines()[1:]

        dates = [
            line.strip()
            for line in lines
            if line.strip()
        ]

        all_dates.extend(dates)


# --------------------------------------------------------------------------------------------------
# Convert to datetime
# --------------------------------------------------------------------------------------------------

event_dates = pd.to_datetime(
    list(set(all_dates)),
    errors="coerce"
)

event_dates = event_dates[~event_dates.isna()]


# --------------------------------------------------------------------------------------------------
# Filter by analysis interval
# --------------------------------------------------------------------------------------------------

event_dates = event_dates[
    (event_dates.year >= EVENT_PERIOD_START) &
    (event_dates.year <= EVENT_PERIOD_END)
]


print(
    f"Number of valid event dates: {len(event_dates)}"
)


# ==================================================================================================
# STEP 2 — SELECT EVENT DATES IN NETCDF
# ==================================================================================================

t850_event = t850.sel(
    time=t850.time.isin(event_dates)
)


# ==================================================================================================
# STEP 3 — COMPUTE EVENT COMPOSITE
# ==================================================================================================

event_mean = t850_event.mean(dim="time")


# ==================================================================================================
# STEP 4 — EXTRACT SOUTH AMERICA REGION
# ==================================================================================================

event_values = (
    event_mean
    .sel(
        latitude=lat_selected,
        longitude=lon_selected
    )
    .values
)


# ==================================================================================================
# STEP 5 — APPLY SPATIAL SMOOTHING
# ==================================================================================================

points = np.vstack([
    lon_grid.ravel(),
    lat_grid.ravel()
]).T

values = event_values.ravel()

event_smoothed = griddata(
    points,
    values,
    (lon_grid, lat_grid),
    method="linear"
)


# ==================================================================================================
# STEP 6 — GENERATE EVENT COMPOSITE MAP
# ==================================================================================================

fig, ax = plt.subplots(
    figsize=(10, 10),
    dpi=200,
    subplot_kw={"projection": ccrs.PlateCarree()}
)

ax.set_extent(
    [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    crs=ccrs.PlateCarree()
)


# --------------------------------------------------------------------------------------------------
# Filled contours
# --------------------------------------------------------------------------------------------------

im = ax.pcolormesh(
    lon_grid,
    lat_grid,
    event_smoothed,
    cmap=cmap_custom,
    shading="gouraud",
    vmin=COLORBAR_MIN,
    vmax=COLORBAR_MAX,
    transform=ccrs.PlateCarree()
)


# --------------------------------------------------------------------------------------------------
# Geographic features
# --------------------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------------------
# Optional reference locations
# --------------------------------------------------------------------------------------------------
#
# Replace SITE01, SITE02, etc. with user-defined coordinates.
# Longitude values should follow the same convention as the dataset.
# --------------------------------------------------------------------------------------------------

site_coordinates = {
    "SITE01": (-75.0, -70.0),
    "SITE02": (-78.0, -65.0),
}

for label, (lat, lon) in site_coordinates.items():

    ax.plot(
        lon,
        lat,
        marker="*",
        color="red",
        markersize=3,
        transform=ccrs.PlateCarree(),
        zorder=10
    )


# --------------------------------------------------------------------------------------------------
# Colorbar
# --------------------------------------------------------------------------------------------------

cbar = plt.colorbar(
    im,
    ax=ax,
    orientation="vertical",
    pad=0.02,
    fraction=0.05
)

cbar.set_label(
    f"Mean standardized T850 anomaly "
    f"({EVENT_PERIOD_START}–{EVENT_PERIOD_END})"
)


# --------------------------------------------------------------------------------------------------
# Title
# --------------------------------------------------------------------------------------------------

plt.title(
    f"Event-based mean standardized T850 anomalies\n"
    f"({EVENT_PERIOD_START}–{EVENT_PERIOD_END})",
    fontsize=12
)


# ==================================================================================================
# STEP 7 — SAVE EVENT FIGURE
# ==================================================================================================

event_output_file = (
    f"{OUTPUT_DIR}/"
    f"event_composite_t850_"
    f"{EVENT_PERIOD_START}_{EVENT_PERIOD_END}.png"
)

fig.savefig(
    event_output_file,
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.show()


print("Composite figures generated successfully.")