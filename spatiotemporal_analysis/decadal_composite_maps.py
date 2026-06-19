"""
@author: andressa_marcher

@reviewers: write_here

"""

# ==================================================================================================
# Composite Analysis of T850, Z500, Vapor Transport, and Wind Fields During Extreme Events
# ==================================================================================================
#
# Description
# --------------------------------------------------------------------------------------------------
# This script computes and visualizes atmospheric composites associated with selected extreme
# precipitation event (EPE) dates.
#
# The workflow includes:
#
# 1. Loading standardized atmospheric datasets:
#       - Temperature anomalies at 850 hPa (T850)
#       - Geopotential height anomalies at 500 hPa (Z500)
#       - Standardized vapor transport (VT)
#       - 850 hPa wind vectors (u and v components)
#
# 2. Loading event-date files (CSV format) corresponding to selected groups/seasons.
#
# 3. Combining event dates from multiple seasons if desired.
#
# 4. Splitting the event dates into two independent periods:
#       - Period 1 (P1)
#       - Period 2 (P2)
#
# 5. Computing composite mean fields for each period.
#
# 6. Generating publication-quality maps including:
#       - Filled anomaly fields
#       - Z500 contours
#       - Wind vectors
#       - Site markers
#
# --------------------------------------------------------------------------------------------------
# USER INPUTS REQUIRED
# --------------------------------------------------------------------------------------------------
#
# The following sections MUST be configured by the user:
#
# 1. File paths
#       - Atmospheric NetCDF files
#       - Event-date CSV files
#       - Output figure directories
#
# 2. Event groups and seasons
#       Example:
#           Group_1
#           Group_2
#           Summer
#           Autumn
#           Winter
#           Spring
#
# 3. Time intervals
#       Define:
#           P1_START
#           P1_END
#           P2_START
#           P2_END
#
# 4. Site coordinates
#       Replace placeholder coordinates (SITE01, SITE02, etc.)
#       with the desired study-site locations.
#
# --------------------------------------------------------------------------------------------------
# Expected CSV Format
# --------------------------------------------------------------------------------------------------
#
# Each event-date CSV file must contain a column named:
#
#       date
#
# Example:
#
#       date
#       1980-06-15
#       1981-07-22
#       1985-08-10
#
# --------------------------------------------------------------------------------------------------
# Expected NetCDF Variables
# --------------------------------------------------------------------------------------------------
#
# T850 file:
#       Variable name: t850
#
# Z500 file:
#       Variable name: z500
#
# Vapor Transport file:
#       Variable name: VT_standardized
#
# Wind dataset:
#       Variables:
#           u_component
#           v_component
#
# --------------------------------------------------------------------------------------------------
# Spatial Domain
# --------------------------------------------------------------------------------------------------
#
# The script automatically subsets the Southern Hemisphere domain:
#
#       Latitude : 90°S – 20°S
#       Longitude: 180°W – 0°
#
# --------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------
#
# The script produces:
#
# 1. T850 + Z500 composite figures
# 2. Vapor transport + wind vector composite figures
#
# Figures are saved at high resolution (500 dpi).
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ==================================================================================================
# USER INPUTS — FILE PATHS
# ==================================================================================================

# Atmospheric datasets
t850_path = r"DEFINE_PATH_HERE"
z500_path = r"DEFINE_PATH_HERE"

vt_path = r"DEFINE_PATH_HERE"
wind_ds_path = r"DEFINE_PATH_HERE"

# Event-date files (CSV)
season1_file = r"DEFINE_PATH_HERE"
season2_file = r"DEFINE_PATH_HERE"

# Output directory
output_dir = r"DEFINE_OUTPUT_DIRECTORY_HERE"


# ==================================================================================================
# USER INPUTS — GROUP / SEASON INFORMATION
# ==================================================================================================

GROUP_NAME = ""
SEASON_NAME = ""


# ==================================================================================================
# USER INPUTS — TIME PERIODS
# ==================================================================================================

P1_START = ""
P1_END   = ""

P2_START = ""
P2_END   = ""


# ==================================================================================================
# USER INPUTS — SITE COORDINATES
# ==================================================================================================
#
# Replace coordinates with the desired study sites.
#
# Longitude convention:
#       0–360 or -180–180
#
# Ensure consistency with the datasets.
#

site_coords = {

    "SITE01": (-75.00, 270.00),
    "SITE02": (-76.00, 272.00),
    "SITE03": (-77.00, 275.00),
    "SITE04": (-78.00, 278.00),
    "SITE05": (-79.00, 280.00),

}


# ==================================================================================================
# LOAD DATASETS
# ==================================================================================================

print("\nLoading datasets...\n")

# T850 and Z500
t850 = xr.open_dataset(t850_path)["t850"]
z500 = xr.open_dataset(z500_path)["z500"]

# Vapor transport and wind vectors
VT = xr.open_dataset(vt_path)["VT_standardized"]

wind_ds = xr.open_dataset(wind_ds_path)

u = wind_ds["u_component"]
v = wind_ds["v_component"]


# ==================================================================================================
# LONGITUDE CORRECTION
# ==================================================================================================
#
# Converts longitudes from:
#       0–360  -->  -180–180
#

def fix_longitudes(arr):

    if (arr.longitude > 180).any():

        arr = arr.assign_coords(
            longitude=(((arr.longitude + 180) % 360) - 180)
        ).sortby("longitude")

    return arr


t850 = fix_longitudes(t850)
z500 = fix_longitudes(z500)

VT = fix_longitudes(VT)

u = fix_longitudes(u)
v = fix_longitudes(v)


# ==================================================================================================
# SPATIAL SUBSET
# ==================================================================================================

t850 = t850.sel(latitude=slice(-90, -20),
                longitude=slice(-180, 0))

z500 = z500.sel(latitude=slice(-90, -20),
                longitude=slice(-180, 0))

VT = VT.sel(latitude=slice(-90, -20),
            longitude=slice(-180, 0))

u = u.sel(latitude=slice(-90, -20),
          longitude=slice(-180, 0))

v = v.sel(latitude=slice(-90, -20),
          longitude=slice(-180, 0))


# ==================================================================================================
# CONVERT TIME COORDINATES
# ==================================================================================================

t850["time"] = pd.to_datetime(t850.time.values)
z500["time"] = pd.to_datetime(z500.time.values)

VT["time"] = pd.to_datetime(VT.time.values)

u["time"] = pd.to_datetime(u.time.values)
v["time"] = pd.to_datetime(v.time.values)


# ==================================================================================================
# LOAD EVENT DATES
# ==================================================================================================
#
# Multiple seasons/groups may be combined.
#

print("\nLoading event-date files...\n")

season1_dates = pd.read_csv(
    season1_file,
    parse_dates=["date"]
)["date"].dropna()

season2_dates = pd.read_csv(
    season2_file,
    parse_dates=["date"]
)["date"].dropna()

# Remove hour/minute/second information
season1_dates = season1_dates.dt.normalize()
season2_dates = season2_dates.dt.normalize()

# Combine unique dates
epe_dates = pd.Series(
    pd.concat([season1_dates, season2_dates]).unique()
).sort_values()


# ==================================================================================================
# INTERSECT WITH AVAILABLE NETCDF DATES
# ==================================================================================================

time_reference = pd.DatetimeIndex(
    pd.to_datetime(t850.time.values)
).normalize()

valid_dates = pd.DatetimeIndex(
    np.intersect1d(time_reference.values, epe_dates.values)
).normalize()


# ==================================================================================================
# SPLIT DATES INTO TWO PERIODS
# ==================================================================================================

print("\nSplitting dates into periods...\n")

p1_dates = valid_dates[
    (valid_dates >= P1_START) &
    (valid_dates <= P1_END)
]

p2_dates = valid_dates[
    (valid_dates >= P2_START) &
    (valid_dates <= P2_END)
]

mask_p1 = time_reference.isin(p1_dates)
mask_p2 = time_reference.isin(p2_dates)

if not mask_p1.any():
    raise ValueError("No valid dates found for Period 1")

if not mask_p2.any():
    raise ValueError("No valid dates found for Period 2")


# ==================================================================================================
# COMPUTE COMPOSITES
# ==================================================================================================

print("\nComputing composites...\n")

# ----------------------------------------
# T850 + Z500
# ----------------------------------------

t850_p1 = t850.isel(time=mask_p1).mean(dim="time")
z500_p1 = z500.isel(time=mask_p1).mean(dim="time")

t850_p2 = t850.isel(time=mask_p2).mean(dim="time")
z500_p2 = z500.isel(time=mask_p2).mean(dim="time")

# ----------------------------------------
# Vapor Transport + Wind
# ----------------------------------------

VT_p1 = VT.isel(time=mask_p1).mean(dim="time")
u_p1  = u.isel(time=mask_p1).mean(dim="time")
v_p1  = v.isel(time=mask_p1).mean(dim="time")

VT_p2 = VT.isel(time=mask_p2).mean(dim="time")
u_p2  = u.isel(time=mask_p2).mean(dim="time")
v_p2  = v.isel(time=mask_p2).mean(dim="time")


# ==================================================================================================
# PLOT FUNCTION — T850 + Z500
# ==================================================================================================

def plot_t850_z500(
    t850_comp,
    z500_comp,
    title,
    output_file
):

    fig, ax = plt.subplots(
        figsize=(5, 3),
        subplot_kw={
            "projection": ccrs.Orthographic(
                central_longitude=-90,
                central_latitude=-60
            )
        }
    )

    ax.spines["geo"].set_visible(False)

    # ------------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------------

    colors = ["blue", "white", "crimson"]

    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red",
        colors,
        N=256
    )

    levels = np.arange(-1.5, 1.6, 0.1)

    norm = TwoSlopeNorm(
        vcenter=0,
        vmin=-1.5,
        vmax=1.5
    )

    # ------------------------------------------------------------------
    # Base map
    # ------------------------------------------------------------------

    ax.set_extent(
        [-180, 0, -90, -20],
        crs=ccrs.PlateCarree()
    )

    ax.coastlines()

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.3
    )

    ax.gridlines(
        draw_labels=False,
        linewidth=0.3
    )

    # ------------------------------------------------------------------
    # Filled contours
    # ------------------------------------------------------------------

    im = ax.contourf(
        t850_comp.longitude,
        t850_comp.latitude,
        t850_comp,

        levels=levels,
        cmap=cmap,
        norm=norm,

        transform=ccrs.PlateCarree(),
        extend="both"
    )

    # ------------------------------------------------------------------
    # Z500 contours
    # ------------------------------------------------------------------

    z_levels = np.arange(-1.0, 1.1, 0.1)

    neg = z_levels[z_levels < 0]
    pos = z_levels[z_levels > 0]

    neg_contour = ax.contour(
        z500_comp.longitude,
        z500_comp.latitude,
        z500_comp,

        levels=neg,
        colors="black",
        alpha=0.5,

        linestyles="dashed",
        linewidths=0.8,

        transform=ccrs.PlateCarree()
    )

    pos_contour = ax.contour(
        z500_comp.longitude,
        z500_comp.latitude,
        z500_comp,

        levels=pos,
        colors="black",
        alpha=0.5,

        linestyles="solid",
        linewidths=0.8,

        transform=ccrs.PlateCarree()
    )

    ax.clabel(neg_contour, fontsize=5, fmt="%.1f")
    ax.clabel(pos_contour, fontsize=5, fmt="%.1f")

    # ------------------------------------------------------------------
    # Plot sites
    # ------------------------------------------------------------------

    for label, (lat, lon) in site_coords.items():

        ax.plot(
            lon,
            lat,

            marker="*",
            color="red",
            markersize=3,

            transform=ccrs.PlateCarree(),
            zorder=11
        )

    # ------------------------------------------------------------------
    # Title and colorbar
    # ------------------------------------------------------------------

    ax.set_title(title, fontsize=7)

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        shrink=0.7,
        pad=0.05
    )

    cbar.set_label(
        "Standardized Temperature Anomaly at 850 hPa",
        fontsize=6
    )

    cbar.ax.tick_params(labelsize=5)

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------

    plt.savefig(
        output_file,
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.show()


# ==================================================================================================
# PLOT FUNCTION — VAPOR TRANSPORT + WIND
# ==================================================================================================

def plot_vt_composite(
    VT_comp,
    u_comp,
    v_comp,
    title,
    output_file
):

    fig, ax = plt.subplots(
        figsize=(4.5, 4.5),
        subplot_kw={
            "projection": ccrs.PlateCarree()
        }
    )

    # ------------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------------

    colors = ["#8B4513", "white", "#436EEE"]

    cmap = LinearSegmentedColormap.from_list(
        "brown_white_blue",
        colors,
        N=256
    )

    levels = np.arange(-1.5, 1.6, 0.1)

    norm = TwoSlopeNorm(
        vcenter=0,
        vmin=-1.5,
        vmax=1.5
    )

    # ------------------------------------------------------------------
    # Base map
    # ------------------------------------------------------------------

    ax.set_extent(
        [-180, 0, -90, -20],
        crs=ccrs.PlateCarree()
    )

    ax.coastlines(linewidth=0.5)

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.3
    )

    ax.gridlines(
        draw_labels=False,
        linewidth=0.3,
        color="gray",
        linestyle="--"
    )

    ax.set_aspect("equal", adjustable="box")

    # ------------------------------------------------------------------
    # Vapor transport field
    # ------------------------------------------------------------------

    im = ax.contourf(
        VT_comp.longitude,
        VT_comp.latitude,
        VT_comp,

        levels=levels,
        cmap=cmap,
        norm=norm,

        transform=ccrs.PlateCarree(),
        extend="both"
    )

    # ------------------------------------------------------------------
    # Wind vectors
    # ------------------------------------------------------------------

    step = 3

    lon2d, lat2d = np.meshgrid(
        u_comp.longitude,
        u_comp.latitude
    )

    Q = ax.quiver(
        lon2d[::step, ::step],
        lat2d[::step, ::step],

        u_comp.values[::step, ::step],
        v_comp.values[::step, ::step],

        transform=ccrs.PlateCarree(),

        scale=200,
        width=0.002,

        headwidth=3,
        headlength=4,
        headaxislength=3,

        color="k",
        alpha=0.6,
        pivot="middle"
    )

    qk = ax.quiverkey(
        Q,
        0.90,
        1.05,
        10,

        label="10 m/s",
        labelpos="E",
        coordinates="axes"
    )

    qk.text.set_fontsize(6)

    # ------------------------------------------------------------------
    # Plot sites
    # ------------------------------------------------------------------

    for label, (lat, lon) in site_coords.items():

        ax.plot(
            lon,
            lat,

            marker="*",
            color="red",
            markersize=3,

            transform=ccrs.PlateCarree(),
            zorder=11
        )

    # ------------------------------------------------------------------
    # Title and colorbar
    # ------------------------------------------------------------------

    ax.set_title(title, fontsize=7)

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        shrink=0.7,
        pad=0.05
    )

    cbar.set_label(
        "Standardized Vapor Transport at 850 hPa",
        fontsize=6
    )

    cbar.ax.tick_params(labelsize=5)

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------

    plt.savefig(
        output_file,
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.show()


# ==================================================================================================
# GENERATE FIGURES
# ==================================================================================================

print("\nGenerating figures...\n")

# ----------------------------------------
# T850 + Z500
# ----------------------------------------

plot_t850_z500(
    t850_p1,
    z500_p1,

    f"T850 + Z500 Composite | {GROUP_NAME} | Period 1",

    rf"{output_dir}\T850_Z500_P1.png"
)

plot_t850_z500(
    t850_p2,
    z500_p2,

    f"T850 + Z500 Composite | {GROUP_NAME} | Period 2",

    rf"{output_dir}\T850_Z500_P2.png"
)

# ----------------------------------------
# Vapor Transport + Wind
# ----------------------------------------

plot_vt_composite(
    VT_p1,
    u_p1,
    v_p1,

    f"VT Composite | {GROUP_NAME} | Period 1",

    rf"{output_dir}\VT_P1.png"
)

plot_vt_composite(
    VT_p2,
    u_p2,
    v_p2,

    f"VT Composite | {GROUP_NAME} | Period 2",

    rf"{output_dir}\VT_P2.png"
)

print("\n✅ Composite analysis completed successfully.\n")