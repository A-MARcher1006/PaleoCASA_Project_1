"""
@author: andressa_marcher

@reviewers: write_here

"""

# ==========================================================================================================
# ERA5 TOTAL PRECIPITATION PROCESSING WORKFLOW
# ==========================================================================================================
#
# Description:
# This script processes ERA5 hourly total precipitation (tp) data and generates
# daily standardized precipitation anomaly fields for climate and synoptic analyses.
#
# The workflow is fully adaptable and can be applied to:
#     - Any temporal interval;
#     - Any spatial domain;
#     - Any regridding resolution;
#     - Any precipitation dataset with compatible structure.
#
# Main workflow:
#
# 1. Convert hourly ERA5 precipitation into daily accumulated precipitation;
# 2. (Optional) Subset the dataset to a user-defined spatial domain;
# 3. Regrid precipitation fields to a regular spatial grid;
# 4. Convert precipitation units from meters to millimeters;
# 5. Compute seasonal standardized precipitation anomalies.
#
# Notes:
# - ERA5 total precipitation (tp) is originally provided in meters.
# - Seasonal standardization is performed independently for DJF, MAM, JJA, and SON.
# - Dry days are INCLUDED in the climatological mean and standard deviation
#   calculations.
# - Step 02 (spatial subsetting) is optional and may be skipped if full-domain
#   analyses are desired.
# - Users must define:
#       * Input and output directories;
#       * Spatial domain;
#       * Regridding resolution;
#       * Output filenames.
#
# Main libraries:
# xarray
# numpy
# scipy
# matplotlib
# cartopy
#
# Tested with:
# Python 3.12+ using Spyder IDE
#
#
# ==========================================================================================================

# ==========================================================================================================
# Import libraries
# ==========================================================================================================

import os
from glob import glob
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# ==========================================================================================================
# Step 01 — Convert hourly ERA5 precipitation into daily accumulated precipitation
# ==========================================================================================================

# Define input and output directories
input_dir = r"DEFINE_INPUT_DIRECTORY"
output_dir = r"DEFINE_OUTPUT_DIRECTORY"

os.makedirs(output_dir, exist_ok=True)

# Define output filename
output_file = os.path.join(
    output_dir,
    "DEFINE_OUTPUT_FILENAME.nc"
)

# Collect ERA5 files
file_pattern = os.path.join(
    input_dir,
    "DEFINE_FILE_PATTERN.nc"
)

all_files = sorted(glob(file_pattern))

# Temporary storage for intermediate daily files
daily_files = []

for index, file in enumerate(all_files):

    print(f"Processing: {file}")

    # Lazy loading
    ds = xr.open_dataset(
        file,
        chunks={"time": 100}
    )

    # Verify whether the dataset contains hourly data
    if (
        "valid_time" in ds.coords and
        ds["valid_time"].dt.hour.to_series().nunique() > 1
    ):

        # Convert hourly precipitation into daily accumulated precipitation
        ds_daily = ds.resample(
            valid_time="1D"
        ).sum()

        ds_daily = ds_daily.rename(
            {"valid_time": "time"}
        )

        # Save temporary daily file
        temporary_file = os.path.join(
            output_dir,
            f"daily_{index:04d}.nc"
        )

        ds_daily.to_netcdf(temporary_file)

        daily_files.append(temporary_file)

    else:

        print(f"Skipping file (already daily): {file}")

# Concatenate all daily files
ds_combined = xr.open_mfdataset(
    daily_files,
    combine="by_coords",
    chunks={"time": 100}
)

ds_combined = ds_combined.sortby("time")

# Save final daily dataset
ds_combined.to_netcdf(
    output_file,
    engine="netcdf4"
)

print(f"Daily precipitation dataset saved: {output_file}")

# ==========================================================================================================
# Step 02 — Optional spatial subsetting
# ==========================================================================================================

# This step is optional and may be skipped if full-domain analyses are desired.

# Define input dataset
input_path = r"DEFINE_INPUT_NETCDF_FILE.nc"

# Load dataset
ds = xr.open_dataset(input_path)

# Select precipitation variable
tp = ds["tp"]

# Define spatial domain
# Example:
# Latitude: 20°S–90°S
# Longitude: 180°W–0°

tp_filtered = tp.sel(
    latitude=slice(-20, -90),
    longitude=slice(-180, 0)
)

# Define output file
output_path = r"DEFINE_OUTPUT_FILE.nc"

# Save filtered dataset
tp_filtered.to_netcdf(output_path)

print(f"Filtered dataset saved: {output_path}")

# ==========================================================================================================
# Step 03 — Regrid precipitation data to a regular spatial grid
# ==========================================================================================================

# Define input dataset
input_path = r"DEFINE_INPUT_NETCDF_FILE.nc"

# Load dataset
ds = xr.open_dataset(input_path)

# Extract variables
tp = ds["tp"]

lat = ds["latitude"]
lon = ds["longitude"]
time = ds["time"]

# Source coordinates
lat_source = lat.values
lon_source = lon.values

# Define target grid resolution
# Example:
# 1.5° × 1.5°

lat_target = np.arange(-90, -19.5, 1.5)
lon_target = np.arange(-180, 0.1, 1.5)

# Create target mesh
lon_target_mesh, lat_target_mesh = np.meshgrid(
    lon_target,
    lat_target
)

target_points = np.array([
    lat_target_mesh.ravel(),
    lon_target_mesh.ravel()
]).T

# Empty array for interpolated data
tp_regridded = np.full(
    (
        len(time),
        len(lat_target),
        len(lon_target)
    ),
    np.nan
)

# Interpolate each time step
for t in range(len(time)):

    tp_slice = tp.isel(time=t).values

    if tp_slice.ndim == 2:

        interpolation_function = RegularGridInterpolator(
            (lat_source, lon_source),
            tp_slice,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )

        interpolated = interpolation_function(
            target_points
        ).reshape(
            len(lat_target),
            len(lon_target)
        )

        tp_regridded[t, :, :] = interpolated

    else:

        print(
            f"Skipping time index {t} due to unexpected dimensions: "
            f"{tp_slice.shape}"
        )

# Create regridded dataset
tp_regridded_ds = xr.Dataset(
    {
        "tp": (
            ["time", "latitude", "longitude"],
            tp_regridded
        )
    },
    coords={
        "time": time,
        "latitude": lat_target,
        "longitude": lon_target
    }
)

# Define output file
output_path = r"DEFINE_OUTPUT_FILE.nc"

# Save regridded dataset
tp_regridded_ds.to_netcdf(output_path)

print(f"Regridded dataset saved: {output_path}")

# ==========================================================================================================
# Step 04 — Convert precipitation units from meters to millimeters
# ==========================================================================================================

# Define input dataset
input_path = r"DEFINE_INPUT_NETCDF_FILE.nc"

# Load dataset
ds = xr.open_dataset(input_path)

# Select precipitation variable
tp = ds["tp"]

# Convert precipitation from meters to millimeters
tp_mm = tp * 1000

# Preserve metadata
tp_mm.attrs = tp.attrs
tp_mm.attrs["units"] = "mm"

tp_mm.name = "tp_mm"

# Create dataset
tp_mm_ds = tp_mm.to_dataset()

# Define output file
output_path = r"DEFINE_OUTPUT_FILE.nc"

# Save dataset
tp_mm_ds.to_netcdf(output_path)

print(f"Millimeter-converted dataset saved: {output_path}")

# ==========================================================================================================
# Step 05 — Compute seasonal standardized precipitation anomalies
# ==========================================================================================================

# Define input dataset
file_path = r"DEFINE_INPUT_NETCDF_FILE.nc"

# Load dataset
ds = xr.open_dataset(file_path)

# Select precipitation variable
tp = ds["tp_mm"]

# ----------------------------------------------------------------------------------------------------------
# Define seasonal classification
# ----------------------------------------------------------------------------------------------------------

def get_season(month):

    if month in [12, 1, 2]:
        return "DJF"

    elif month in [3, 4, 5]:
        return "MAM"

    elif month in [6, 7, 8]:
        return "JJA"

    else:
        return "SON"

# Assign seasonal labels
months = tp["time.month"].values

seasons = np.vectorize(get_season)(months)

tp = tp.assign_coords(
    season=("time", seasons)
)

# ----------------------------------------------------------------------------------------------------------
# Seasonal standardization
# ----------------------------------------------------------------------------------------------------------

standardized_seasonal_data = []

for season in ["DJF", "MAM", "JJA", "SON"]:

    # Select season
    tp_season = tp.sel(
        time=tp["season"] == season
    )

    # Compute seasonal climatology including all days
    seasonal_mean = tp_season.mean(
        "time",
        skipna=True
    )

    seasonal_std = tp_season.std(
        "time",
        skipna=True
    )

    # Compute standardized anomalies
    standardized = (
        tp_season - seasonal_mean
    ) / seasonal_std

    standardized_seasonal_data.append(
        standardized
    )

# Concatenate seasonal datasets
tp_standardized = xr.concat(
    standardized_seasonal_data,
    dim="time"
)

# Sort time coordinate
tp_standardized = tp_standardized.sortby("time")

# Rename variable
tp_standardized.name = "tp_mm"

# Define output file
output_path = r"DEFINE_OUTPUT_FILE.nc"

# Save dataset
tp_standardized.to_netcdf(output_path)

print(
    "Seasonal standardized precipitation dataset successfully saved."
)

# ==========================================================================================================