"""
@author: andressa_marcher

@reviewers: write_here

"""

# ==========================================================================================================
# ERA5 Z500 PROCESSING WORKFLOW
# ==========================================================================================================
#
# Description:
# This script processes ERA5 hourly geopotential height (Z500) data and prepares
# daily standardized fields for climatological and synoptic analyses.
#
# The workflow is fully adaptable and can be applied to:
#     - Any temporal interval;
#     - Any spatial domain;
#     - Any atmospheric pressure level;
#     - Any regridding resolution.
#
# Main workflow:
#
# 1. Convert hourly ERA5 Z500 data into daily means;
# 2. (Optional) Filter the dataset to a user-defined spatial domain;
# 3. Regrid the data to a regular spatial grid;
# 4. Convert geopotential (m² s⁻²) to geopotential height (m);
# 5. Compute seasonal standardized anomalies at each grid point.
#
# Notes:
# - ERA5 geopotential ('z') is originally provided in m² s⁻².
# - Conversion to geopotential height is performed using:
#
#       height = z / 9.80665
#
# - The spatial filtering step is optional and may be skipped if global fields
#   are required.
# - Seasonal standardization is performed independently for DJF, MAM, JJA, and SON.
# - Users must define:
#       * Input and output directories;
#       * Spatial domain;
#       * Regridding resolution;
#       * Pressure level;
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
# Author:
# Andressa Marcher
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
# STEP 01 — Convert hourly ERA5 Z500 data to daily means
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

# Temporary list for daily files
daily_files = []

for index, file in enumerate(all_files):

    print(f"Processing: {file}")

    ds = xr.open_dataset(
        file,
        chunks={"time": 100}
    )

    # Verify whether the dataset contains hourly data
    if (
        "valid_time" in ds.coords and
        ds["valid_time"].dt.hour.to_series().nunique() > 1
    ):

        # Compute daily means
        ds_daily = ds.resample(
            valid_time="1D"
        ).mean()

        # Rename coordinate
        ds_daily = ds_daily.rename(
            {"valid_time": "time"}
        )

        # Save temporary daily file
        daily_path = os.path.join(
            output_dir,
            f"daily_{index:04d}.nc"
        )

        ds_daily.to_netcdf(daily_path)

        daily_files.append(daily_path)

    else:

        print(f"Skipping file (already daily?): {file}")

# Merge all daily datasets
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

print(f"Daily Z500 dataset saved to: {output_file}")

# ==========================================================================================================
# STEP 02 — OPTIONAL: Filter dataset to a selected spatial domain
# ==========================================================================================================

# This step is optional and may be skipped if global fields are required.

# Define input dataset
input_path = r"DEFINE_INPUT_NETCDF_FILE.nc"

# Load dataset
ds = xr.open_dataset(input_path)

# Select geopotential variable
z500 = ds["z"]

# Define spatial domain
# Example:
# Latitude: 20°S–90°S
# Longitude: 180°W–0°

z500_filtered = z500.sel(
    latitude=slice(-20, -90),
    longitude=slice(-180, 0)
)

# Define output file
output_path = r"DEFINE_OUTPUT_FILE.nc"

# Save filtered dataset
z500_filtered.to_netcdf(output_path)

print(f"Filtered dataset saved to: {output_path}")

# Optional validation
ds_test = xr.open_dataset(output_path)

sample = ds_test["z"].sel(
    latitude=-60.0,
    longitude=-90.0
)

print(sample.values)

# ==========================================================================================================
# STEP 03 — Regrid Z500 data to a regular spatial grid
# ==========================================================================================================

# Define input dataset
input_path = r"DEFINE_INPUT_NETCDF_FILE.nc"

# Load dataset
ds = xr.open_dataset(input_path)

# Remove singleton pressure dimension if necessary
z500 = ds["z"].squeeze("pressure_level")

lat_source = ds.latitude.values
lon_source = ds.longitude.values
time = ds.time

print("Original shape:", z500.shape)

# Define target grid resolution
# Example:
# 1.5° × 1.5°

lat_target = np.arange(-90, -19.5, 1.5)
lon_target = np.arange(-180, 0.1, 1.5)

# Generate target coordinate mesh
lon_mesh, lat_mesh = np.meshgrid(
    lon_target,
    lat_target
)

target_points = np.array([
    lat_mesh.ravel(),
    lon_mesh.ravel()
]).T

# Initialize output array
z500_regridded = np.full(
    (
        len(time),
        len(lat_target),
        len(lon_target)
    ),
    np.nan
)

# Interpolate each time slice
for t in range(len(time)):

    z_slice = z500.isel(time=t).values

    if z_slice.ndim == 2:

        interpolator = RegularGridInterpolator(
            (lat_source, lon_source),
            z_slice,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )

        interpolated = interpolator(
            target_points
        ).reshape(
            len(lat_target),
            len(lon_target)
        )

        z500_regridded[t, :, :] = interpolated

    else:

        print(
            f"Skipping time index {t} "
            f"due to unexpected dimensions: {z_slice.shape}"
        )

# Create regridded dataset
z500_regridded_ds = xr.Dataset(

    {
        "z": (
            ["time", "latitude", "longitude"],
            z500_regridded
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
z500_regridded_ds.to_netcdf(output_path)

print(f"Regridded dataset saved to: {output_path}")

# ==========================================================================================================
# STEP 04 — Convert geopotential to geopotential height (meters)
# ==========================================================================================================

# Define input and output files
input_path = r"DEFINE_INPUT_NETCDF_FILE.nc"
output_path = r"DEFINE_OUTPUT_FILE.nc"

# Load dataset
ds = xr.open_dataset(input_path)

# Convert geopotential to geopotential height
ds["z"] = ds["z"] / 9.80665

# Update metadata
ds["z"].attrs["units"] = "meters"

ds["z"].attrs["long_name"] = (
    "500 hPa geopotential height"
)

# Save converted dataset
ds.to_netcdf(output_path)

print(f"Converted dataset saved to: {output_path}")

# ==========================================================================================================
# STEP 05 — Compute seasonal standardized anomalies
# ==========================================================================================================

# Define input dataset
file_path = r"DEFINE_INPUT_NETCDF_FILE.nc"

# Load dataset
ds = xr.open_dataset(file_path)

# Select geopotential height variable
z500 = ds["z"]

# ----------------------------------------------------------------------------------------------------------
# Define climatological seasons
# ----------------------------------------------------------------------------------------------------------

def assign_season(month):

    if month in [12, 1, 2]:
        return "DJF"

    elif month in [3, 4, 5]:
        return "MAM"

    elif month in [6, 7, 8]:
        return "JJA"

    else:
        return "SON"

# Assign seasonal labels
months = z500["time.month"].values

seasons = np.vectorize(assign_season)(months)

z500 = z500.assign_coords(
    season=("time", seasons)
)

# ----------------------------------------------------------------------------------------------------------
# Standardize each season independently
# ----------------------------------------------------------------------------------------------------------

standardized_data = []

season_order = ["DJF", "MAM", "JJA", "SON"]

for season in season_order:

    # Select season
    z_season = z500.sel(
        time=z500["season"] == season
    )

    # Compute seasonal climatology
    seasonal_mean = z_season.mean(
        "time",
        skipna=True
    )

    seasonal_std = z_season.std(
        "time",
        skipna=True
    )

    # Standardize anomalies
    z_standardized = (
        z_season - seasonal_mean
    ) / seasonal_std

    standardized_data.append(
        z_standardized
    )

# Recombine seasonal datasets
z500_standardized = xr.concat(
    standardized_data,
    dim="time"
)

# Sort chronologically
z500_standardized = z500_standardized.sortby("time")

# Rename variable
z500_standardized.name = "z500"

# Define output file
output_path = r"DEFINE_OUTPUT_FILE.nc"

# Save standardized dataset
z500_standardized.to_netcdf(output_path)

print(
    "Seasonal standardization completed successfully."
)

# ==========================================================================================================