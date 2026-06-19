"""
@author: andressa_marcher

@reviewers: write_here

"""
# ==========================================================================================================
# ERA5 850 hPa Temperature (T850) Processing Workflow
#
# Description:
# This script processes ERA5 hourly 850 hPa air temperature data and prepares
# daily standardized fields for climatological and synoptic analyses.
#
# The workflow includes:
#
# 1. Converting hourly ERA5 T850 data into daily means;
# 2. (Optional) Filtering the dataset to a selected spatial domain;
# 3. Regridding the data to a regular 1.5° × 1.5° grid;
# 4. Converting temperature from Kelvin to Celsius;
# 5. Computing seasonal standardized anomalies at each grid point.
#
# Notes:
# - ERA5 temperature ('t') is originally provided in Kelvin.
# - Conversion to Celsius is performed using:
#
#       T(°C) = T(K) - 273.15
#
# - The spatial filtering step is optional and may be skipped if global fields
#   are required.
#
# - Users must define:
#       * Input directories;
#       * Output directories;
#       * Spatial domain;
#       * Variable names;
#       * Target regridding resolution.
#
# Required libraries:
# - xarray
# - numpy
# - scipy
# - matplotlib
# - cartopy
#
# Tested with:
# Python 3.12+ using Spyder IDE
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
# STEP 01 — Convert hourly ERA5 T850 data to daily means
# ==========================================================================================================

# Define input and output directories
input_dir = r"/path/to/hourly_ERA5_temperature_files/"
output_dir = r"/path/to/output_daily_files/"

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "t850_ERA5_daily.nc"
)

# Collect all ERA5 T850 files
file_pattern = os.path.join(
    input_dir,
    "ERA5_data_t850_*.nc"
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

    # Verify that the dataset contains hourly data
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

print(f"Daily T850 dataset saved to: {output_file}")

# ==========================================================================================================
# STEP 02 — OPTIONAL: Filter dataset to a selected spatial domain
# ==========================================================================================================

# This step is optional and may be skipped if global fields are required.

# Load daily T850 dataset
input_path = r"/path/to/daily_temperature_dataset.nc"

ds = xr.open_dataset(input_path)

# Select temperature variable
t850 = ds["t"]

# Example spatial subset:
# Latitude: 20°S–90°S
# Longitude: 180°W–0°

t850_filtered = t850.sel(
    latitude=slice(-20, -90),
    longitude=slice(-180, 0)
)

# Save filtered dataset
output_path = r"/path/to/output_filtered_temperature_dataset.nc"

t850_filtered.to_netcdf(output_path)

print(f"Filtered dataset saved to: {output_path}")

# Optional validation
ds_test = xr.open_dataset(output_path)

sample = ds_test["t"].sel(
    latitude=-60.0,
    longitude=-90.0
)

print(sample.values)

# ==========================================================================================================
# STEP 03 — Regrid T850 data to a regular 1.5° × 1.5° grid
# ==========================================================================================================

# Load filtered dataset
input_path = r"/path/to/output_filtered_temperature_dataset.nc"

ds = xr.open_dataset(input_path)

# Remove singleton pressure-level dimension
t850 = ds["t"].squeeze("pressure_level")

lat_source = ds.latitude.values
lon_source = ds.longitude.values
time = ds.time

print("Original shape:", t850.shape)

# Define target grid
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
t850_regridded = np.full(
    (
        len(time),
        len(lat_target),
        len(lon_target)
    ),
    np.nan
)

# Interpolate each time slice
for t in range(len(time)):

    t_slice = t850.isel(time=t).values

    if t_slice.ndim == 2:

        interpolator = RegularGridInterpolator(
            (lat_source, lon_source),
            t_slice,
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

        t850_regridded[t, :, :] = interpolated

    else:

        print(
            f"Skipping time index {t} "
            f"due to unexpected dimensions: {t_slice.shape}"
        )

# Create regridded dataset
t850_regridded_ds = xr.Dataset(

    {
        "t": (
            ["time", "latitude", "longitude"],
            t850_regridded
        )
    },

    coords={
        "time": time,
        "latitude": lat_target,
        "longitude": lon_target
    }
)

# Save regridded dataset
output_path = r"/path/to/output_regridded_temperature_dataset.nc"

t850_regridded_ds.to_netcdf(output_path)

print(f"Regridded dataset saved to: {output_path}")

# ==========================================================================================================
# STEP 04 — Convert temperature from Kelvin to Celsius
# ==========================================================================================================

input_path = r"/path/to/output_regridded_temperature_dataset.nc"

output_path = r"/path/to/output_temperature_dataset_celsius.nc"

# Load dataset
ds = xr.open_dataset(input_path)

# Convert temperature
ds["t"] = ds["t"] - 273.15

# Update metadata
ds["t"].attrs["units"] = "°C"
ds["t"].attrs["long_name"] = (
    "850 hPa air temperature"
)

# Save converted dataset
ds.to_netcdf(output_path)

print(f"Converted dataset saved to: {output_path}")

# ==========================================================================================================
# STEP 05 — Compute seasonal standardized anomalies
# ==========================================================================================================

# Load temperature dataset
file_path = r"/path/to/output_temperature_dataset_celsius.nc"

ds = xr.open_dataset(file_path)

t850 = ds["t"]

# ----------------------------------------------------------------------------------------------
# Define climatological seasons
# ----------------------------------------------------------------------------------------------

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
months = t850["time.month"].values

seasons = np.vectorize(assign_season)(months)

t850 = t850.assign_coords(
    season=("time", seasons)
)

# ----------------------------------------------------------------------------------------------
# Standardize each season independently
# ----------------------------------------------------------------------------------------------

standardized_data = []

season_order = ["DJF", "MAM", "JJA", "SON"]

for season in season_order:

    # Select season
    t_season = t850.sel(
        time=t850["season"] == season
    )

    # Compute seasonal climatology
    seasonal_mean = t_season.mean(
        "time",
        skipna=True
    )

    seasonal_std = t_season.std(
        "time",
        skipna=True
    )

    # Standardize
    t_standardized = (
        t_season - seasonal_mean
    ) / seasonal_std

    standardized_data.append(
        t_standardized
    )

# Recombine seasons
t850_standardized = xr.concat(
    standardized_data,
    dim="time"
)

# Sort chronologically
t850_standardized = t850_standardized.sortby("time")

# Rename variable
t850_standardized.name = "t850"

# Save standardized dataset
output_path = r"/path/to/output_standardized_temperature_dataset.nc"

t850_standardized.to_netcdf(output_path)

print(
    "Seasonal standardization completed successfully."
)

# ==========================================================================================================