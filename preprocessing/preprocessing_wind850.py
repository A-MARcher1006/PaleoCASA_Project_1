"""
@author: andressa_marcher

@reviewers: write_here

"""
# ==========================================================================================================
# ERA5 850 hPa Wind Processing Workflow
#
# Description:
# This script processes ERA5 850 hPa zonal (u) and meridional (v) wind fields.
#
# The workflow includes:
#
# 1. Computing daily mean wind fields from hourly ERA5 data;
# 2. Optionally filtering the data to a selected spatial domain;
# 3. Regridding datasets to a regular 1.5° × 1.5° grid;
# 4. Computing wind speed and wind direction fields;
# 5. Exporting processed datasets to NetCDF format.
#
# Notes:
# - The spatial filtering step is optional and may be skipped if global or
#   full-domain analyses are required.
#
# - Input ERA5 datasets are assumed to contain hourly data and a valid_time
#   coordinate.
#
# - Wind direction follows the meteorological convention
#   (direction from which the wind originates).
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
# - glob
# - os
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
# Section 1 — Meridional Wind (v850)
# ==========================================================================================================

# ----------------------------------------------------------------------------------------------------------
# Step 1 — Compute daily mean v850 fields from hourly ERA5 data
# ----------------------------------------------------------------------------------------------------------

# Define input/output paths
input_dir = r"/path/to/hourly_ERA5_wind_files/"
output_dir = r"/path/to/output_daily_files/"

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "v850_ERA5_daily.nc"
)

# Collect all ERA5 files
file_pattern = os.path.join(
    input_dir,
    "ERA5_data_v850_*.nc"
)

all_files = sorted(glob(file_pattern))

# Temporary storage for intermediate daily files
daily_files = []

for index, file in enumerate(all_files):

    print(f"Processing {file}...")

    ds = xr.open_dataset(
        file,
        chunks={"time": 100}
    )

    # Verify that dataset contains hourly data
    if (
        "valid_time" in ds.coords and
        ds["valid_time"].dt.hour.to_series().nunique() > 1
    ):

        # Compute daily mean
        ds_daily = ds.resample(valid_time="1D").mean()

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

# Concatenate all daily datasets
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

print(f"Daily v850 dataset saved to: {output_file}")

# ==========================================================================================================
# Step 2 — Optional spatial filtering
# ==========================================================================================================

# NOTE:
# This step is optional and may be skipped if the full ERA5 spatial domain
# is required for subsequent analyses.

# Load dataset
input_path = r"/path/to/daily_v850_dataset.nc"

ds = xr.open_dataset(input_path)

# Select meridional wind variable
v850 = ds["v"]

# Example subset domain:
# 20°S–90°S and 180°W–0°

v850_filtered = v850.sel(
    latitude=slice(-20, -90),
    longitude=slice(-180, 0)
)

# Save filtered dataset
output_path = r"/path/to/output_filtered_v850_dataset.nc"

v850_filtered.to_netcdf(output_path)

print(f"Filtered v850 dataset saved to: {output_path}")

# Optional grid-point verification
ds = xr.open_dataset(output_path)

v850_point = ds["v"].sel(
    latitude=-60.0,
    longitude=-90.0
)

print("Sample values:", v850_point.values)

# ==========================================================================================================
# Step 3 — Regrid v850 to a 1.5° × 1.5° grid
# ==========================================================================================================

# Load dataset
input_path = r"/path/to/output_filtered_v850_dataset.nc"

ds = xr.open_dataset(input_path)

# Remove singleton pressure-level dimension
v850 = ds["v"].squeeze("pressure_level")

lat_source = ds.latitude.values
lon_source = ds.longitude.values
time = ds.time

print("Original v850 shape:", v850.shape)

# Define target grid
lat_target = np.arange(-90, -19.5, 1.5)
lon_target = np.arange(-180, 0.1, 1.5)

# Create target mesh
lon_mesh, lat_mesh = np.meshgrid(
    lon_target,
    lat_target
)

target_points = np.array([
    lat_mesh.ravel(),
    lon_mesh.ravel()
]).T

# Allocate output array
v850_regridded = np.full(
    (
        len(time),
        len(lat_target),
        len(lon_target)
    ),
    np.nan
)

# Interpolate each time slice
for t in range(len(time)):

    v_slice = v850.isel(time=t).values

    if v_slice.ndim == 2:

        interpolator = RegularGridInterpolator(
            (lat_source, lon_source),
            v_slice,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )

        interpolated = interpolator(target_points)

        interpolated = interpolated.reshape(
            len(lat_target),
            len(lon_target)
        )

        v850_regridded[t, :, :] = interpolated

    else:

        print(
            f"Skipping time index {t} "
            f"due to unexpected shape: {v_slice.shape}"
        )

# Create output dataset
v850_regridded_ds = xr.Dataset(
    {
        "v": (
            ["time", "latitude", "longitude"],
            v850_regridded
        )
    },
    coords={
        "time": time,
        "latitude": lat_target,
        "longitude": lon_target
    }
)

# Save dataset
output_path = r"/path/to/output_regridded_v850_dataset.nc"

v850_regridded_ds.to_netcdf(output_path)

print(f"Regridded v850 dataset saved to: {output_path}")

# Optional verification
ds = xr.open_dataset(output_path)

v850_point = ds["v"].sel(
    latitude=-60.0,
    longitude=-90.0
)

print("Regridded values:", v850_point.values)

# ==========================================================================================================
# Section 2 — Zonal Wind (u850)
# ==========================================================================================================

# ----------------------------------------------------------------------------------------------------------
# Step 4 — Compute daily mean u850 fields from hourly ERA5 data
# ----------------------------------------------------------------------------------------------------------

# Define output file
output_file = os.path.join(
    output_dir,
    "u850_ERA5_daily.nc"
)

# Collect ERA5 files
file_pattern = os.path.join(
    input_dir,
    "ERA5_data_u850_*.nc"
)

all_files = sorted(glob(file_pattern))

daily_files = []

for index, file in enumerate(all_files):

    print(f"Processing {file}...")

    ds = xr.open_dataset(
        file,
        chunks={"time": 100}
    )

    if (
        "valid_time" in ds.coords and
        ds["valid_time"].dt.hour.to_series().nunique() > 1
    ):

        ds_daily = ds.resample(valid_time="1D").mean()

        ds_daily = ds_daily.rename(
            {"valid_time": "time"}
        )

        daily_path = os.path.join(
            output_dir,
            f"daily_{index:04d}.nc"
        )

        ds_daily.to_netcdf(daily_path)

        daily_files.append(daily_path)

    else:

        print(f"Skipping file (already daily?): {file}")

# Concatenate datasets
ds_combined = xr.open_mfdataset(
    daily_files,
    combine="by_coords",
    chunks={"time": 100}
)

ds_combined = ds_combined.sortby("time")

# Save dataset
ds_combined.to_netcdf(
    output_file,
    engine="netcdf4"
)

print(f"Daily u850 dataset saved to: {output_file}")

# ==========================================================================================================
# Step 5 — Optional spatial filtering
# ==========================================================================================================

# NOTE:
# This step is optional and may be skipped if full-domain analyses are desired.

# Load dataset
input_path = r"/path/to/daily_u850_dataset.nc"

ds = xr.open_dataset(input_path)

# Select zonal wind variable
u850 = ds["u"]

# Example subset region
u850_filtered = u850.sel(
    latitude=slice(-20, -90),
    longitude=slice(-180, 0)
)

# Save filtered dataset
output_path = r"/path/to/output_filtered_u850_dataset.nc"

u850_filtered.to_netcdf(output_path)

print(f"Filtered u850 dataset saved to: {output_path}")

# Optional verification
ds = xr.open_dataset(output_path)

u850_point = ds["u"].sel(
    latitude=-60.0,
    longitude=-90.0
)

print("Sample values:", u850_point.values)

# ==========================================================================================================
# Step 6 — Regrid u850 to a 1.5° × 1.5° grid
# ==========================================================================================================

# Load dataset
input_path = r"/path/to/output_filtered_u850_dataset.nc"

ds = xr.open_dataset(input_path)

# Remove singleton pressure-level dimension
u850 = ds["u"].squeeze("pressure_level")

lat_source = ds.latitude.values
lon_source = ds.longitude.values
time = ds.time

print("Original u850 shape:", u850.shape)

# Define target grid
lat_target = np.arange(-90, -19.5, 1.5)
lon_target = np.arange(-180, 0.1, 1.5)

# Create mesh
lon_mesh, lat_mesh = np.meshgrid(
    lon_target,
    lat_target
)

target_points = np.array([
    lat_mesh.ravel(),
    lon_mesh.ravel()
]).T

# Allocate output array
u850_regridded = np.full(
    (
        len(time),
        len(lat_target),
        len(lon_target)
    ),
    np.nan
)

# Interpolate data
for t in range(len(time)):

    u_slice = u850.isel(time=t).values

    if u_slice.ndim == 2:

        interpolator = RegularGridInterpolator(
            (lat_source, lon_source),
            u_slice,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )

        interpolated = interpolator(target_points)

        interpolated = interpolated.reshape(
            len(lat_target),
            len(lon_target)
        )

        u850_regridded[t, :, :] = interpolated

    else:

        print(
            f"Skipping time index {t} "
            f"due to unexpected shape: {u_slice.shape}"
        )

# Create output dataset
u850_regridded_ds = xr.Dataset(
    {
        "u": (
            ["time", "latitude", "longitude"],
            u850_regridded
        )
    },
    coords={
        "time": time,
        "latitude": lat_target,
        "longitude": lon_target
    }
)

# Save dataset
output_path = r"/path/to/output_regridded_u850_dataset.nc"

u850_regridded_ds.to_netcdf(output_path)

print(f"Regridded u850 dataset saved to: {output_path}")

# Optional verification
ds = xr.open_dataset(output_path)

u850_point = ds["u"].sel(
    latitude=-60.0,
    longitude=-90.0
)

print("Regridded values:", u850_point.values)

# ==========================================================================================================
# Section 3 — Wind Vector Computation
# ==========================================================================================================

# ----------------------------------------------------------------------------------------------------------
# Step 7 — Compute wind speed and direction
# ----------------------------------------------------------------------------------------------------------

# Load regridded datasets
u850_ds = xr.open_dataset(
    r"/path/to/output_regridded_u850_dataset.nc"
)

v850_ds = xr.open_dataset(
    r"/path/to/output_regridded_v850_dataset.nc"
)

u850 = u850_ds["u"]
v850 = v850_ds["v"]

# Convert longitude convention if required
if (u850.longitude > 180).any():

    u850 = u850.assign_coords(
        longitude=(
            ((u850.longitude + 180) % 360) - 180
        )
    ).sortby("longitude")

    v850 = v850.assign_coords(
        longitude=(
            ((v850.longitude + 180) % 360) - 180
        )
    ).sortby("longitude")

# Compute wind speed
wind_speed = np.sqrt(
    u850**2 + v850**2
)

# Compute meteorological wind direction
wind_direction = (
    np.arctan2(-u850, -v850) * 180 / np.pi
) % 360

# Create dataset
wind_dataset = xr.Dataset({

    "wind_speed": wind_speed,
    "wind_direction": wind_direction,
    "u_component": u850,
    "v_component": v850

})

# Metadata
wind_dataset["wind_speed"].attrs["units"] = "m s-1"

wind_dataset["wind_direction"].attrs["units"] = (
    "degrees_from_north"
)

wind_dataset["u_component"].attrs["description"] = (
    "Zonal wind component"
)

wind_dataset["v_component"].attrs["description"] = (
    "Meridional wind component"
)

# Save dataset
output_path = r"/path/to/output_wind_vector_dataset.nc"

wind_dataset.to_netcdf(output_path)

print(f"Wind vector dataset saved to: {output_path}")

# ==========================================================================================================