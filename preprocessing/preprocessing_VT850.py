"""
@author: andressa_marcher

@reviewers: write_here

"""

# ==========================================================================================================
# ERA5 Vapor Transport (VT) Processing Workflow
#
# Description:
# This script computes daily vapor transport (VT) at 850 hPa from ERA5
# specific humidity, temperature, and wind components. The workflow includes:
#
# 1. Computing daily VT fields from hourly ERA5 data;
# 2. (Optional) Filtering data to a selected spatial domain;
# 3. Removing unnecessary singleton dimensions;
# 4. Regridding data to a regular 1.5° × 1.5° grid;
# 5. Performing quality-control checks;
# 6. Standardizing VT fields using seasonal climatology.
#
# Notes:
# - ERA5 hourly NetCDF files are expected for:
#       q850 (specific humidity)
#       u850 (zonal wind)
#       v850 (meridional wind)
#       t850 (temperature)
# - The workflow assumes files span 1979–2024.
# - Seasonal standardization is performed independently for:
#       DJF, MAM, JJA, and SON.
#
# Required libraries:
# xarray
# numpy
# tqdm
# scipy
#
# Tested with:
# Python 3.12+ using Spyde IDE
#
# ==========================================================================================================

# ==========================================================================================================
# Import libraries
# ==========================================================================================================

import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator

# ==========================================================================================================
# Physical constants
# ==========================================================================================================

Rd = 287.05       # Gas constant for dry air [J kg⁻¹ K⁻¹]
g = 9.80665       # Gravitational acceleration [m s⁻²]
p = 85000         # Pressure level [Pa] corresponding to 850 hPa

# ==========================================================================================================
# User-defined directories
# ==========================================================================================================

input_dir = r"PATH_TO_HOURLY_ERA5_FILES"
temp_dir = r"PATH_TO_TEMPORARY_DAILY_FILES"

os.makedirs(temp_dir, exist_ok=True)

output_file = os.path.join(
    temp_dir,
    "VT_ERA5_daily_1979_2024.nc"
)

# ==========================================================================================================
# Step 1 — Compute daily vapor transport from hourly ERA5 data
# ==========================================================================================================

# Total number of months from January 1979 to December 2024
n_months = 552

print("\nStarting vapor transport computation...\n")

for i in tqdm(
    range(n_months),
    desc="Monthly processing",
    ncols=100
):

    daily_path = os.path.join(
        temp_dir,
        f"daily_{i:04d}.nc"
    )

    # Skip already processed files
    if os.path.exists(daily_path):

        print(f"Skipping month index {i:04d} (already processed).")
        continue

    year = 1979 + (i // 12)
    month = (i % 12) + 1

    # ----------------------------------------------------------------------------------------------
    # File paths
    # ----------------------------------------------------------------------------------------------

    q_file = os.path.join(
        input_dir,
        f"ERA5_data_q850_{year}_{month:02d}.nc"
    )

    u_file = os.path.join(
        input_dir,
        f"ERA5_data_u850_{year}_{month:02d}.nc"
    )

    v_file = os.path.join(
        input_dir,
        f"ERA5_data_v850_{year}_{month:02d}.nc"
    )

    t_file = os.path.join(
        input_dir,
        f"ERA5_data_t850_{year}_{month:02d}.nc"
    )

    try:

        # ------------------------------------------------------------------------------------------
        # Open datasets
        # ------------------------------------------------------------------------------------------

        q = xr.open_dataset(
            q_file,
            chunks={"valid_time": 24}
        )["q"].rename({"valid_time": "time"})

        u = xr.open_dataset(
            u_file,
            chunks={"valid_time": 24}
        )["u"].rename({"valid_time": "time"})

        v = xr.open_dataset(
            v_file,
            chunks={"valid_time": 24}
        )["v"].rename({"valid_time": "time"})

        t = xr.open_dataset(
            t_file,
            chunks={"valid_time": 24}
        )["t"].rename({"valid_time": "time"})

        # ------------------------------------------------------------------------------------------
        # Compute vapor transport
        # ------------------------------------------------------------------------------------------

        # Virtual temperature [K]
        Tv = t * (1 + 0.61 * q)

        # Air density [kg m⁻³]
        rho = p / (Rd * Tv)

        # Wind speed magnitude [m s⁻¹]
        wind_speed = np.sqrt(u**2 + v**2)

        # Instantaneous vapor transport [kg m⁻² s⁻¹]
        vt_inst = rho * q * wind_speed

        # Daily mean VT
        vt_daily = vt_inst.resample(time="1D").mean(dim="time")

        vt_daily.name = "VT"

        vt_daily.attrs["units"] = "kg m-2 s-1"
        vt_daily.attrs["description"] = (
            "Daily vapor transport at 850 hPa"
        )

        # ------------------------------------------------------------------------------------------
        # Save temporary daily file
        # ------------------------------------------------------------------------------------------

        vt_daily.to_netcdf(daily_path)

        print(f"Saved: {daily_path}")

    except FileNotFoundError as error:

        print(f"Missing file for {year}-{month:02d}: {error}")
        continue

# ----------------------------------------------------------------------------------------------------------
# Merge all temporary daily files
# ----------------------------------------------------------------------------------------------------------

print("\nMerging daily files...\n")

daily_files = sorted([

    os.path.join(temp_dir, file)

    for file in os.listdir(temp_dir)

    if file.startswith("daily_") and file.endswith(".nc")

])

combined = xr.open_mfdataset(
    daily_files,
    combine="by_coords"
)

combined.to_netcdf(output_file)

print(f"Final VT dataset saved to:\n{output_file}")

# ==========================================================================================================
# Step 2 — OPTIONAL: Filter dataset to a selected spatial domain
# ==========================================================================================================

# This step may be skipped if the full ERA5 domain is required.

input_path = r"PATH_TO_DAILY_VT_DATASET.nc"

ds = xr.open_dataset(input_path)

print("Variables in dataset:")
print(ds.data_vars)

VT = ds["VT"]

# ----------------------------------------------------------------------------------------------------------
# Example domain:
#     Latitude : 20°S–90°S
#     Longitude: 180°W–0°
# ----------------------------------------------------------------------------------------------------------

VT_filtered = VT.sel(
    latitude=slice(-20, -90),
    longitude=slice(-180, 0)
)

print("Filtered longitude coordinates:")
print(VT_filtered.coords["longitude"])

output_path = (
    r"PATH_TO_OUTPUT_DIRECTORY\VT_filtered_20S-90S_180W-0_1979_2024.nc"
)

VT_filtered.to_netcdf(output_path)

print(f"Filtered VT dataset saved to:\n{output_path}")

# ----------------------------------------------------------------------------------------------------------
# Optional validation at a grid point
# ----------------------------------------------------------------------------------------------------------

ds_test = xr.open_dataset(output_path)

VT_point = ds_test["VT"].sel(
    latitude=-60.0,
    longitude=-90.0
)

print("Sample values at (60°S, 90°W):")
print(VT_point.values)

# ==========================================================================================================
# Step 3 — Remove singleton pressure-level dimension
# ==========================================================================================================

input_path = (
    r"PATH_TO_OUTPUT_DIRECTORY\VT_filtered_20S-90S_180W-0_1979_2024.nc"
)

ds = xr.open_dataset(input_path)

ds_no_pl = ds.isel(
    pressure_level=0
).drop_vars("pressure_level")

output_path = (
    r"PATH_TO_OUTPUT_DIRECTORY\VT_filtered_no_pressure_level.nc"
)

ds_no_pl.to_netcdf(output_path)

print(
    "Dataset without pressure_level dimension saved to:\n"
    f"{output_path}"
)

# ==========================================================================================================
# Step 4 — Regrid VT data to 1.5° × 1.5°
# ==========================================================================================================

input_path = (
    r"PATH_TO_OUTPUT_DIRECTORY\VT_filtered_no_pressure_level.nc"
)

ds = xr.open_dataset(input_path)

VT = ds["VT"]

lat = ds["latitude"]
lon = ds["longitude"]
time = ds["time"]

# ----------------------------------------------------------------------------------------------------------
# Print dataset information
# ----------------------------------------------------------------------------------------------------------

print("VT shape:", VT.shape)
print("Latitude shape:", lat.shape)
print("Longitude shape:", lon.shape)
print("Time shape:", time.shape)
print("Dimensions:", VT.dims)

# ----------------------------------------------------------------------------------------------------------
# Define source coordinates
# ----------------------------------------------------------------------------------------------------------

lat_source = lat.values
lon_source = lon.values

# ----------------------------------------------------------------------------------------------------------
# Define target grid
# ----------------------------------------------------------------------------------------------------------

lat_target = np.arange(-90, -19.5, 1.5)
lon_target = np.arange(-180, 0.1, 1.5)

lon_target_mesh, lat_target_mesh = np.meshgrid(
    lon_target,
    lat_target
)

target_points = np.array([
    lat_target_mesh.ravel(),
    lon_target_mesh.ravel()
]).T

# ----------------------------------------------------------------------------------------------------------
# Prepare output array
# ----------------------------------------------------------------------------------------------------------

VT_regridded = np.full(
    (
        len(time),
        len(lat_target),
        len(lon_target)
    ),
    np.nan
)

# ----------------------------------------------------------------------------------------------------------
# Interpolate each time slice
# ----------------------------------------------------------------------------------------------------------

for t in range(len(time)):

    VT_slice = VT.isel(time=t).values

    if VT_slice.ndim == 2:

        interp_func = RegularGridInterpolator(
            (lat_source, lon_source),
            VT_slice,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )

        VT_interp = interp_func(target_points).reshape(
            len(lat_target),
            len(lon_target)
        )

        VT_regridded[t, :, :] = VT_interp

    else:

        print(
            f"Skipping time index {t} "
            f"(unexpected shape: {VT_slice.shape})"
        )

# ----------------------------------------------------------------------------------------------------------
# Create output dataset
# ----------------------------------------------------------------------------------------------------------

VT_regridded_ds = xr.Dataset(

    {
        "VT": (
            ["time", "latitude", "longitude"],
            VT_regridded
        )
    },

    coords={

        "time": time,
        "latitude": lat_target,
        "longitude": lon_target
    }
)

output_path = (
    r"PATH_TO_OUTPUT_DIRECTORY\VT_regridded_1.5deg_1979_2024.nc"
)

VT_regridded_ds.to_netcdf(output_path)

print(f"Regridded VT dataset saved to:\n{output_path}")


# ==========================================================================================================
# Step 5 — Standardize VT fields using seasonal climatology
# ==========================================================================================================

file_path = (
    r"PATH_TO_OUTPUT_DIRECTORY\VT_regridded_1.5deg_1979_2024.nc"
)

ds = xr.open_dataset(file_path)

VT = ds["VT"]

# ----------------------------------------------------------------------------------------------------------
# Function to assign climatological seasons
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

# ----------------------------------------------------------------------------------------------------------
# Assign seasonal coordinate
# ----------------------------------------------------------------------------------------------------------

months = VT["time.month"].values

seasons = np.vectorize(get_season)(months)

VT = VT.assign_coords(
    season=("time", seasons)
)

# ----------------------------------------------------------------------------------------------------------
# Standardize independently for each season
# ----------------------------------------------------------------------------------------------------------

standardized_seasonal_data = []

for season in ["DJF", "MAM", "JJA", "SON"]:

    VT_season = VT.sel(
        time=VT["season"] == season
    )

    seasonal_mean = VT_season.mean(
        dim="time",
        skipna=True
    )

    seasonal_std = VT_season.std(
        dim="time",
        skipna=True
    )

    standardized = (
        VT_season - seasonal_mean
    ) / seasonal_std

    standardized_seasonal_data.append(
        standardized
    )

# ----------------------------------------------------------------------------------------------------------
# Recombine standardized seasonal datasets
# ----------------------------------------------------------------------------------------------------------

VT_standardized = xr.concat(
    standardized_seasonal_data,
    dim="time"
)

VT_standardized = VT_standardized.sortby("time")

VT_standardized.name = "VT_standardized"

# ----------------------------------------------------------------------------------------------------------
# Save standardized dataset
# ----------------------------------------------------------------------------------------------------------

output_path = (
    r"PATH_TO_OUTPUT_DIRECTORY\VT_standardized_1.5deg_1979_2024_seasonal.nc"
)

VT_standardized.to_netcdf(output_path)

print(
    "Seasonal standardization complete."
)

print(
    f"Standardized dataset saved to:\n{output_path}"
)