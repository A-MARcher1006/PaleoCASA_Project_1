"""
@author: andressa_marcher

@reviewers: 
    # if applicable !

"""
# ===========================================================================================================
# ERA5 Daily Precipitation Extraction and Extreme Precipitation Event (EPE) Identification
#
# Description:
# This script extracts daily total precipitation from ERA5 reanalysis data for selected
# ice core sites or geographic locations. The script:
#
# 1. Extracts precipitation at the nearest ERA5 grid point for each site;
# 2. Converts hourly cumulative precipitation from meters to millimeters;
# 3. Aggregates hourly data into daily totals;
# 4. Saves daily precipitation time series as CSV files;
# 5. Creates datasets containing only non-zero precipitation days;
# 6. Identifies Extreme Precipitation Events (EPEs) above the 90th percentile;
# 7. Computes descriptive statistics and percentile summaries for each site.
#
# Notes:
# - ERA5 total precipitation ('tp') is cumulative and provided in meters.
# - This script assumes monthly ERA5 NetCDF files.
# - Longitude coordinates should follow the ERA5 convention (-180 to 180).
#
# Required libraries:
# xarray, pandas, os
#
# Tested with Python 3.12+ using Spyder IDE
#
# ===========================================================================================================

# ===========================================================================================================
# Import libraries
# ===========================================================================================================

import xarray as xr
import pandas as pd
import os

# ===========================================================================================================
# User inputs
# ===========================================================================================================

# Path containing ERA5 NetCDF files
data_dir = r"PATH_TO_ERA5_FILES"

# Output directory for CSV files
output_dir = r"PATH_TO_OUTPUT_DIRECTORY"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Analysis period
start_year = 1979
end_year = 2024

# -----------------------------------------------------------------------------------------------------------
# Site coordinates
#
# Replace the example coordinates and labels below with your own study locations if needed.
# Coordinates must follow the format:
# (latitude, longitude)
# -----------------------------------------------------------------------------------------------------------

site_coordinates = [
    (-84.00000, -79.50000),
    (-80.00000, -85.00000),
]

# Site labels used in output filenames
site_labels = [
    "SITE_01",
    "SITE_02",
]

# ===========================================================================================================
# Generate file list
# ===========================================================================================================

file_list = [
    os.path.join(data_dir, f"ERA5_data_tp_{year}_{month:02d}.nc")
    for year in range(start_year, end_year + 1)
    for month in range(1, 13)
]

# ===========================================================================================================
# Extract daily precipitation for each site
# ===========================================================================================================

daily_dfs = []

for (lat, lon), label in zip(site_coordinates, site_labels):

    print(f"Processing {label} (lat: {lat}, lon: {lon})")

    all_data = []

    for file in file_list:

        # Open ERA5 NetCDF file
        ds = xr.open_dataset(file)

        # Select nearest ERA5 grid point
        point_ds = ds.sel(
            latitude=lat,
            longitude=lon,
            method="nearest"
        )

        # Convert precipitation from meters to millimeters
        tp_mm = point_ds["tp"] * 1000

        # Convert to DataFrame
        df = tp_mm.to_dataframe().reset_index()

        # Standardize time column name
        if "valid_time" in df.columns:
            df = df.rename(columns={"valid_time": "time"})
        elif "time" not in df.columns:
            raise KeyError("Time column not found in dataset.")

        all_data.append(df[["time", "tp"]])

    # Merge all monthly datasets
    df_all = pd.concat(all_data)

    # Convert to datetime format
    df_all["time"] = pd.to_datetime(df_all["time"])

    # Aggregate hourly precipitation into daily totals
    df_all["date"] = df_all["time"].dt.date

    daily_precip = (
        df_all
        .groupby("date")["tp"]
        .sum()
        .reset_index()
    )

    daily_precip.columns = ["date", f"daily_precip_{label}"]

    # Save daily precipitation CSV
    output_file = os.path.join(
        output_dir,
        f"daily_precip_{label}_{start_year}_{end_year}.csv"
    )

    daily_precip.to_csv(output_file, index=False)

    daily_dfs.append(daily_precip)

    print(f"Saved: {output_file}")

# ===========================================================================================================
# Create non-zero precipitation datasets
# ===========================================================================================================

for df, label in zip(daily_dfs, site_labels):

    precip_col = f"daily_precip_{label}"

    # Filter days with precipitation > 0 mm
    non_zero_df = df[df[precip_col] > 0]

    # Save filtered dataset
    output_file = os.path.join(
        output_dir,
        f"non_zero_precip_{label}_{start_year}_{end_year}.csv"
    )

    non_zero_df.to_csv(output_file, index=False)

    print(f"Non-zero precipitation saved: {output_file}")

# ===========================================================================================================
# Identify Extreme Precipitation Events (EPEs)
#
# Definition:
# EPEs are defined here as precipitation values above the 90th percentile
# of non-zero daily precipitation.
# ===========================================================================================================

for label in site_labels:

    input_file = os.path.join(
        output_dir,
        f"non_zero_precip_{label}_{start_year}_{end_year}.csv"
    )

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        continue

    # Load dataset
    df = pd.read_csv(input_file)

    # Precipitation column
    precip_col = df.columns[1]

    precip_values = df[precip_col]

    # Compute 90th percentile threshold
    p90 = precip_values.quantile(0.90)

    # Select EPEs
    epe_df = df[precip_values >= p90]

    # Save EPE dataset
    output_file = os.path.join(
        output_dir,
        f"EPE_above_p90_{label}_{start_year}_{end_year}.csv"
    )

    epe_df.to_csv(output_file, index=False)

    print(f"EPE dataset saved: {output_file}")

# ===========================================================================================================
# Generate precipitation summary statistics
# ===========================================================================================================

summary_data = []

for label in site_labels:

    input_file = os.path.join(
        output_dir,
        f"non_zero_precip_{label}_{start_year}_{end_year}.csv"
    )

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        continue

    df = pd.read_csv(input_file)

    # Assume precipitation values are stored in the second column
    precip_values = df.iloc[:, 1]

    stats = {
        "Site": label,
        "Mean (mm)": precip_values.mean(),
        "Median (mm)": precip_values.median(),
        "Maximum (mm)": precip_values.max(),
        "Minimum (mm)": precip_values.min(),
        "10th Percentile (mm)": precip_values.quantile(0.10),
        "20th Percentile (mm)": precip_values.quantile(0.20),
        "30th Percentile (mm)": precip_values.quantile(0.30),
        "40th Percentile (mm)": precip_values.quantile(0.40),
        "60th Percentile (mm)": precip_values.quantile(0.60),
        "70th Percentile (mm)": precip_values.quantile(0.70),
        "80th Percentile (mm)": precip_values.quantile(0.80),
        "90th Percentile (mm)": precip_values.quantile(0.90),
    }

    summary_data.append(stats)

# Convert summary statistics to DataFrame
summary_df = pd.DataFrame(summary_data)

# Save summary CSV
summary_output = os.path.join(
    output_dir,
    f"EPE_summary_{start_year}_{end_year}.csv"
)

summary_df.to_csv(summary_output, index=False)

print(f"Summary statistics saved: {summary_output}")

# ===========================================================================================================