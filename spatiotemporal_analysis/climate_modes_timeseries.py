"""
@author: andressa_marcher

@reviewers: write_here

"""

# ==================================================================================================
# Seasonal Climate Index Time Series Analysis
# ==================================================================================================
#
# DESCRIPTION
# --------------------------------------------------------------------------------------------------
# This script processes and visualizes seasonal climate-index variability using:
#
#   • Seasonal averaging
#   • Standardization (z-score)
#   • Cubic spline interpolation
#   • Gaussian smoothing
#
# The workflow is designed for climate-index analysis and can be applied to ANY
# monthly climate index time series, including:
#
#   • PSA1
#   • PSA2
#   • SAM
#   • ENSO indices
#   • PDO
#   • AMO
#   • IPO
#   • AAO
#   • SOI
#   • Custom regional indices
#
# The script:
#
#   1. Reads monthly climate-index data;
#   2. Computes seasonal means;
#   3. Combines multiple seasons if desired;
#   4. Standardizes the resulting series;
#   5. Applies spline interpolation;
#   6. Applies Gaussian smoothing;
#   7. Produces high-resolution figures.
#
# INPUT REQUIREMENTS
# --------------------------------------------------------------------------------------------------
# Input file must contain:
#
#   • A date column
#   • A climate-index column
#
# Example structure:
#
#   date         PSA2
#   1979-01-01   -0.34
#   1979-02-01    0.21
#   ...
#
# USER-DEFINED PARAMETERS
# --------------------------------------------------------------------------------------------------
# The user MUST define:
#
#   • Input file path
#   • Output directory
#   • Climate-index column name
#   • Analysis period
#   • Seasons to include
#   • Smoothing parameters
#   • Figure title
#
# OUTPUT
# --------------------------------------------------------------------------------------------------
# High-resolution PNG figure showing:
#
#   • Smoothed climate-index variability
#   • Positive and negative phases
#   • Standardized seasonal anomalies
#
# ==================================================================================================
# Tested with:
# Python 3.12+ using Spyder IDE
#
# ==================================================================================================
# Import Libraries
# ==================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline
import os


# ==================================================================================================
# USER INPUT SECTION
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Input climate-index file
#
# File must contain:
#   • date column
#   • climate-index column
# --------------------------------------------------------------------------------------------------

DATA_FILE = r"PATH_TO_CLIMATE_INDEX_FILE.txt"


# --------------------------------------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------------------------------------

OUTPUT_DIR = r"PATH_TO_OUTPUT_DIRECTORY"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------------------------------
# Climate-index column name
#
# IMPORTANT:
# Replace with the variable name present in the input file.
#
# Examples:
#   "PSA2"
#   "SAM"
#   "PDO"
#   "AMO"
#   "SOI"
# --------------------------------------------------------------------------------------------------

INDEX_COLUMN = ""


# --------------------------------------------------------------------------------------------------
# Analysis period
# --------------------------------------------------------------------------------------------------

START_YEAR = 1979
END_YEAR   = 2024


# --------------------------------------------------------------------------------------------------
# Seasons to include
#
# Examples:
#   ["DJF"]
#   ["SON"]
#   ["DJF", "SON"]
#   ["MAM", "JJA"]
# --------------------------------------------------------------------------------------------------

SEASONS_TO_COMBINE = []


# --------------------------------------------------------------------------------------------------
# Gaussian smoothing parameter
#
# Larger sigma = smoother curve
# --------------------------------------------------------------------------------------------------

GAUSSIAN_SIGMA = 2


# --------------------------------------------------------------------------------------------------
# Figure settings
# --------------------------------------------------------------------------------------------------

FIGURE_WIDTH  = 10
FIGURE_HEIGHT = 2

FIGURE_DPI = 600


# ==================================================================================================
# STEP 1 — LOAD CLIMATE INDEX
# ==================================================================================================

df = pd.read_csv(
    DATA_FILE,
    sep="\t",
    parse_dates=["date"]
)


# --------------------------------------------------------------------------------------------------
# Filter analysis interval
# --------------------------------------------------------------------------------------------------

df = df[
    (df["date"].dt.year >= START_YEAR) &
    (df["date"].dt.year <= END_YEAR)
].reset_index(drop=True)


# ==================================================================================================
# STEP 2 — EXTRACT YEAR AND MONTH
# ==================================================================================================

df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month


# --------------------------------------------------------------------------------------------------
# DJF adjustment
#
# December belongs to the following year.
# Example:
#   Dec 1979 → DJF 1980
# --------------------------------------------------------------------------------------------------

df["year_DJF"] = np.where(
    df["month"] == 12,
    df["year"] + 1,
    df["year"]
)


# ==================================================================================================
# STEP 3 — DEFINE SEASONAL MONTHS
# ==================================================================================================

season_months = {

    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11]

}


# ==================================================================================================
# STEP 4 — FUNCTION TO COMPUTE SEASONAL MEANS
# ==================================================================================================

def seasonal_mean(dataframe, season):

    # ----------------------------------------------------------------------------------------------
    # DJF requires special handling because December belongs to the following year.
    # ----------------------------------------------------------------------------------------------

    if season == "DJF":

        df_season = dataframe[
            dataframe["month"].isin([12, 1, 2])
        ]

        grouped = (
            df_season
            .groupby("year_DJF")[INDEX_COLUMN]
            .mean()
        )

    else:

        months = season_months[season]

        df_season = dataframe[
            dataframe["month"].isin(months)
        ]

        grouped = (
            df_season
            .groupby("year")[INDEX_COLUMN]
            .mean()
        )

    return grouped


# ==================================================================================================
# STEP 5 — COMPUTE SEASONAL SERIES
# ==================================================================================================

years = np.arange(
    START_YEAR,
    END_YEAR + 1
)

seasonal_arrays = []

for season in SEASONS_TO_COMBINE:

    values = (
        seasonal_mean(df, season)
        .reindex(years)
        .values
    )

    seasonal_arrays.append(values)


# --------------------------------------------------------------------------------------------------
# Combine seasonal means
#
# Example:
#   DJF + SON combined variability
# --------------------------------------------------------------------------------------------------

combined_values = np.nanmean(
    seasonal_arrays,
    axis=0
)


# ==================================================================================================
# STEP 6 — STANDARDIZATION (Z-SCORE)
# ==================================================================================================
#
# Standardization allows comparison across indices with different amplitudes.
# ==================================================================================================

combined_values = (
    combined_values - np.nanmean(combined_values)
) / np.nanstd(combined_values)


# ==================================================================================================
# STEP 7 — REMOVE NaNs FOR SPLINE INTERPOLATION
# ==================================================================================================

x = np.arange(len(combined_values))

valid_mask = ~np.isnan(combined_values)

x_valid = x[valid_mask]

values_valid = combined_values[valid_mask]


# ==================================================================================================
# STEP 8 — CUBIC SPLINE INTERPOLATION
# ==================================================================================================
#
# Cubic spline interpolation increases temporal smoothness and visualization quality.
# ==================================================================================================

x_smooth = np.linspace(
    x_valid.min(),
    x_valid.max(),
    1000
)

spline = make_interp_spline(
    x_valid,
    values_valid,
    k=3
)

values_spline = spline(x_smooth)


# ==================================================================================================
# STEP 9 — GAUSSIAN SMOOTHING
# ==================================================================================================
#
# Gaussian filtering highlights low-frequency variability.
# ==================================================================================================

values_smooth = gaussian_filter1d(
    values_spline,
    sigma=GAUSSIAN_SIGMA
)


# --------------------------------------------------------------------------------------------------
# Convert smoothed x-axis back to years
# --------------------------------------------------------------------------------------------------

years_smooth = np.interp(
    x_smooth,
    x_valid,
    years[valid_mask]
)


# ==================================================================================================
# STEP 10 — PLOT TIME SERIES
# ==================================================================================================

plt.figure(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
)


# --------------------------------------------------------------------------------------------------
# Main smoothed curve
# --------------------------------------------------------------------------------------------------

plt.plot(
    years_smooth,
    values_smooth,
    color="black",
    linewidth=2
)


# --------------------------------------------------------------------------------------------------
# Positive phase shading
# --------------------------------------------------------------------------------------------------

plt.fill_between(
    years_smooth,
    0,
    values_smooth,
    where=(values_smooth >= 0),
    facecolor="red",
    alpha=0.4
)


# --------------------------------------------------------------------------------------------------
# Negative phase shading
# --------------------------------------------------------------------------------------------------

plt.fill_between(
    years_smooth,
    0,
    values_smooth,
    where=(values_smooth < 0),
    facecolor="blue",
    alpha=0.4
)


# --------------------------------------------------------------------------------------------------
# Labels and formatting
# --------------------------------------------------------------------------------------------------

plt.title(
    f"Combined Z-score {INDEX_COLUMN} Index "
    f"({' + '.join(SEASONS_TO_COMBINE)})"
)

plt.xlabel("Year")

plt.ylabel("Z-score")

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)


# ==================================================================================================
# STEP 11 — SAVE FIGURE
# ==================================================================================================

output_file = os.path.join(
    OUTPUT_DIR,
    f"{INDEX_COLUMN}_Spline_Gaussian_"
    f"{'_'.join(SEASONS_TO_COMBINE)}.png"
)

plt.savefig(
    output_file,
    dpi=FIGURE_DPI,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True
)

plt.show()

print(f"Figure successfully saved at:\n{output_file}")

# ==================================================================================================
