"""
@author: andressa_marcher

@reviewers: 
    # add if applicable !

"""

# ===========================================================================================================
# Seasonal Frequency Heatmap of Extreme Precipitation Events (EPEs)
#
# Description:
# This script computes and visualizes the seasonal frequency distribution of
# Extreme Precipitation Events (EPEs) for multiple ice core sites or geographic
# locations across predefined multi-year periods.
#
# The script:
#
# 1. Reads daily EPE datasets from text files;
# 2. Assigns each event to a climatological season (DJF, MAM, JJA, SON);
# 3. Correctly adjusts DJF seasons by assigning December to the following year;
# 4. Computes the percentage contribution of each season within selected periods;
# 5. Organizes results into a heatmap matrix;
# 6. Generates and saves a seasonal frequency heatmap figure.
#
# Notes:
# - Percentages are normalized so that seasonal frequencies sum to 100%
#   within each analysis period.
# - The script assumes daily EPE datasets containing a "date" column.
# - The heatmap highlights seasonal variability in EPE occurrence through time.
#
# Input requirements:
# - Daily EPE files must contain:
#     • A column named "date"
#
# Expected input filename format:
# daily_EPE_SITE_NAME.txt
#
# Output generated:
# - EPE_frequency_heatmap.png
#
# Required libraries:
# pandas, numpy, matplotlib, os
#
# Tested with Python 3.12+ using Spyder
#
# ===========================================================================================================

# ===========================================================================================================
# Import libraries
# ===========================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# ===========================================================================================================
# User inputs
# ===========================================================================================================

# Directory containing daily EPE files
data_dir = r"PATH_TO_INPUT_FILES"

# Output directory for figures
output_dir = r"PATH_TO_OUTPUT_DIRECTORY"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------------------
# Input files
#
# Replace the example filenames below with your own datasets.
# -----------------------------------------------------------------------------------------------------------

files = [
    "daily_EPE_SITE_01.txt",
    "daily_EPE_SITE_02.txt",
]

# -----------------------------------------------------------------------------------------------------------
# Seasons
# -----------------------------------------------------------------------------------------------------------

seasons = ["DJF", "MAM", "JJA", "SON"]

# -----------------------------------------------------------------------------------------------------------
# Analysis periods
#
# Modify the periods below as needed.
# Format:
# (start_year, end_year)
# -----------------------------------------------------------------------------------------------------------

analysis_periods = [
    (1979, 1990),
    (1991, 2001),
    (2002, 2012),
    (2013, 2024)
]

# ===========================================================================================================
# Helper functions
# ===========================================================================================================

def get_season(month):
    """
    Assign climatological season based on month.
    """

    if month in [12, 1, 2]:
        return "DJF"

    elif month in [3, 4, 5]:
        return "MAM"

    elif month in [6, 7, 8]:
        return "JJA"

    else:
        return "SON"


def process_file(filepath):
    """
    Read and preprocess EPE dataset.
    """

    df = pd.read_csv(
        filepath,
        sep="\t"
    )

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Extract temporal information
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # ===============================================================================================
    # Seasonal year adjustment
    #
    # December is assigned to the following year to correctly represent DJF seasons.
    # ===============================================================================================

    df["season_year"] = df["year"]

    df.loc[
        df["month"] == 12,
        "season_year"
    ] += 1

    # Assign climatological season
    df["season"] = df["month"].apply(get_season)

    return df

# ===========================================================================================================
# Main processing
# ===========================================================================================================

heatmap_matrix = []

site_labels = []

for file in files:

    filepath = os.path.join(
        data_dir,
        file
    )

    print(f"Processing: {file}")

    # ===============================================================================================
    # Load dataset
    # ===============================================================================================

    df = process_file(filepath)

    row_values = []

    # ===============================================================================================
    # Compute seasonal frequencies for each analysis period
    # ===============================================================================================

    for start_year, end_year in analysis_periods:

        # Select period
        df_period = df[
            (df["season_year"] >= start_year) &
            (df["season_year"] <= end_year)
        ]

        # Seasonal counts
        counts = np.array([
            np.sum(df_period["season"] == season)
            for season in seasons
        ])

        total = counts.sum()

        # Convert counts to percentages
        if total > 0:

            percentages = (
                counts / total
            ) * 100

            # Ensure total equals 100%
            percentages = (
                percentages / percentages.sum()
            ) * 100

        else:

            percentages = np.zeros(4)

        row_values.extend(percentages)

    # Store row
    heatmap_matrix.append(row_values)

    # Generate site label from filename
    label = (
        file
        .replace("daily_EPE_", "")
        .replace(".txt", "")
    )

    site_labels.append(label)

# Convert matrix to NumPy array
heatmap_matrix = np.array(heatmap_matrix)

# ===========================================================================================================
# Colormap
# ===========================================================================================================

colors = [
    "#f1faee",
    "#caf0f8",
    "#90e0ef",
    "#48cae4",
    "#0096c7",
    "#023e8a"
]

custom_cmap = LinearSegmentedColormap.from_list(
    "ocean_colormap",
    colors,
    N=256
)

# ===========================================================================================================
# Generate heatmap
# ===========================================================================================================

fig, ax = plt.subplots(
    figsize=(12, 8)
)

image = ax.imshow(
    heatmap_matrix,
    cmap=custom_cmap,
    aspect="auto",
    vmin=5,
    vmax=45
)

# -----------------------------------------------------------------------------------------------------------
# Y-axis
# -----------------------------------------------------------------------------------------------------------

ax.set_yticks(
    np.arange(len(site_labels))
)

ax.set_yticklabels(
    site_labels,
    fontsize=7
)

# -----------------------------------------------------------------------------------------------------------
# X-axis
# -----------------------------------------------------------------------------------------------------------

x_labels = []

for _ in analysis_periods:
    x_labels.extend(seasons)

ax.set_xticks(
    np.arange(len(x_labels))
)

ax.set_xticklabels(
    x_labels,
    fontsize=8
)

# -----------------------------------------------------------------------------------------------------------
# Add vertical separators between periods
# -----------------------------------------------------------------------------------------------------------

for i in range(1, len(analysis_periods)):

    ax.axvline(
        i * 4 - 0.5,
        color="gray",
        linewidth=1.0
    )

# -----------------------------------------------------------------------------------------------------------
# Add period labels below heatmap
# -----------------------------------------------------------------------------------------------------------

for i, (start_year, end_year) in enumerate(analysis_periods):

    center = i * 4 + 1.5

    ax.text(
        center,
        -0.08,
        f"{start_year}-{end_year}",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.get_xaxis_transform()
    )

# -----------------------------------------------------------------------------------------------------------
# Colorbar
# -----------------------------------------------------------------------------------------------------------

colorbar = plt.colorbar(
    image,
    ax=ax
)

colorbar.set_label(
    "Percentage (%)"
)

colorbar.set_ticks(
    [5, 15, 25, 35, 45]
)

# -----------------------------------------------------------------------------------------------------------
# Figure title
# -----------------------------------------------------------------------------------------------------------

ax.set_title(
    "Seasonal Frequency of Extreme Precipitation Events (%)",
    fontsize=12
)

# -----------------------------------------------------------------------------------------------------------
# Layout adjustments
# -----------------------------------------------------------------------------------------------------------

plt.subplots_adjust(
    bottom=0.15
)

plt.tight_layout()

# ===========================================================================================================
# Save figure
# ===========================================================================================================

output_file = os.path.join(
    output_dir,
    "EPE_frequency_heatmap.png"
)

plt.savefig(
    output_file,
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print("Seasonal EPE frequency heatmap generated successfully.")

# ===========================================================================================================
