"""
@author: andressa_marcher

@reviewers: 
    # add if applicable !

"""

# ===========================================================================================================
# Seasonal and Annual Extreme Precipitation Event (EPE) Trend Analysis
#
# Description:
# This script processes daily Extreme Precipitation Event (EPE) time series from multiple
# ice core sites or geographic locations and computes seasonal and annual standardized
# anomalies (z-scores).
#
# The script:
#
# 1. Reads daily EPE datasets from text files;
# 2. Aggregates EPE values by season and year;
# 3. Computes standardized anomalies (z-scores);
# 4. Performs linear trend analysis using ordinary least squares regression;
# 5. Calculates trend statistics (slope, intercept, R², and p-value);
# 6. Generates seasonal and annual time series plots;
# 7. Saves figures and trend statistics as output files.
#
# Notes:
# - DJF seasons are assigned to the following calendar year.
# - Trend lines are plotted only for statistically significant trends (p < 0.05).
# - Standardization is performed independently for each seasonal and annual series.
#
# Input requirements:
# - Daily EPE files must contain:
#     • A column named "date"
#     • A precipitation column in the second column position
#
# Expected input filename format:
# daily_EPE_SITE_NAME.txt
#
# Output files generated:
# - EPE_group_01.png
# - EPE_group_02.png
# - trend_statistics_all_groups.csv
#
# Required libraries:
# pandas, numpy, matplotlib, scipy, os
#
# Tested with Python 3.12+ using Spyder IDE
#
# ===========================================================================================================

# ===========================================================================================================
# Import libraries
# ===========================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

# ===========================================================================================================
# User inputs
# ===========================================================================================================

# Directory containing daily EPE files
data_dir = r"PATH_TO_INPUT_FILES"

# Output directory
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

# Aggregation method
# Options:
# - "sum"  : cumulative seasonal/annual precipitation
# - "mean" : average seasonal/annual precipitation
aggregation = "sum"

# Seasons to analyze
seasons = ["DJF", "MAM", "JJA", "SON"]

# ===========================================================================================================
# Quality control
# ===========================================================================================================

if aggregation not in ["sum", "mean"]:
    raise ValueError("Aggregation must be either 'sum' or 'mean'.")

# ===========================================================================================================
# Helper functions
# ===========================================================================================================

def get_season(month):
    """
    Assign season based on calendar month.
    """

    if month in [12, 1, 2]:
        return "DJF"

    elif month in [3, 4, 5]:
        return "MAM"

    elif month in [6, 7, 8]:
        return "JJA"

    else:
        return "SON"


def compute_zscore(series):
    """
    Compute standardized anomaly (z-score).
    """

    return (series - series.mean()) / series.std()


def process_file(filepath):
    """
    Read and process a daily EPE dataset.
    """

    df = pd.read_csv(filepath, sep="\t")

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Extract year and month
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Assign DJF to the following year
    df["season_year"] = df["year"]
    df.loc[df["month"] == 12, "season_year"] += 1

    # Assign seasons
    df["season"] = df["month"].apply(get_season)

    # Assume precipitation values are stored in the second column
    value_col = df.columns[1]

    # Aggregate data
    if aggregation == "sum":

        seasonal = (
            df.groupby(["season_year", "season"])[value_col]
            .sum()
        )

        annual = (
            df.groupby("year")[value_col]
            .sum()
        )

    else:

        seasonal = (
            df.groupby(["season_year", "season"])[value_col]
            .mean()
        )

        annual = (
            df.groupby("year")[value_col]
            .mean()
        )

    seasonal = seasonal.unstack()

    return seasonal, annual


def compute_trend(x, y):
    """
    Compute linear trend statistics using least squares regression.
    """

    mask = ~np.isnan(y)

    # Require at least 3 valid values
    if np.sum(mask) < 3:
        return None

    result = linregress(x[mask], y[mask])

    slope = result.slope
    intercept = result.intercept
    r2 = result.rvalue**2
    p_value = result.pvalue

    y_fit = slope * x + intercept

    return slope, intercept, r2, p_value, y_fit

# ===========================================================================================================
# Process datasets
# ===========================================================================================================

data = []
labels = []

for file in files:

    filepath = os.path.join(data_dir, file)

    seasonal, annual = process_file(filepath)

    # Standardize data
    seasonal_std = seasonal.apply(compute_zscore)
    annual_std = compute_zscore(annual)

    data.append((seasonal_std, annual_std))

    # Generate label from filename
    label = (
        file
        .replace("daily_EPE_", "")
        .replace(".txt", "")
    )

    labels.append(label)

# ===========================================================================================================
# Plotting function
# ===========================================================================================================

def plot_group(indices, output_figure):

    """
    Generate seasonal and annual EPE anomaly plots for a group of sites.
    """

    n = len(indices)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=5,
        figsize=(8.27, 11.69)
    )

    results = []

    for row, i in enumerate(indices):

        df_seasonal, df_annual = data[i]

        years = df_annual.index.values

        year_ticks = np.arange(
            years.min(),
            years.max() + 1,
            5
        )

        # ===============================================================================================
        # Seasonal plots
        # ===============================================================================================

        for col, season in enumerate(seasons):

            ax = axes[row, col]

            if season in df_seasonal.columns:

                x = df_seasonal.index.values
                y = df_seasonal[season].values

                # Plot standardized anomaly
                ax.plot(
                    x,
                    y,
                    color="#4a4a4a",
                    linewidth=0.8
                )

                ax.axhline(
                    0,
                    color="black",
                    linewidth=0.5,
                    linestyle="--"
                )

                # Compute trend
                trend = compute_trend(x, y)

                if trend is not None:

                    slope, intercept, r2, p_value, y_fit = trend

                    # Save statistics
                    results.append({
                        "site": labels[i],
                        "season": season,
                        "slope": slope,
                        "intercept": intercept,
                        "r2": r2,
                        "p_value": p_value
                    })

                    # Plot only statistically significant trends
                    if p_value < 0.05:

                        ax.plot(
                            x,
                            y_fit,
                            color="black",
                            linewidth=0.8
                        )

            # Y-axis formatting
            if col == 0:

                ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
                ax.set_ylim(-4, 4)

                ax.tick_params(
                    axis="y",
                    labelsize=6
                )

            else:

                ax.set_yticks([])
                ax.set_ylim(-4, 4)

            # X-axis formatting
            ax.set_xticks(year_ticks)

            if row == n - 1:

                ax.set_xticklabels(
                    year_ticks,
                    rotation=90,
                    fontsize=6
                )

            else:

                ax.set_xticklabels([])

            if row == 0:
                ax.set_title(season, fontsize=8)

        # ===============================================================================================
        # Annual plot
        # ===============================================================================================

        ax = axes[row, 4]

        x = df_annual.index.values
        y = df_annual.values

        ax.plot(
            x,
            y,
            color="#4a4a4a",
            linewidth=0.8
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.5,
            linestyle="--"
        )

        trend = compute_trend(x, y)

        if trend is not None:

            slope, intercept, r2, p_value, y_fit = trend

            results.append({
                "site": labels[i],
                "season": "ANN",
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "p_value": p_value
            })

            if p_value < 0.05:

                ax.plot(
                    x,
                    y_fit,
                    color="black",
                    linewidth=0.8
                )

        ax.set_yticks([])
        ax.set_ylim(-4, 4)

        ax.set_xticks(year_ticks)

        if row == n - 1:

            ax.set_xticklabels(
                year_ticks,
                rotation=90,
                fontsize=6
            )

        else:

            ax.set_xticklabels([])

        if row == 0:
            ax.set_title("ANN", fontsize=8)

        # Site label
        axes[row, 0].set_ylabel(
            labels[i],
            fontsize=6,
            rotation=0,
            labelpad=30,
            va="center"
        )

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(
        output_dir,
        output_figure
    )

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()

    return results

# ===========================================================================================================
# Run analysis
# ===========================================================================================================

# Example grouping
group_01 = list(range(0, 1))
group_02 = list(range(1, len(files)))

results_all = []

results_all += plot_group(
    group_01,
    "EPE_group_01.png"
)

results_all += plot_group(
    group_02,
    "EPE_group_02.png"
)

# ===========================================================================================================
# Save trend statistics
# ===========================================================================================================

df_results = pd.DataFrame(results_all)

csv_output = os.path.join(
    output_dir,
    "trend_statistics_all_groups.csv"
)

df_results.to_csv(csv_output, index=False)

print("Trend analysis completed successfully.")

# ===========================================================================================================

