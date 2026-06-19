"""
@author: andressa_marcher

@reviewers: 
    # add if applicable !

"""

# ===========================================================================================================
# Group Coherence Analysis of Extreme Precipitation Events (EPEs)
#
# Description:
# This script evaluates the temporal coherence of Extreme Precipitation Event (EPE)
# variability among predefined groups of ice core sites or geographic locations.
#
# The script:
#
# 1. Reads daily EPE datasets from text files;
# 2. Aggregates EPE values into seasonal and annual totals;
# 3. Computes standardized anomalies (z-scores);
# 4. Calculates pairwise correlations between sites within each group;
# 5. Computes mean correlation and mean p-value for each group and season;
# 6. Supports Pearson, Spearman, or Kendall correlation methods;
# 7. Saves group coherence statistics as a CSV file.
#
# Notes:
# - DJF seasons are assigned to the following calendar year.
# - Correlations are computed using pairwise-complete observations.
# - Mean correlations are derived from the upper triangle of the correlation matrix.
#
# Supported correlation methods:
# - "pearson"
# - "spearman"
# - "kendall"
#
# Input requirements:
# - Daily EPE files must contain:
#     • A column named "date"
#     • A precipitation column in the second column position
#
# Expected input filename format:
# daily_EPE_SITE_NAME.txt
#
# Output generated:
# - group_coherence_<method>.csv
#
# Required libraries:
# pandas, numpy, scipy, os
#
# Tested with Python 3.12+ using Spyder IDE
#
# ===========================================================================================================

# ===========================================================================================================
# Import libraries
# ===========================================================================================================

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

# ===========================================================================================================
# User inputs
# ===========================================================================================================

# Directory containing EPE files
data_dir = r"PATH_TO_INPUT_FILES"

# Output directory
output_dir = r"PATH_TO_OUTPUT_DIRECTORY"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------------------
# Correlation method
#
# Options:
# - "pearson"
# - "spearman"
# - "kendall"
# -----------------------------------------------------------------------------------------------------------

correlation_method = "pearson"

# -----------------------------------------------------------------------------------------------------------
# Input files
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
# Site groups
#
# Replace example groups with your own site classifications.
# -----------------------------------------------------------------------------------------------------------

groups = {
    "GROUP_01": ["SITE_01"],
    "GROUP_02": ["SITE_02"]
}

# ===========================================================================================================
# Quality control
# ===========================================================================================================

valid_methods = ["pearson", "spearman", "kendall"]

if correlation_method not in valid_methods:

    raise ValueError(
        f"Invalid correlation method: {correlation_method}. "
        f"Choose from: {valid_methods}"
    )

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

    # Extract temporal information
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Assign DJF to following year
    df["season_year"] = df["year"]
    df.loc[df["month"] == 12, "season_year"] += 1

    # Assign seasons
    df["season"] = df["month"].apply(get_season)

    # Assume precipitation values are stored in the second column
    value_col = df.columns[1]

    # Aggregate precipitation
    seasonal = (
        df.groupby(["season_year", "season"])[value_col]
        .sum()
    )

    annual = (
        df.groupby("year")[value_col]
        .sum()
    )

    seasonal = seasonal.unstack()

    return seasonal, annual


def compute_pairwise_correlation(x, y, method):
    """
    Compute pairwise correlation coefficient and p-value.
    """

    if method == "pearson":
        return pearsonr(x, y)

    elif method == "spearman":
        return spearmanr(x, y)

    elif method == "kendall":
        return kendalltau(x, y)

# ===========================================================================================================
# Compute standardized anomalies
# ===========================================================================================================

seasonal_dict = {}
annual_dict = {}

for file in files:

    filepath = os.path.join(data_dir, file)

    label = (
        file
        .replace("daily_EPE_", "")
        .replace(".txt", "")
    )

    seasonal, annual = process_file(filepath)

    # Standardize data
    seasonal_std = seasonal.apply(compute_zscore)
    annual_std = compute_zscore(annual)

    seasonal_dict[label] = seasonal_std
    annual_dict[label] = annual_std

# ===========================================================================================================
# Group coherence analysis
# ===========================================================================================================

results = []

all_seasons = ["DJF", "MAM", "JJA", "SON", "ANN"]

for group_name, group_labels in groups.items():

    for season in all_seasons:

        series_list = []

        # ===============================================================================================
        # Extract seasonal or annual series
        # ===============================================================================================

        for label in group_labels:

            if season == "ANN":

                ts = annual_dict[label]

            else:

                seasonal_df = seasonal_dict[label]

                if season in seasonal_df.columns:
                    ts = seasonal_df[season]

                else:
                    continue

            series_list.append(ts)

        # Require at least two sites
        if len(series_list) < 2:
            continue

        # Merge series
        df = pd.concat(series_list, axis=1).dropna()

        # Require minimum sample size
        if df.shape[0] < 5:
            continue

        # ===============================================================================================
        # Correlation matrix
        # ===============================================================================================

        corr_matrix = df.corr(method=correlation_method)

        # Extract upper triangle without diagonal
        corr_values = corr_matrix.values[
            np.triu_indices_from(corr_matrix, k=1)
        ]

        mean_correlation = np.nanmean(corr_values)

        # ===============================================================================================
        # Pairwise p-values
        # ===============================================================================================

        p_values = []

        cols = df.columns

        for i in range(len(cols)):

            for j in range(i + 1, len(cols)):

                correlation, p_value = compute_pairwise_correlation(
                    df.iloc[:, i],
                    df.iloc[:, j],
                    correlation_method
                )

                p_values.append(p_value)

        mean_p_value = np.nanmean(p_values)

        # ===============================================================================================
        # Save results
        # ===============================================================================================

        results.append({
            "group": group_name,
            "season": season,
            "correlation_method": correlation_method,
            "mean_correlation": mean_correlation,
            "mean_p_value": mean_p_value,
            "n_years": df.shape[0],
            "n_sites": df.shape[1]
        })

# ===========================================================================================================
# Save results
# ===========================================================================================================

df_output = pd.DataFrame(results)

output_file = os.path.join(
    output_dir,
    f"group_coherence_{correlation_method}.csv"
)

df_output.to_csv(output_file, index=False)

print(f"Group coherence analysis completed successfully using "
      f"{correlation_method} correlation.")

# ===========================================================================================================
