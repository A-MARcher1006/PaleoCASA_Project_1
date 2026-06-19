"""
@author: andressa_marcher

@reviewers: write_here

"""
# ===========================================================================================================
# Seasonal Group Aggregation of Extreme Precipitation Events (EPEs)
#
# Description:
# This script aggregates Extreme Precipitation Event (EPE) datasets from
# predefined groups of Antarctic ice core sites and exports combined seasonal
# datasets for further climatological analysis.
#
# The workflow includes:
#
# 1. Reading EPE datasets for individual sites;
# 2. Filtering EPE dates according to climatological seasons;
# 3. Combining EPE events from grouped sites;
# 4. Removing duplicated dates within each group;
# 5. Exporting consolidated seasonal group datasets as CSV files.
#
# Notes:
# - Input datasets must contain a "date" column.
# - DJF represents austral summer (December–February).
# - Duplicate dates are removed to avoid repeated events within a group.
#
# Required libraries:
# pandas
# os
#
# Tested with:
# Python 3.12+ using Spyder IDE
#
# ===========================================================================================================

# ===========================================================================================================
# Import libraries
# ===========================================================================================================

import os
import pandas as pd

# ===========================================================================================================
# User inputs
# ===========================================================================================================

# Directory containing EPE CSV files
data_dir = r"PATH_TO_EPE_FILES"

# Directory for output CSV files
output_dir = r"PATH_TO_OUTPUT_DIRECTORY"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------------------
# Seasonal definitions
# -----------------------------------------------------------------------------------------------------------

season_months = {

    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11]

}

# -----------------------------------------------------------------------------------------------------------
# Seasonal group definitions
#
# Replace or modify the groups below according to the study design.
# -----------------------------------------------------------------------------------------------------------

group_definitions = {

    "DJF": {

        "Group 1": ["SITE_01", "SITE_09", "SITE_02", "SITE_03", "SITE_04"],
        "Group 2": ["SITE_10", "SITE_15", "SITE_16"],
        "Group 3": ["SITE_07", "SITE_05", "SITE_06"],
        "Group 4": ["SITE_08", "SITE_11", "SITE_14", "SITE_13", "SITE_12"],
        "Group 5": ["SITE_17", "SITE_20", "SITE_19", "SITE_18", "SITE_21"]

    },

    "MAM": {

        "Group 1": ["SITE_01", "SITE_09", "SITE_02"],
        "Group 2": ["SITE_10", "SITE_15", "SITE_16"],
        "Group 3": ["SITE_05", "SITE_06", "SITE_03", "SITE_04"],
        "Group 4": ["SITE_07", "SITE_08", "SITE_11", "SITE_14", "SITE_13", "SITE_12"],
        "Group 5": ["SITE_17", "SITE_20", "SITE_19", "SITE_18", "SITE_21"]

    },

    "JJA": {

        "Group 1": ["SITE_01", "SITE_09", "SITE_02"],
        "Group 2": ["SITE_10", "SITE_15", "SITE_16"],
        "Group 3": ["SITE_05", "SITE_03", "SITE_04"],
        "Group 4": ["SITE_07", "SITE_08", "SITE_11", "SITE_14", "SITE_13", "SITE_12", "SITE_06"],
        "Group 5": ["SITE_17", "SITE_20", "SITE_19", "SITE_18", "SITE_21"]

    },

    "SON": {

        "Group 1": ["SITE_01", "SITE_09", "SITE_02", "SITE_03", "SITE_04"],
        "Group 2": ["SITE_10", "SITE_15", "SITE_16"],
        "Group 3": ["SITE_05", "SITE_06"],
        "Group 4": ["SITE_07", "SITE_08", "SITE_11", "SITE_14", "SITE_13", "SITE_12"],
        "Group 5": ["SITE_17", "SITE_20", "SITE_19", "SITE_18", "SITE_21"]

    }

}
# -----------------------------------------------------------------------------------------------------------
# Season labels for output filenames
# -----------------------------------------------------------------------------------------------------------

season_labels = {

    "DJF": "summer",
    "MAM": "autumn",
    "JJA": "winter",
    "SON": "spring"

}

# ===========================================================================================================
# Process seasonal groups
# ===========================================================================================================

for season, groups in group_definitions.items():

    months = season_months[season]

    print(f"\nProcessing season: {season}")

    for group_name, group_members in groups.items():

        combined_df = pd.DataFrame()

        # ---------------------------------------------------------------------------------------------------
        # Read and concatenate EPE datasets
        # ---------------------------------------------------------------------------------------------------

        for label in group_members:

            input_file = os.path.join(
                data_dir,
                f"EPE_above_p90_{label}_1979_2024.csv"
            )

            if not os.path.exists(input_file):

                print(f"Warning: File not found for {label}")
                continue

            # Read EPE dataset
            df = pd.read_csv(
                input_file,
                parse_dates=["date"]
            )

            # Filter by seasonal months
            df = df[
                df["date"].dt.month.isin(months)
            ]

            # Append to combined dataset
            combined_df = pd.concat(
                [combined_df, df],
                ignore_index=True
            )

        # ---------------------------------------------------------------------------------------------------
        # Remove duplicate dates
        # ---------------------------------------------------------------------------------------------------

        combined_df = (
            combined_df
            .drop_duplicates(subset="date")
            .sort_values(by="date")
        )

        # ---------------------------------------------------------------------------------------------------
        # Generate output filename
        # ---------------------------------------------------------------------------------------------------

        season_label = season_labels[season]

        group_number = group_name.split()[-1]

        output_filename = (
            f"{season_label}_group{group_number}.csv"
        )

        output_file = os.path.join(
            output_dir,
            output_filename
        )

        # ---------------------------------------------------------------------------------------------------
        # Save output CSV
        # ---------------------------------------------------------------------------------------------------

        combined_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"Saved: {output_filename} "
            f"({len(combined_df)} rows)"
        )

# ===========================================================================================================
print("\nSeasonal group aggregation completed successfully.")
# ===========================================================================================================