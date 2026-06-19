"""
@author: andressa_marcher

@reviewers: 
    # add if applicable !

"""
# ===========================================================================================================
# Hierarchical Clustering and Shared EPE Analysis
#
# Description:
# This script performs hierarchical clustering and shared-event analysis for
# Extreme Precipitation Events (EPEs) identified at Antarctic ice core sites.
#
# The workflow includes:
#
# 1. Reading EPE datasets for multiple Antarctic ice core sites;
# 2. Computing seasonal EPE coincidence matrices between sites;
# 3. Performing hierarchical clustering based on shared EPE dates;
# 4. Objectively deriving seasonal site groups from the clustering results;
# 5. Plotting seasonal dendrograms for all sites;
# 6. Computing the percentage of shared EPE dates within clustering-derived groups;
# 7. Assessing statistical significance using permutation testing;
# 8. Exporting figures and summary statistics.
#
# Methodological notes:
# - Groups are objectively derived from the hierarchical clustering results
#   for each season using a fixed number of clusters.
# - Shared EPE statistics are therefore evaluated using groups obtained
#   directly from the clustering analysis itself.
#
# Additional notes:
# - EPE datasets must contain a "date" column.
# - Dates are assumed to correspond to previously identified EPE days.
# - DJF seasons are treated climatologically (December–February).
#
# Required libraries:
# pandas
# numpy
# matplotlib
# scipy
#
# Tested with:
# Python 3.12+ using Spyder IDE
#
# ===========================================================================================================

# ===========================================================================================================
# Import libraries
# ===========================================================================================================

import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ===========================================================================================================
# User inputs
# ===========================================================================================================

# Directory containing EPE CSV files
data_dir = r"PATH_TO_EPE_FILES"

# Directory for output figures and tables
output_dir = r"PATH_TO_OUTPUT_DIRECTORY"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------------------
# Site labels
# -----------------------------------------------------------------------------------------------------------

site_labels = [
    "SITE_01", "SITE_02", "SITE_03", "SITE_04", "SITE_05",
    "SITE_06", "SITE_07", "SITE_08", "SITE_09", "SITE_10",
    "SITE_11", "SITE_12", "SITE_13", "SITE_14", "SITE_15",
    "SITE_16", "SITE_17", "SITE_18", "SITE_19", "SITE_20",
    "SITE_21"
]

# -----------------------------------------------------------------------------------------------------------
# Seasonal definitions
# -----------------------------------------------------------------------------------------------------------

season_definitions = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11]
}

# -----------------------------------------------------------------------------------------------------------
# Number of clusters
#
# Seasonal groups are derived automatically from the clustering results.
# -----------------------------------------------------------------------------------------------------------

num_clusters = 5

# -----------------------------------------------------------------------------------------------------------
# Seasonal color palettes for dendrogram clusters
# -----------------------------------------------------------------------------------------------------------

season_colors = {
    "DJF": ["grey", "salmon", "deepskyblue", "goldenrod", "teal"],
    "MAM": ["salmon", "deepskyblue", "grey", "goldenrod", "teal"],
    "JJA": ["salmon", "deepskyblue", "grey", "goldenrod", "teal"],
    "SON": ["deepskyblue", "grey", "salmon", "goldenrod", "teal"]
}

# ===========================================================================================================
# Load EPE datasets
# ===========================================================================================================

epe_data = {}

for label in site_labels:

    input_file = os.path.join(
        data_dir,
        f"EPE_above_p90_{label}_1979_2024.csv"
    )

    try:

        df = pd.read_csv(
            input_file,
            parse_dates=["date"]
        )

        df["month"] = df["date"].dt.month
        df["date"] = df["date"].dt.date

        epe_data[label] = df

        print(f"Loaded: {label}")

    except Exception as error:

        print(f"Failed to read {label}: {error}")

# ===========================================================================================================
# Hierarchical clustering analysis
# ===========================================================================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(10, 8)
)

axes = axes.flatten()

season_order = ["DJF", "MAM", "JJA", "SON"]

# Dictionary to store clustering-derived groups
group_definitions = {}

for index, season in enumerate(season_order):

    months = season_definitions[season]
    custom_colors = season_colors[season]

    # -------------------------------------------------------------------------------------------------------
    # Compute EPE coincidence matrix
    # -------------------------------------------------------------------------------------------------------

    match_matrix = pd.DataFrame(
        index=site_labels,
        columns=site_labels,
        dtype=int
    )

    for site1 in site_labels:

        for site2 in site_labels:

            try:

                dates1 = set(
                    epe_data[site1]
                    .loc[epe_data[site1]["month"].isin(months), "date"]
                )

                dates2 = set(
                    epe_data[site2]
                    .loc[epe_data[site2]["month"].isin(months), "date"]
                )

                match_matrix.loc[site1, site2] = len(dates1 & dates2)

            except Exception:

                match_matrix.loc[site1, site2] = 0

    # -------------------------------------------------------------------------------------------------------
    # Convert similarity matrix to distance matrix
    # -------------------------------------------------------------------------------------------------------

    max_matches = match_matrix.values.max()

    distance_matrix = max_matches - match_matrix.astype(float)

    condensed_distance = squareform(
        distance_matrix,
        checks=False
    )

    # -------------------------------------------------------------------------------------------------------
    # Hierarchical clustering
    # -------------------------------------------------------------------------------------------------------

    linkage_matrix = linkage(
        condensed_distance,
        method="ward"
    )

    cluster_assignments = fcluster(
        linkage_matrix,
        num_clusters,
        criterion="maxclust"
    )

    # -------------------------------------------------------------------------------------------------------
    # Extract groups automatically from clustering results
    # -------------------------------------------------------------------------------------------------------

    season_groups = {}

    for cluster_id in range(1, num_clusters + 1):

        cluster_sites = [

            site_labels[i]

            for i in range(len(site_labels))
            if cluster_assignments[i] == cluster_id
        ]

        season_groups[f"Group {cluster_id}"] = cluster_sites

    group_definitions[season] = season_groups

    # -------------------------------------------------------------------------------------------------------
    # Print clustering-derived groups
    # -------------------------------------------------------------------------------------------------------

    print(f"\n{season} groups derived from clustering:")

    for group_name, group_sites in season_groups.items():

        print(f"{group_name}: {group_sites}")

    # -------------------------------------------------------------------------------------------------------
    # Plot dendrogram
    # -------------------------------------------------------------------------------------------------------

    ax = axes[index]

    dendro = dendrogram(
        linkage_matrix,
        labels=site_labels,
        leaf_rotation=90,
        leaf_font_size=9,
        ax=ax,
        color_threshold=0,
        above_threshold_color="gray"
    )

    icoord = np.array(dendro["icoord"])
    dcoord = np.array(dendro["dcoord"])
    leaves = dendro["leaves"]

    label_to_cluster = dict(
        zip(site_labels, cluster_assignments)
    )

    leaf_colors = {
        label: custom_colors[label_to_cluster[label] - 1]
        for label in site_labels
    }

    # -------------------------------------------------------------------------------------------------------
    # Redraw branches with cluster colors
    # -------------------------------------------------------------------------------------------------------

    for xs, ys in zip(icoord, dcoord):

        try:

            i1 = int((xs[1] - 5) / 10)
            i2 = int((xs[2] - 5) / 10)

            if (
                i1 < 0 or
                i2 < 0 or
                i1 >= len(leaves) or
                i2 >= len(leaves)
            ):

                ax.plot(xs, ys, color="gray", lw=2)
                continue

            label1 = site_labels[leaves[i1]]
            label2 = site_labels[leaves[i2]]

            if label_to_cluster[label1] == label_to_cluster[label2]:

                branch_color = leaf_colors[label1]

            else:

                branch_color = "gray"

            ax.plot(xs, ys, color=branch_color, lw=2)

        except Exception:

            ax.plot(xs, ys, color="gray", lw=2)

    ax.set_title(season, fontsize=13)
    ax.set_ylabel("Dissimilarity")
    ax.grid(False)

# -----------------------------------------------------------------------------------------------------------
# Final figure formatting
# -----------------------------------------------------------------------------------------------------------

fig.suptitle(
    "Hierarchical Clustering of Antarctic Sites Based on Shared EPE Dates",
    fontsize=16
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
dendrogram_output = os.path.join(
    output_dir,
    "seasonal_dendrograms.png"
)

fig.savefig(
    dendrogram_output,
    dpi=500,
    bbox_inches="tight"
)

plt.show()

print(f"\nDendrogram figure saved: {dendrogram_output}")

# ===========================================================================================================
# Shared EPE percentage analysis
# ===========================================================================================================

def compute_shared_percentage(group_sites, season):

    """
    Compute the percentage of shared EPE dates among sites.

    Shared dates are defined as dates occurring in at least two sites.
    """

    months = season_definitions[season]

    date_sets = [

        set(
            epe_data[site]
            .loc[epe_data[site]["month"].isin(months), "date"]
        )

        for site in group_sites
    ]

    union_dates = set().union(*date_sets)

    if not union_dates:
        return 0.0

    counter = Counter()

    for dataset in date_sets:

        for date in dataset:

            counter[date] += 1

    shared_dates = [

        date for date, count in counter.items()
        if count >= 2
    ]

    return 100.0 * len(shared_dates) / len(union_dates)

# ===========================================================================================================
# Permutation test
# ===========================================================================================================

def permutation_test(group_sites, season, n_permutations=1000):

    """
    Perform a permutation test for shared EPE occurrence.
    """

    observed_percentage = compute_shared_percentage(
        group_sites,
        season
    )

    months = season_definitions[season]

    all_dates = []

    for site in site_labels:

        all_dates.extend(

            list(
                epe_data[site]
                .loc[epe_data[site]["month"].isin(months), "date"]
            )
        )

    all_dates = np.array(all_dates)

    dates_per_site = [

        len(
            epe_data[site]
            .loc[epe_data[site]["month"].isin(months), "date"]
        )

        for site in group_sites
    ]

    null_distribution = []

    for _ in range(n_permutations):

        permuted_sets = []

        for n_dates in dates_per_site:

            sampled_dates = set(

                np.random.choice(
                    all_dates,
                    size=n_dates,
                    replace=False
                )
            )

            permuted_sets.append(sampled_dates)

        counter = Counter()

        for dataset in permuted_sets:

            for date in dataset:

                counter[date] += 1

        shared_dates = [

            date for date, count in counter.items()
            if count >= 2
        ]

        percentage = (
            100.0 *
            len(shared_dates) /
            len(set().union(*permuted_sets))
        )

        null_distribution.append(percentage)

    p_value = (
        np.sum(np.array(null_distribution) >= observed_percentage) + 1
    ) / (n_permutations + 1)

    return observed_percentage, p_value

# ===========================================================================================================
# Compute permutation statistics for clustering-derived groups
# ===========================================================================================================

results = []

for season in season_order:

    groups = group_definitions[season]

    for group_name, group_sites in groups.items():

        observed_percentage, p_value = permutation_test(
            group_sites,
            season,
            n_permutations=1000
        )

        results.append({

            "Season": season,
            "Group": group_name,
            "Sites": ", ".join(group_sites),
            "Observed Shared Percentage (%)": round(observed_percentage, 2),
            "p-value": round(p_value, 3)

        })

# ===========================================================================================================
# Save results
# ===========================================================================================================

results_df = pd.DataFrame(results)

output_csv = os.path.join(
    output_dir,
    "shared_EPE_permutation_statistics.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)

print(f"\nPermutation-test results saved: {output_csv}")

# ===========================================================================================================