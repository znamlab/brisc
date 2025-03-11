import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import iss_preprocess as iss


def pairwise_barcode_distances_with_nearest_diff(
    data_path="becalia_rabies_barseq/BRAC8498.3e/chamber_07/",
):
    """
    Compute pairwise distances among starter cells that share at least one barcode,

    Args:
        data_path (str): Path to the data folder.

    Returns:
        distances_same_bc (np.array): Pairwise distances among valid
        starters that share at least one barcode.
        distances_diff_bc (np.array): Pairwise distances from each
        valid starter to all other starters that do not share a barcode.
        nearest_diff_bc (np.array): From each valid starter, the single
        nearest starter that does not share a barcode.
        adjacency_df (pd.DataFrame): Counts for same/diff bc pairs, split
        by adjacent vs non-adjacent sections.
        distances_same_bc_nonadj (np.array): Same_bc distances,
        excluding adjacent pairs.
        distances_diff_bc_nonadj (np.array): Diff_bc distances,
        excluding adjacent pairs.
    """

    chamber_map = {
        "chamber_07": 0,
        "chamber_08": 10,
        "chamber_09": 20,
        "chamber_10": 30,
    }
    processed_path = iss.io.load.get_processed_path(data_path)
    rabies_cell_properties = pd.read_pickle(
        processed_path.parent / "analysis" / "merged_cell_df_curated_mcherry.pkl"
    )
    rabies_cell_properties = rabies_cell_properties[
        rabies_cell_properties["main_barcode"].notnull()
    ]
    # rename is_starter column to starter
    rabies_cell_properties.rename(columns={"is_starter": "starter"}, inplace=True)
    rabies_cell_properties["slice_number"] = rabies_cell_properties[
        "roi"
    ] + rabies_cell_properties["chamber"].map(chamber_map)

    # Filter to only starter cells
    starter_df = rabies_cell_properties.query("starter == True").copy()
    n_starters = len(starter_df)
    if n_starters < 2:
        return (
            np.array([]),  # distances_same_bc
            np.array([]),  # distances_diff_bc
            np.array([]),  # nearest_diff_bc
            pd.DataFrame({"category": [], "count": []}),  # adjacency_df
            np.array([]),  # distances_same_bc_nonadj
            np.array([]),  # distances_diff_bc_nonadj
        )

    # Convert each row's all_barcodes to a set for easy intersection checks
    barcodes_as_sets = starter_df["all_barcodes"].apply(set).values

    # Identify valid starters
    valid_starters_mask = np.zeros(n_starters, dtype=bool)
    for i in range(n_starters):
        for j in range(n_starters):
            if i == j:
                continue
            if len(barcodes_as_sets[i].intersection(barcodes_as_sets[j])) > 0:
                valid_starters_mask[i] = True
                break

    # Extract coordinates and slice_number
    coords = starter_df[["ara_x", "ara_y", "ara_z"]].values
    slice_numbers = starter_df["slice_number"].values

    # Compute pairwise distance matrix
    dist_matrix = cdist(coords, coords, metric="euclidean")

    distances_same_bc = []
    distances_diff_bc = []
    nearest_diff_bc = []
    distances_same_bc_nonadj = []
    distances_diff_bc_nonadj = []

    # For adjacency counts
    count_same_bc_adj = 0
    count_same_bc_nonadj = 0
    count_diff_bc_adj = 0
    count_diff_bc_nonadj = 0

    # Collect "same_bc" and "diff_bc" distributions + adjacency counts
    for i in range(n_starters):
        for j in range(i + 1, n_starters):
            d_ij = dist_matrix[i, j]
            share_bc = len(barcodes_as_sets[i].intersection(barcodes_as_sets[j])) > 0
            adj_slices = abs(slice_numbers[i] - slice_numbers[j]) == 1

            # same_bc
            if valid_starters_mask[i] and valid_starters_mask[j] and share_bc:
                distances_same_bc.append(d_ij)
                if adj_slices:
                    count_same_bc_adj += 1
                else:
                    count_same_bc_nonadj += 1
                    distances_same_bc_nonadj.append(d_ij)

            # diff_bc
            if not share_bc:
                # If i is valid
                if valid_starters_mask[i]:
                    distances_diff_bc.append(d_ij)
                    if adj_slices:
                        count_diff_bc_adj += 1
                    else:
                        count_diff_bc_nonadj += 1
                        distances_diff_bc_nonadj.append(d_ij)
                # If j is valid
                if valid_starters_mask[j]:
                    distances_diff_bc.append(d_ij)
                    if adj_slices:
                        count_diff_bc_adj += 1
                    else:
                        count_diff_bc_nonadj += 1
                        distances_diff_bc_nonadj.append(d_ij)

    # Collect "nearest_diff_bc" (for each valid starter)
    for i in range(n_starters):
        if not valid_starters_mask[i]:
            continue

        # Indices of cells that do NOT share bc with i
        non_overlap_indices = []
        for j in range(n_starters):
            if i == j:
                continue
            if len(barcodes_as_sets[i].intersection(barcodes_as_sets[j])) == 0:
                non_overlap_indices.append(j)

        if len(non_overlap_indices) == 0:
            continue

        # Among these non-overlapping indices, pick the minimum distance
        d_vals = dist_matrix[i, non_overlap_indices]
        nearest_diff_bc.append(d_vals.min())

    # Convert lists to arrays
    distances_same_bc = np.array(distances_same_bc)
    distances_diff_bc = np.array(distances_diff_bc)
    nearest_diff_bc = np.array(nearest_diff_bc)
    distances_same_bc_nonadj = np.array(distances_same_bc_nonadj)
    distances_diff_bc_nonadj = np.array(distances_diff_bc_nonadj)

    adjacency_df = pd.DataFrame(
        {
            "category": [
                "same_bc_adjacent",
                "same_bc_nonadjacent",
                "diff_bc_adjacent",
                "diff_bc_nonadjacent",
            ],
            "count": [
                count_same_bc_adj,
                count_same_bc_nonadj,
                count_diff_bc_adj,
                count_diff_bc_nonadj,
            ],
        }
    )

    return (
        distances_same_bc,
        distances_diff_bc,
        nearest_diff_bc,
        adjacency_df,
        distances_same_bc_nonadj,
        distances_diff_bc_nonadj,
    )


def plot_dist_between_starters(
    distances_diff_bc_nonadj,
    distances_same_bc_nonadj,
    ax=None,
    bins=40,
    max_dist=2,
    label_fontsize=12,
    tick_fontsize=10,
    line_width=0.9,
):
    """
    Plot histograms of distances between starter cells that share at least one barcode,
    and those that do not share any barcodes.

    Args:
        distances_diff_bc_nonadj (np.array): Pairwise distances from each
        valid starter to all other starters that do not share a barcode.
        distances_same_bc_nonadj (np.array): Pairwise distances among valid
        starters that share at least one barcode.
        ax (matplotlib.axes.Axes): Axes to plot on.
        bins (int): Number of bins for the histograms.
        max_dist (float): Maximum distance for the histograms.
        label_fontsize (int): Font size for labels.
        ticks_fontsize (int): Font size for ticks
    """
    # Calculate medians
    median_diff_bc_nonadj = np.median(distances_diff_bc_nonadj)
    median_same_bc_nonadj = np.median(distances_same_bc_nonadj)

    # Plot distances_diff_bc_nonadj
    ax.hist(
        distances_diff_bc_nonadj,
        bins=bins,
        range=(0, max_dist),
        weights=np.ones(len(distances_diff_bc_nonadj)) / len(distances_diff_bc_nonadj),
        alpha=0.8,
        histtype="step",
        linewidth=line_width,
        edgecolor="blue",
        label="Different barcode",
    )

    # Plot distances_same_bc_nonadj
    ax.hist(
        distances_same_bc_nonadj,
        bins=bins,
        range=(0, max_dist),
        weights=np.ones(len(distances_same_bc_nonadj)) / len(distances_same_bc_nonadj),
        alpha=0.8,
        histtype="step",
        linewidth=line_width,
        edgecolor="red",
        label="Same barcode",
    )

    # Add vertical dashed lines for the medians of the main distributions
    ax.axvline(
        median_diff_bc_nonadj,
        color="blue",
        linestyle="--",
        linewidth=line_width,
        label=f"Median (Diff. BC): {median_diff_bc_nonadj:.2f} mm",
    )
    ax.axvline(
        median_same_bc_nonadj,
        color="red",
        linestyle="--",
        linewidth=line_width,
        label=f"Median (Same BC): {median_same_bc_nonadj:.2f} mm",
    )

    # Add labels and legend
    ax.set_xlabel("Distance (mm)", fontsize=label_fontsize)
    ax.set_ylabel("Proportion of cell pairs", fontsize=label_fontsize)
    # move legend to the side
    ax.legend(
        loc="upper right",
        # bbox_to_anchor=(1, 0.5),
        fontsize=tick_fontsize,
    )
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
