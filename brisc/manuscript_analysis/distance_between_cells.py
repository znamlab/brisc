import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map

from .barcodes_in_cells import find_singleton_bcs

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
        processed_path.parent / "analysis" / "cell_barcode_df.pkl"
    )
    rabies_cell_properties = rabies_cell_properties[
        rabies_cell_properties["all_barcodes"].notnull()
    ]
    # rename is_starter column to starter
    rabies_cell_properties["slice_number"] = rabies_cell_properties[
        "roi"
    ] + rabies_cell_properties["chamber"].map(chamber_map)

    # Filter to only starter cells
    starter_df = rabies_cell_properties.query("is_starter == True").copy()
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

    # Add vertical dashed lines for the medians of the distributions
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


def determine_presynaptic_distances(cell_barcode_df):
    """
    Determine the distances between starter and presynaptic cells, based on
    sharing at least one unique barcode.

    Args:
        cell_barcode_df (pd.DataFrame): DataFrame with cell properties,
            including 'starter' (boolean) and 'unique_barcodes' (set) for each cell.

    Returns:
        relative_presyn_coords (numpy.ndarray): 2D array (N x 3) with x, y, z
            coordinates of presynaptic cells (relative to their starter).
            Only includes cells where distance > 5 after the norm.
        distances (numpy.ndarray): 1D array (N,) of distances between starter
            and presynaptic cells (in micrometers).
    """
    # Filter only the starter cells
    valid = cell_barcode_df.query("is_starter == True").copy()
    # Prepare columns to store the number of presynaptic cells and their coordinates
    valid["n_presynaptic"] = 0
    valid[
        "presynaptic_coords"
    ] = None  # stores list of relative coords for each starter
    valid["non_rel_coords"] = None  # stores list of absolute coords for each starter

    # Loop through each starter cell
    for stid, starter_row in valid.iterrows():
        # Get the set of unique barcodes for this starter
        starter_bcs = starter_row["unique_barcodes"]

        # Find all cells that are NOT starters but share at least one of these barcodes
        # (i.e., their unique_barcodes set intersects with starter_bcs is non-empty)
        presy = cell_barcode_df[
            (cell_barcode_df["is_starter"] == False)
            & (
                cell_barcode_df["unique_barcodes"].apply(
                    lambda x: len(x.intersection(starter_bcs)) > 0
                )
            )
        ]

        # Count the number of presynaptic cells and store in valid DataFrame
        valid.loc[stid, "n_presynaptic"] = len(presy)

        # Compute coordinates
        start_coords = starter_row[["ara_x", "ara_y", "ara_z"]].values
        presy_coords = presy[["ara_x", "ara_y", "ara_z"]].values
        relative_coords = presy_coords - start_coords

        # Store these coordinate arrays in valid DataFrame
        valid.at[stid, "presynaptic_coords"] = [relative_coords]
        valid.at[stid, "non_rel_coords"] = [presy_coords]

    # Gather all relative coords from those starters that actually have presynaptic cells
    pres = valid[valid["n_presynaptic"] > 0]
    # np.vstack nicely stacks a list of 2D arrays into one big 2D array
    all_coords = np.hstack(pres["presynaptic_coords"].values)[0]
    all_coords = np.array(all_coords, dtype=np.float64)

    # Convert from mm to µm if your coordinates are in mm
    distances = np.linalg.norm(all_coords, axis=1) * 1000.0

    # If you only want to keep those coords whose distance > 5
    # (e.g., to remove the first few microns for some reason),
    # we take a negative sign in front as in your original code:
    relative_presyn_coords = -all_coords[distances > 5]

    return relative_presyn_coords, distances


def load_presynaptic_distances(
    processed_path,
):
    """
    Load unique starter cells and presynaptic cells that share barcodes with starter cells.

    Args:
        processed_path (str): Path to the processed data folder.

    Returns:
        rabies_cell_properties (pd.DataFrame): DataFrame with starter and presynaptic cell
    """
    ara_is_starters = pd.read_pickle(
        processed_path / "analysis" / "cell_barcode_df.pkl"
    )
    ara_is_starters = ara_is_starters[ara_is_starters["all_barcodes"].notna()]

    ara_is_starters = find_singleton_bcs(ara_is_starters)

    # Flatten all barcodes from starter cells to count their occurrences
    starter_barcodes_counts = (
        ara_is_starters[ara_is_starters["is_starter"] == True]["all_barcodes"]
        .explode()
        .value_counts()
    )
    # Identify barcodes that are unique to a single starter cell
    unique_starter_barcodes = starter_barcodes_counts[
        starter_barcodes_counts == 1
    ].index
    # Filter starter cells where all their barcodes are unique among starter cells
    starter_cells_with_unique_barcodes = ara_is_starters[
        (ara_is_starters["is_starter"] == True)
        & (
            ara_is_starters["all_barcodes"].apply(
                lambda barcodes: all(b in unique_starter_barcodes for b in barcodes)
            )
        )
    ]
    # Get all barcodes from the identified starter cells
    unique_barcodes_from_starters = (
        starter_cells_with_unique_barcodes["all_barcodes"].explode().unique()
    )
    # Filter presynaptic cells that contain at least one of these barcodes
    presynaptic_cells_with_shared_barcodes = ara_is_starters[
        (ara_is_starters["is_starter"] == False)
        & (
            ara_is_starters["all_barcodes"].apply(
                lambda barcodes: any(
                    b in unique_barcodes_from_starters for b in barcodes
                )
            )
        )
    ]
    # Combine the filtered starter cells and the filtered presynaptic cells
    ara_starters = pd.concat(
        [starter_cells_with_unique_barcodes, presynaptic_cells_with_shared_barcodes]
    )
    rabies_cell_properties = ara_starters.rename(columns={"is_starter": "starter"})

    return rabies_cell_properties


def plot_AP_ML_relative_coords(
    relative_presyn_coords,
    ax=None,
    lims={"x": (-1000, 1000), "z": (-3000, 3000)},
    s=1,
    alpha=0.1,
    color="black",
    label_fontsize=12,
    tick_fontsize=10,
):
    """
    Plot the distribution of presynaptic cells relative to starter cells in the
    anterior-posterior and medial-lateral plane

    Args:
        relative_presyn_coords (np.array): 2D array with x, y, z coordinates of presynaptic cells
        ax (matplotlib.axes.Axes): Axes to plot on.
        lims (dict): Limits for the x and z axes.
        s (int): Marker size.
        alpha (float): Alpha value for the markers.
        color (str): Color of the markers.
        label_fontsize (int): Font size for labels.
        tick_fontsize (int): Font size for ticks
    """
    labels = dict(z="Medio-lateral (μm)", x="Antero-posterior (μm)")
    ax.scatter(
        relative_presyn_coords[:, 2] * 1000,  # z
        relative_presyn_coords[:, 0] * 1000,  # x
        s=s,
        alpha=alpha,
        color=color,
        edgecolors="none",
    )
    ax.set_xlabel(
        labels["z"],
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        labels["x"],
        fontsize=label_fontsize,
    )
    ax.set_xlim(lims["z"])
    ax.set_ylim(lims["x"])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)


def shuffle_iteration(seed, starters_df, non_starters_df):
    """
    Performs a single iteration of barcode shuffling using the 'unique_barcodes' sets.

    Args:
        seed (int): Seed for random number generation.
        starters_df (pd.DataFrame): DataFrame with starter cell properties, including 'unique_barcodes'.
        non_starters_df (pd.DataFrame): DataFrame with non-starter cell properties, including 'unique_barcodes'.

    Returns:
        iteration_distances (np.array): 2D array (N x 3) with relative distances (x, y, z)
            of all presynaptic cells from their corresponding shuffled starter.
        starter_coords_shuffle (np.array): 2D array (S x 3) with x, y, z coordinates
            of the shuffled starter cells.
    """
    np.random.seed(seed)

    # 1. Shuffle the sets of unique_barcodes among starters
    shuffled_starters = starters_df.copy()
    shuffled_starters["unique_barcodes"] = np.random.permutation(
        starters_df["unique_barcodes"].values
    )

    # 2. Shuffle the sets of unique_barcodes among non-starters
    shuffled_non_starters = non_starters_df.copy()
    shuffled_non_starters["unique_barcodes"] = np.random.permutation(
        non_starters_df["unique_barcodes"].values
    )

    # 3. Re-combine these two groups into one DataFrame
    shuffled_data = pd.concat(
        [shuffled_starters, shuffled_non_starters], ignore_index=True
    )

    # 4. Identify (shuffled) starter cells for distance calculations
    valid = shuffled_data.query("is_starter == True").copy()
    valid["n_presynaptic"] = 0

    # 5. Count how many presynaptic cells each starter has, based on intersections of barcodes
    for stid, starter_row in valid.iterrows():
        bc_set = starter_row["unique_barcodes"]
        presy = shuffled_data[
            (shuffled_data["is_starter"] == False)
            & (
                shuffled_data["unique_barcodes"].apply(
                    lambda x: len(bc_set.intersection(x)) > 0
                )
            )
        ]
        valid.loc[stid, "n_presynaptic"] = presy.shape[0]

    # 6. Collect the 3D coordinates of the shuffled starter cells
    starter_coords_shuffle = valid[["ara_x", "ara_y", "ara_z"]].values

    # 7. Build an array of relative distances between each starter and its presynaptic cells
    iteration_distances = []

    for stid, starter_row in valid.iterrows():
        bc_set = starter_row["unique_barcodes"]
        presy = shuffled_data[
            (shuffled_data["is_starter"] == False)
            & (
                shuffled_data["unique_barcodes"].apply(
                    lambda x: len(bc_set.intersection(x)) > 0
                )
            )
        ]

        # Calculate relative coordinates
        starter_coords = starter_row[["ara_x", "ara_y", "ara_z"]].values
        presy_coords = presy[["ara_x", "ara_y", "ara_z"]].values
        relative_coords = presy_coords - starter_coords

        iteration_distances.extend(relative_coords)

    # 8. Convert to numpy arrays and return
    iteration_distances = np.array(iteration_distances)

    return iteration_distances, starter_coords_shuffle


def shuffle_wrapper(arg):
    seed, starters_df, non_starters_df = arg
    return shuffle_iteration(seed, starters_df, non_starters_df)


def create_barcode_shuffled_nulls_parallel(cell_barcode_df, N_iter=1000):
    """
    Parallelized version of barcode shuffling with live tqdm updates.

    Args:
        rabies_cell_properties (pd.DataFrame): DataFrame with cell properties.
        N_iter (int): Number of iterations to perform.

    Returns:
        all_shuffled_distances (list): List of 2D arrays with distances between
        starter and presynaptic cells for each iteration.
        all_starter_coords (list): List of 2D arrays with x, y, z coordinates of
        starter cells for each iteration.
    """
    starters_df = cell_barcode_df.query("is_starter == True").copy()
    non_starters_df = cell_barcode_df.query("is_starter == False").copy()

    args = [(i, starters_df, non_starters_df) for i in range(N_iter)]
    # Use process_map for parallel processing with tqdm
    results = process_map(shuffle_wrapper, args, max_workers=cpu_count())

    all_shuffled_distances, all_starter_coords = zip(*results)
    all_shuffled_distances = list(all_shuffled_distances)
    all_starter_coords = list(all_starter_coords)

    return all_shuffled_distances, all_starter_coords


def plot_3d_distance_histo(
    distances,
    all_shuffled_distances,
    ax=None,
    label_fontsize=12,
    tick_fontsize=10,
    bins=100,
    max_dist=5,
    linewidth=2,
):
    distances = distances.copy() / 1000
    flat_all_shuffled_distances = np.vstack(all_shuffled_distances).astype(float)
    all_shuffled_3d_distances = np.sqrt(
        np.sum(flat_all_shuffled_distances**2, axis=1)
    )

    # Calculate medians
    median_shuffled = np.median(all_shuffled_3d_distances)
    median_distances = np.median(distances)

    # Histogram for shuffled distances
    ax.hist(
        all_shuffled_3d_distances,
        bins=bins,
        range=(0, max_dist),
        weights=np.ones(len(all_shuffled_3d_distances))
        / len(all_shuffled_3d_distances),
        edgecolor="blue",
        histtype="step",
        linewidth=linewidth,
        label="Shuffled barcode presynaptic distances",
        alpha=0.5,
    )

    # Histogram for actual distances
    ax.hist(
        distances,
        bins=bins,
        range=(0, max_dist),
        weights=np.ones(len(distances)) / len(distances),
        edgecolor="orange",
        histtype="step",
        linewidth=linewidth,
        label="Shared barcode presynaptic distances",
        alpha=0.8,
    )

    # Vertical dashed lines for medians
    ax.axvline(
        median_shuffled,
        color="blue",
        linestyle="--",
        linewidth=linewidth,
        label=f"Shuffled median: {median_shuffled:.2f}",
    )
    ax.axvline(
        median_distances,
        color="orange",
        linestyle="--",
        linewidth=linewidth,
        label=f"Shared median: {median_distances:.2f}",
    )

    # Labels, legend, and formatting
    ax.set_xlabel(
        "Distance (mm)",
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        "Proportion of cell pairs",
        fontsize=label_fontsize,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
    ax.legend(
        fontsize=tick_fontsize,
        loc="upper right",
    )
