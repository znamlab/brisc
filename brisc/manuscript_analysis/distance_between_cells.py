import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map


def add_connection_distances(cells_df, cols=["ara_x", "ara_y", "ara_z"], skipna=False):
    """Calculates and adds Euclidean distances to connected starter cells.

    For each cell in `cells_df`, this function identifies its connected
    starter cells via the `cells_df['starters']` column (which should
    contain a list or set of indices/IDs of starter cells). It then
    computes the Euclidean distance from the cell to each of these
    connected starter cells using the coordinate columns specified by `cols`.

    A new column named 'distances' is added to `cells_df` in-place.
    Each entry in this 'distances' column is a list of floats, where each
    float is the distance to one of the connected starter cells. If a cell
    has no connected starters (i.e., `cell['starters']` is empty), its
    'distances' entry will be an empty list.

    Args:
        cells_df (pd.DataFrame): DataFrame containing cell data.
            Must include:
            - 'is_starter' (bool): True if the cell is a starter cell.
            - 'starters' (list or set): A collection of indices/IDs of
              starter cells connected to the cell in the current row.
              These indices should correspond to the index of `cells_df`.
            - Coordinate columns as specified by the `cols` argument.
        cols (list[str], optional): A list of three column names representing
            the x, y, and z coordinates for distance calculation.
            Defaults to ["ara_x", "ara_y", "ara_z"].
        skipna (bool, optional): How to handle NaN values in coordinates.
            Defaults to False.
    """

    def calculate_distances(cell, starters):
        distances = []
        for starter in cell["starters"]:
            dist_sq = ((cell[cols] - starters.loc[starter, cols]) ** 2).sum(
                skipna=skipna
            )
            distances.append(np.sqrt(dist_sq))
        return distances

    starters_df = cells_df[cells_df["is_starter"] == True]
    cells_df["distances"] = cells_df.apply(
        lambda cell: calculate_distances(cell, starters_df), axis=1
    )


def determine_presynaptic_distances(
    cells_df, col_prefix="ara_", col_suffix="", subtract_z=True
):
    """Determine the distances between starter and presynaptic cells.

    cells_df must contain 'is_starter' and 'unique_barcodes' columns.

    Args:
        cell_barcode_df (pd.DataFrame): DataFrame with cell properties,
            including 'starter' (boolean) and 'unique_barcodes' (set) for each cell.
        col_prefix (str): Prefix for column names. Default: "ara_".
        col_suffix (str): Suffix for column names. Default: "".

    Returns:
        relative_presyn_coords (numpy.ndarray): 2D array (N x 3) with x, y, z
            coordinates of presynaptic cells (relative to their starter).
            Only includes cells where distance > 5 after the norm.
        distances (numpy.ndarray): 1D array (N,) of distances between starter
            and presynaptic cells (in micrometers).
    """
    starters_df = cells_df[cells_df["is_starter"] == True].copy()
    presynaptic_df = cells_df[cells_df["is_starter"] == False]
    starters_df["n_presynaptic"] = 0
    starters_df["presynaptic_coors"] = None
    starters_df["presynaptic_coors_relative"] = None
    xcol = col_prefix + "x" + col_suffix
    ycol = col_prefix + "y" + col_suffix
    zcol = col_prefix + "z" + col_suffix
    for i, starter in starters_df.iterrows():
        matches = presynaptic_df["unique_barcodes"].apply(
            lambda bcs: len(bcs.intersection(starter["unique_barcodes"])) > 0
        )
        presynaptic_matches = presynaptic_df[matches]
        starters_df.loc[i, "n_presynaptic"] = presynaptic_matches.shape[0]
        starter_coors = starter[[xcol, ycol, zcol]].values
        presynaptic_coors = presynaptic_matches[[xcol, ycol, zcol]].values
        if not subtract_z:
            starter_coors[2] = 0
        relative_coors = presynaptic_coors - starter_coors
        starters_df.loc[i, "presynaptic_coors_relative"] = [relative_coors]
        starters_df.loc[i, "presynaptic_coors"] = [presynaptic_coors]

    pres = starters_df[starters_df.n_presynaptic > 0]
    presynaptic_coors_relative = np.hstack(pres["presynaptic_coors_relative"].values)[
        0
    ].astype(float)
    distances = np.linalg.norm(presynaptic_coors_relative, axis=1) * 1000

    return presynaptic_coors_relative, distances, starters_df


def select_unique_barcodes(cells_df):
    """
    Load unique starter cells and presynaptic cells that share barcodes with starter cells.

    Args:
        processed_path (str): Path to the processed data folder.

    Returns:
        rabies_cell_properties (pd.DataFrame): DataFrame with starter and presynaptic cell
    """
    cells_df = cells_df[cells_df["all_barcodes"].notna()].copy()

    def shorten_barcodes(barcodes):
        return [barcode[:10] for barcode in barcodes]

    cells_df["all_barcodes"] = cells_df["all_barcodes"].apply(shorten_barcodes)
    # Flatten all barcodes from starter cells to count their occurrences
    starter_barcodes_counts = (
        cells_df[cells_df["is_starter"] == True]["all_barcodes"]
        .explode()
        .value_counts()
    )
    # Identify barcodes that are unique to a single starter cell
    unique_starter_barcodes = set(
        starter_barcodes_counts[starter_barcodes_counts == 1].index
    )
    # Filter starter cells where all their barcodes are unique among starter cells
    cells_df["unique_barcodes"] = cells_df["all_barcodes"].apply(
        lambda bcs: unique_starter_barcodes.intersection(set(bcs))
    )
    cells_df = cells_df[cells_df["unique_barcodes"].apply(len) > 0]
    return cells_df


def plot_relative_coors(
    relative_presyn_coors,
    ax=None,
    coors_to_plot=(2, 0),
    lims=[(-3000, 3000), (-1000, 1000)],
    s=1,
    alpha=0.1,
    color="black",
    label_fontsize=12,
    tick_fontsize=10,
    labels=["Medio-lateral location (μm)", "Antero-posterior (μm)"],
    rasterized=True,
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
    ax.scatter(
        relative_presyn_coors[:, coors_to_plot[0]],
        relative_presyn_coors[:, coors_to_plot[1]],
        s=s,
        alpha=alpha,
        color=color,
        edgecolors="none",
        rasterized=rasterized,
    )
    ax.set_xlabel(
        labels[0],
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        labels[1],
        fontsize=label_fontsize,
    )
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)


def shuffle_iteration(args):
    """
    Performs a single iteration of barcode shuffling using the 'unique_barcodes' sets.

    Args:
        seed (int): Seed for random number generation.
        starters_df (pd.DataFrame): DataFrame with starter cell properties, including 'unique_barcodes'.
        non_starters_df (pd.DataFrame): DataFrame with non-starter cell properties, including 'unique_barcodes'.

    Returns:
        iteration_distances (np.array): 2D array with distances between starter and presynaptic cells.
        starter_coords_shuffle (np.array): 2D array with x, y, z coordinates of starter cells.

    """
    seed, cells_df, col_prefix, col_suffix = args
    np.random.seed(seed)
    starters_df = cells_df[cells_df["is_starter"] == True]
    # Shuffle barcodes among non-starters
    shuffled_non_starters = cells_df[cells_df["is_starter"] == False].copy()
    shuffled_non_starters["unique_barcodes"] = np.random.permutation(
        shuffled_non_starters["unique_barcodes"].values
    )
    # Re-combine into one DataFrame
    shuffled_data = pd.concat([starters_df, shuffled_non_starters], ignore_index=True)
    return determine_presynaptic_distances(
        shuffled_data, col_prefix=col_prefix, col_suffix=col_suffix
    )


def create_barcode_shuffled_nulls_parallel(
    cells_df, N_iter=1000, col_prefix="ara_", col_suffix=""
):
    """
    Parallelized version of barcode shuffling with live tqdm updates.

    Args:
        rabies_cell_properties (pd.DataFrame): DataFrame with cell properties.
        N_iter (int): Number of iterations to perform.
        col_prefix (str): Prefix for column names. Default: "ara_".
        col_suffix (str): Suffix for column names. Default: "".

    Returns:
        all_shuffled_distances (list): List of 2D arrays with distances between
        starter and presynaptic cells for each iteration.
        all_starter_coords (list): List of 2D arrays with x, y, z coordinates ofd
        starter cells for each iteration.
    """
    args = [(i, cells_df, col_prefix, col_suffix) for i in range(N_iter)]
    # Use process_map for parallel processing with tqdm
    results = process_map(shuffle_iteration, args, max_workers=cpu_count() - 1)

    all_shuffled_distances, all_starter_coords, starters_df = zip(*results)
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
