from functools import partial
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
        starters_df (pd.DataFrame): DataFrame with starter cell properties, including
            'unique_barcodes'.
        non_starters_df (pd.DataFrame): DataFrame with non-starter cell properties,
            including 'unique_barcodes'.

    Returns:
        iteration_distances (np.array): 2D array with distances between starter and
            presynaptic cells.
        starter_coords_shuffle (np.array): 2D array with x, y, z coordinates of starter
            cells.

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


def compute_pairwise_distances(df, return_section_info=False, get_nonadj_id=False):
    """Computes pairwise distances between all rows in a DataFrame.

    This function calculates the Euclidean distance between all pairs of
    points defined by the 'ara_x', 'ara_y', and 'ara_z' columns in the
    input DataFrame. It can optionally also compute the pairwise difference
    in section numbers.

    Args:
        df (pd.DataFrame): DataFrame containing cell data. Must include
            'ara_x', 'ara_y', 'ara_z' columns. If `return_section_info`
            is True, it must also include an 'absolute_section' column.
        return_section_info (bool, optional): If True, the function also
            calculates and returns the pairwise differences in the
            'absolute_section' column. Defaults to False.

    Returns:
        np.ndarray or tuple[np.ndarray, np.ndarray]:
            - If `return_section_info` is False, returns a 1D numpy array
              containing the unique pairwise distances (lower triangle of the
              distance matrix).
            - If `return_section_info` is True, returns a tuple of two 1D
              numpy arrays: (pairwise_distances, section_differences).
    """
    coords = df[["ara_x", "ara_y", "ara_z"]].values.astype(float)
    pairwise = np.linalg.norm(coords[None, :, :] - coords[:, None, :], axis=2)
    pairwise = np.tril(pairwise)
    pairwise = pairwise[pairwise != 0]
    if not return_section_info:
        return pairwise
    sec = df["absolute_section"].values.astype(float)
    delta_sec = sec[None, :] - sec[:, None]
    triu_mask = np.arange(delta_sec.shape[0])[:, None] <= np.arange(delta_sec.shape[1])
    delta_sec[triu_mask] = np.nan
    if get_nonadj_id:
        non_adj = np.logical_and(~np.isnan(delta_sec), np.abs(delta_sec) != 1)
        non_adj = np.where(non_adj.sum(axis=1) > 0)[0]
    delta_sec = delta_sec[~np.isnan(delta_sec)]
    if get_nonadj_id:
        return pairwise, delta_sec, non_adj
    return pairwise, delta_sec


def compute_distances(df1, df2):
    """Computes pairwise Euclidean distances between two sets of points.

    This function calculates the distance between each point in `df1` and every
    point in `df2` based on their 'ara_x', 'ara_y', and 'ara_z' coordinates.

    Args:
        df1 (pd.DataFrame): DataFrame for the first set of points. Must
            include 'ara_x', 'ara_y', and 'ara_z' columns.
        df2 (pd.DataFrame): DataFrame for the second set of points. Must
            also include 'ara_x', 'ara_y', and 'ara_z' columns.

    Returns:
        np.ndarray: A 2D numpy array of shape (len(df2), len(df1)) where
            the element at (i, j) is the distance between the i-th point
            in df2 and the j-th point in df1.
    """
    c1 = df1[["ara_x", "ara_y", "ara_z"]].values.astype(float)
    c2 = df2[["ara_x", "ara_y", "ara_z"]].values.astype(float)
    dst = np.linalg.norm(c1[None, :, :] - c2[:, None, :], axis=2)
    return dst


def distances_between_starters(barcode_list, starter_df, verbose=False):
    """Calculates distances within and between groups of starter cells based on shared
    barcodes.

    This function is designed to be used in a bootstrapping context. For each
    barcode in a given sample (`barcode_list`), it identifies the starter cells
    sharing that barcode. It then computes three sets of distances:
    1. Pairwise distances among the starter cells sharing the barcode.
    2. Pairwise differences in section number for the same group.
    3. Distances from this group of cells to all other starter cells.

    Args:
        barcode_list (iterable): An iterable of barcodes, typically a bootstrap sample.
        starter_df (pd.DataFrame): A DataFrame containing only starter cells.
            Must include 'all_barcodes', 'ara_x', 'ara_y', 'ara_z', and
            'absolute_section' columns.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - dist2same (np.ndarray): 1D array of pairwise distances between
              starter cells sharing the same barcode.
            - sectiondiff (np.ndarray): 1D array of pairwise section
              differences corresponding to `dist2same`.
            - dist2others (np.ndarray): 1D array of distances from starter
              cells with a given barcode to all other starter cells.
    """
    dist2same = []
    sectiondiff = []
    dist2others = []
    if verbose:
        bc_used = [set(), set(), set()]
        cell_used = [set(), set(), set()]
    for bc in barcode_list:
        has_bc = starter_df.all_barcodes.map(lambda x: bc in x)
        cells = starter_df[has_bc]
        other_cells = starter_df[~has_bc]
        p, d, nonadjid = compute_pairwise_distances(
            cells, return_section_info=True, get_nonadj_id=True
        )
        if verbose:
            bc_used[0].add(bc)
            for bc_other in other_cells.all_barcodes:
                bc_used[1].update(bc_other)
            cell_used[0].update(cells.index)
            cell_used[1].update(other_cells.index)
            if len(nonadjid) > 0:
                bc_used[2].add(bc)
            for i in nonadjid:
                cell_used[2].add(cells.index[i])
        d2o = compute_distances(cells, other_cells)
        dist2others.append(d2o.flatten())
        dist2same.append(p)
        sectiondiff.append(d)
    dist2same = np.hstack(dist2same)
    sectiondiff = np.hstack(sectiondiff)
    dist2others = np.hstack(dist2others)
    if verbose:
        print(f"BC used: {[len(b) for b in bc_used]}")
        print(f"Cell used: {[len(b) for b in cell_used]}")
        return dist2same, sectiondiff, dist2others, bc_used, cell_used
    return dist2same, sectiondiff, dist2others


def bootstrap_barocdes_in_multiple_cells(
    starter_df, multi_starter_bcs, n_permutations=1000, random_state=12
):
    """Performs bootstrap analysis of distances between starter cells sharing barcodes.

    This function simulates the distribution of distances between starter cells
    that are labeled with the same barcode. It works by repeatedly sampling
    (with replacement) from a pool of barcodes known to infect multiple starter
    cells (`multi_starter_bcs`). For each bootstrap sample of barcodes, it
    calculates the resulting intra- and inter-group distances in parallel.

    Args:
        starter_df (pd.DataFrame): DataFrame containing all starter cells.
            Must be compatible with `distances_between_starters`.
        multi_starter_bcs (iterable): A collection of barcodes that appear in
            more than one starter cell. This serves as the population for
            bootstrapping.
        n_permutations (int, optional): The number of bootstrap iterations to
            perform. Defaults to 1000.
        random_state (int, optional): Seed for the random number generator to
            ensure reproducibility. Defaults to 12.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
            A tuple of three lists, where each list contains the results from
            the `n_permutations` iterations:
            - dist2same: A list of 1D arrays of pairwise distances between
              starter cells sharing the same barcode.
            - sectiondiff: A list of 1D arrays of pairwise section differences.
            - dist2others: A list of 1D arrays of distances from barcode-sharing
              groups to other starter cells.
    """
    rng = np.random.default_rng(seed=random_state)
    bootsamples = [
        rng.choice(list(multi_starter_bcs), size=len(multi_starter_bcs), replace=True)
        for i in range(n_permutations)
    ]
    func = partial(distances_between_starters, starter_df=starter_df)
    bootstrapped_results = process_map(
        func,
        bootsamples,
        max_workers=round(cpu_count() / 3),
        desc="Hierarchical Bootstrapping",
        total=n_permutations,
        chunksize=10,
    )
    dist2same = [b[0] for b in bootstrapped_results]
    sectiondiff = [b[1] for b in bootstrapped_results]
    dist2others = [b[2] for b in bootstrapped_results]

    return dist2same, sectiondiff, dist2others
