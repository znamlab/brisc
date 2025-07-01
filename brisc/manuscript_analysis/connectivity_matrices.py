import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch  # For drawing arrows


def match_barcodes(cells_df):
    """Matches barcodes between starter cells and all other cells.

    This function identifies which starter cells share 'unique_barcodes' with
    any given cell in the `cells_df`. It modifies `cells_df` in place by
    adding two columns:
    - 'starters': A set of indices of starter cells that share at least one
                  barcode with the cell in that row.
    - 'n_starters': The number of such starter cells (length of the 'starters' set).

    Args:
        cells_df (pd.DataFrame): DataFrame containing cell data. Must include
            'is_starter' (bool) and 'unique_barcodes' (set-like) columns.
            This DataFrame is modified in place.
    """

    def match_barcodes_(series, barcodes):
        return series.apply(lambda bcs: len(bcs.intersection(barcodes)) > 0)

    starters = cells_df[cells_df["is_starter"] == True]
    connectivity_matrix = cells_df["unique_barcodes"].apply(
        lambda bcs: match_barcodes_(starters["unique_barcodes"], bcs)
    )
    cells_df["starters"] = connectivity_matrix.apply(
        lambda row: set(row.index[row]), axis=1
    )
    cells_df["n_starters"] = cells_df["starters"].apply(len)


def compute_input_fractions(starter_row, presyn_cells, presyn_grouping):
    """
    For a single starter cell row, find all presynaptic cells sharing at least one
    barcode, then compute the counts and input fraction of those presynaptic cells
    coming from each presyn group.

    Args:
        starter_row (pd.Series): Row of starter cell data
        presyn_cells (pd.DataFrame): DataFrame of presynaptic cell data
        presyn_grouping (str): Property to group presynaptic cells on

    Returns:
        fraction_by_area (pd.Series): Fraction of presynaptic cells in each area
        counts_by_area (pd.Series): Counts of presynaptic cells in each area
    """
    starter_barcodes = starter_row["unique_barcodes"]

    # Get all possible presynaptic areas (to ensure consistent column ordering later)
    presyn_categories = presyn_cells[presyn_grouping].unique()

    # Find presyn cells that share at least 1 barcode
    # (intersection > 0 means they share at least one)
    shared_presyn = presyn_cells[
        presyn_cells["unique_barcodes"].apply(
            lambda barcodes: len(starter_barcodes & barcodes) > 0
        )
    ]

    if len(shared_presyn) == 0:
        # No presynaptic cells share barcodes for this starter
        # Return zeros for all presyn areas
        fraction_by_area = pd.Series(
            [0.0] * len(presyn_categories), index=presyn_categories
        )
        counts_by_area = pd.Series(
            [0] * len(presyn_categories), index=presyn_categories
        )
    else:
        # Group the shared presyn cells by their area, and compute fraction
        counts_by_area = shared_presyn.groupby(presyn_grouping, observed=True).size()
        counts_by_area = counts_by_area.reindex(presyn_categories, fill_value=0)
        fraction_by_area = counts_by_area / counts_by_area.sum()

    return fraction_by_area, counts_by_area


def compute_connectivity_matrix(
    cell_barcode_df,
    starter_grouping="area_acronym_ancestor_rank1",
    presyn_grouping="area_acronym_ancestor_rank1",
    output_fraction=False,
):
    """
    Compute the connectivity matrix for all starter cells in the given DataFrame.
    Group starters and presyns by any given property present in a df column

    Args:
        cell_barcode_df (pd.DataFrame): DataFrame of barcoded cell data
        starter_grouping (str): Property to group starter cells on
        presyn_grouping (str): Property to group presynaptic cells

    Returns:
        counts_df (pd.DataFrame): DataFrame of counts of presynaptic cells in each area for each starter
        mean_input_frac_df (pd.DataFrame): DataFrame of mean fraction of presynaptic cells in each area for each starter
        fractions_df (pd.DataFrame): DataFrame of fractions of presynaptic cells in each area for each starter

    """
    # Separate starters and presynaptic cells
    starter_cells = cell_barcode_df[cell_barcode_df["is_starter"]].copy()
    presyn_cells = cell_barcode_df[cell_barcode_df["is_starter"] == False].copy()
    # Apply the function to each starter cell to get a table of fractions
    fractions_counts_dfs = starter_cells.apply(
        compute_input_fractions, axis=1, args=(presyn_cells, presyn_grouping)
    )
    fractions_df = pd.DataFrame(
        [t[0] for t in fractions_counts_dfs], index=starter_cells.index
    )
    counts_df = pd.DataFrame(
        [t[1] for t in fractions_counts_dfs], index=starter_cells.index
    )
    # Add a column for a property to group the starter on
    fractions_df[starter_grouping] = starter_cells[starter_grouping].values
    counts_df[starter_grouping] = starter_cells[starter_grouping].values
    # remove rows that sum to 0 (starters with no presynaptic cells)
    fractions_df = fractions_df.loc[
        fractions_df.select_dtypes(include="number").sum(axis=1) > 0
    ]
    # Grouping by starter property, find the mean fraction of each presynaptic grouping
    total_counts_df = counts_df.groupby(starter_grouping, observed=False).sum().T
    if output_fraction:
        # For each presyn area (each row), divide by the row sum so it sums to 1
        mean_frac_df = total_counts_df.div(total_counts_df.sum(axis=1), axis=0)
    else:
        mean_frac_df = fractions_df.groupby(starter_grouping, observed=False).mean().T

    return total_counts_df, mean_frac_df, fractions_df, counts_df


def compute_odds_ratio(
    p_matrix: pd.DataFrame,
    starter_counts: pd.Series,
) -> pd.DataFrame:
    """
    Calculate the odds ratio comparing p_matrix (the fraction of presynaptic outputs)
    to the fraction of all starter cells in each area.

    Parameters
    ----------
    p_matrix : pd.DataFrame
        Rows = presynaptic areas, Columns = starter areas,
        each row sums to 1 (fraction from presyn. area i to each starter area).
    starter_counts : pd.Series
        The total number of starter cells in each area.
        The index should match p_matrix.columns.

    Returns
    -------
    pd.DataFrame
        An odds-ratio matrix of the same shape as p_matrix.
    """
    # Fraction of all starter cells in each area
    fraction_of_starters = starter_counts / starter_counts.sum()

    or_matrix = p_matrix.copy()
    for j in p_matrix.columns:
        q_j = fraction_of_starters[j]

        def odds_ratio(p):
            if p >= 1.0:
                return np.inf
            elif p <= 0.0:
                return 0.0
            elif q_j >= 1.0:
                return 0.0
            elif q_j <= 0.0:
                return np.inf

            return (p * (1 - q_j)) / (q_j * (1 - p))

        or_matrix[j] = p_matrix[j].apply(odds_ratio)

    return or_matrix


def reorganise_matrix(
    matrix,
    areas={
        "VISp1": "1",
        "VISp2/3": "2/3",
        "VISp4": "4",
        "VISp5": "5",
        "VISp6a": "6a",
        "VISp6b": "6b",
    },
):
    """
    Filter the confusion matrix to include only areas of interest and remove rows and columns with all zeros.

    Args:
        matrix (pd.DataFrame): DataFrame of confusion matrix data

    Returns:
        filtered_confusion_matrix (pd.DataFrame): Filtered confusion matrix
    """

    matrix = matrix.reindex(
        index=areas.keys(),
        columns=areas.keys(),
        fill_value=0,
    )

    # Drop rows or columns with only 0s
    matrix = matrix.rename(index=areas, columns=areas)
    return matrix


def plot_area_by_area_connectivity(
    connectivity_matrix,
    starter_counts,
    presynaptic_counts,
    ax,
    cbax=None,
    input_fraction=False,
    odds_ratio=False,
    label_fontsize=12,
    tick_fontsize=12,
    line_width=1,
    show_counts=True,
    cbar_label="Input fraction",
    xlabel="Starter layer",
    ylabel="Presynaptic layer",
    vmin=None,
    vmax=None,
):
    """Plots an area-by-area connectivity matrix as a heatmap.

    This function visualizes a connectivity matrix using `seaborn.heatmap`.
    It can display raw counts, input fractions, or odds ratios, and can
    annotate cells with their values. Optionally, it shows total starter
    cell counts per area.

    Args:
        connectivity_matrix (pd.DataFrame): The matrix to plot. Rows are
            presynaptic areas, columns are starter areas.
        starter_counts (pd.Series or dict): Counts of starter cells in each
            starter area (corresponding to columns of `connectivity_matrix`).
        presynaptic_counts (pd.Series or dict): Counts of presynaptic cells in
            each presynaptic area (not directly used in current plotting logic but
            kept for signature consistency).
        ax (matplotlib.axes.Axes): The main axes to plot the heatmap on.
        cbax (matplotlib.axes.Axes, optional): Axes for the colorbar.
        input_fraction (bool, optional): If True, annotations are formatted as
            fractions (e.g., "0.25"). Otherwise, as integers. Defaults to False.
        odds_ratio (bool, optional): If True, adjusts colormap and normalization
            for odds ratios. Defaults to False.
        label_fontsize (int, optional): Font size for axis labels. Defaults to 12.
        tick_fontsize (int, optional): Font size for tick labels and annotations.
            Defaults to 12.
        line_width (int or float, optional): Line width for cell borders in the
            heatmap. Defaults to 1.
        show_counts (bool, optional): If True, displays starter cell counts below
            the heatmap. Defaults to True.
        cbar_label (str, optional): Label for the colorbar. Defaults to "Input fraction".
    """
    if vmin is None:
        vmin = np.min(connectivity_matrix[connectivity_matrix != -np.inf]) * 0.7
    if vmax is None:
        vmax = -vmin if odds_ratio else connectivity_matrix.max(axis=None)
    cmap = "RdBu_r" if odds_ratio else "inferno"
    # Plot the heatmap
    sns.heatmap(
        connectivity_matrix,
        cmap=cmap,
        cbar=True if cbax else False,
        cbar_ax=cbax,
        yticklabels=True,
        square=True,
        linewidths=line_width,
        linecolor="white",
        annot=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    if cbax:
        cbax.tick_params(labelsize=tick_fontsize)
        cbax.patch.set_edgecolor("black")
        cbax.patch.set_linewidth(1.5)
        cbax.set_title(cbar_label, fontsize=tick_fontsize, loc="left")
        # cbax.set_clim(-1.0, 1.0 if odds_ratio else 0)

    # Annotate with appropriate color based on background
    for (i, j), val in np.ndenumerate(connectivity_matrix):
        if odds_ratio:
            q1 = np.percentile(connectivity_matrix, 15)
            q3 = np.percentile(connectivity_matrix, 85)
            text_color = "white" if val <= q1 or val >= q3 else "black"
        else:
            text_color = (
                "white" if val < connectivity_matrix.max(axis=None) / 2 else "black"
            )
        ax.text(
            j + 0.5,
            i + 0.5,
            f"{val:.2f}" if input_fraction else f"{int(val)}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=tick_fontsize,
        )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            connectivity_matrix.shape[1],
            connectivity_matrix.shape[0],
            fill=False,
            edgecolor="black",
            lw=line_width,
        )
    )
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.tick_params(axis="y", rotation=0)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if not show_counts:
        return
    # Add number of starter cells in each area per column on the bottom of the heatmap
    for i, area in enumerate(connectivity_matrix.columns):
        # if area in starter_counts else put 0
        ax.text(
            i + 0.5,
            connectivity_matrix.shape[0] + 0.5,
            starter_counts.get(area, 0),
            ha="center",
            va="center",
            color="black",
            fontsize=tick_fontsize,
        )

    # add a label saying what the sum is
    ax.text(
        -0.15,
        connectivity_matrix.shape[0] + 0.5,
        "N starters:",
        ha="right",
        va="center",
        color="black",
        fontsize=tick_fontsize,
    )


def make_minimal_df(
    cell_barcode_df,
    starter_filtering_dict=None,
    presyn_filtering_dict=None,
):
    """
    Filter the ARA starters DataFrame to only include the necessary columns and rows.

    Args:
        cell_barcode_df (pd.DataFrame): DataFrame of ARA starters data
        starter_filtering_dict (dict): Dictionary of columns and allowed values to filter starter cells
        presyn_filtering_dict (dict): Dictionary of columns and allowed values to filter presynaptic cells

    Returns:
        minimal_cell_barcode_df (pd.DataFrame): Minimal DataFrame of ARA starters data
    """
    cell_barcode_df = cell_barcode_df[
        [
            "unique_barcodes",
            "area_acronym_ancestor_rank1",
            "Annotated_clusters",
            "is_starter",
            "ara_x",
            "ara_y",
            "ara_z",
            "flatmap_dorsal_x",
            "flatmap_dorsal_y",
            "normalised_depth",
            "cortical_area",
            "cortical_layer",
        ]
    ]

    # Separate starters vs. non-starters
    starter_rows = cell_barcode_df[cell_barcode_df["is_starter"] == True].copy()
    presyn_rows = cell_barcode_df[cell_barcode_df["is_starter"] == False].copy()

    # Filter the starter rows based on starter_filtering_dict
    if starter_filtering_dict:
        for col, allowed_vals in starter_filtering_dict.items():
            # Keep only rows where col is in the list of allowed values
            starter_rows = starter_rows[starter_rows[col].isin(allowed_vals)]

    # Filter the non-starter (presyn) rows based on presyn_filtering_dict
    if presyn_filtering_dict:
        for col, allowed_vals in presyn_filtering_dict.items():
            presyn_rows = presyn_rows[presyn_rows[col].isin(allowed_vals)]

    cell_barcode_df = pd.concat([starter_rows, presyn_rows])

    return cell_barcode_df


def shuffle_barcodes(
    seed,
    cell_barcode_df,
    shuffle_presyn=False,
    shuffle_starters=False,
):
    """
    Perform one permutation of shuffling the barcodes within starter / non-starter cells.

    Args:
        cell_barcode_df (pd.DataFrame): DataFrame of ARA starters data
        shuffle_starters (bool): Whether to shuffle the starter barcodes
        shuffle_presyn (bool): Whether to shuffle the presynaptic barcodes

    Returns:
        shuffled_cell_barcode_df (pd.DataFrame): DataFrame of shuffled ARA starters data

    """
    np.random.seed(seed + 1)
    shuffled_cell_barcode_df = cell_barcode_df.copy()

    # Shuffle barcode sets across starter cells
    if shuffle_starters:
        mask = shuffled_cell_barcode_df["is_starter"] == True
        values = shuffled_cell_barcode_df.loc[mask, "unique_barcodes"].tolist()
        np.random.shuffle(values)  # In-place permutation
        shuffled_cell_barcode_df.loc[mask, "unique_barcodes"] = values

    # Shuffle barcode sets across presynaptic (non-starter) cells
    if shuffle_presyn:
        mask = shuffled_cell_barcode_df["is_starter"] == False
        values = shuffled_cell_barcode_df.loc[mask, "unique_barcodes"].tolist()
        np.random.shuffle(values)  # In-place permutation
        shuffled_cell_barcode_df.loc[mask, "unique_barcodes"] = values

    return shuffled_cell_barcode_df


def shuffle_wrapper(arg):
    """Wrapper for `shuffle_barcodes` for parallel processing.

    This function unpacks arguments and calls `shuffle_barcodes`. It is
    designed to be used with `multiprocessing.Pool.map` or similar
    parallel execution utilities.

    Args:
        arg (tuple): A tuple containing the arguments for `shuffle_barcodes`:
            (seed, minimal_cell_barcode_df, shuffle_presyn, shuffle_starters).
    """
    seed, minimal_cell_barcode_df, shuffle_presyn, shuffle_starters = arg
    return shuffle_barcodes(
        seed, minimal_cell_barcode_df, shuffle_presyn, shuffle_starters
    )


def compare_to_shuffle(
    observed_matrix: pd.DataFrame,
    shuffled_matrices: np.ndarray,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compares an observed matrix to shuffled matrices to find significant differences.

    This function calculates the log2 ratio of an observed connectivity matrix
    to the mean of a distribution of shuffled (null) matrices. It also computes
    empirical p-values and applies False Discovery Rate (FDR) correction.

    Args:
        observed_matrix (pd.DataFrame): The observed connectivity matrix.
        shuffled_matrices (np.ndarray): A 3D numpy array where the first
            dimension represents permutations, and the other two dimensions
            correspond to the rows and columns of the `observed_matrix`.
        alpha (float, optional): The significance level for FDR correction.
            Defaults to 0.05.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - log_ratio_matrix (pd.DataFrame): Matrix of log2 ratios of
              observed values to the mean of the null distributions.
            - pval_corrected_df (pd.DataFrame): Matrix of FDR-corrected p-values.
    """
    # Compute p-values and log ratio of observed connectivity vs mean for bubble plots
    mean_null = shuffled_matrices.mean(axis=0)
    ratio_matrix = observed_matrix.values / (mean_null + 1e-9)
    ratio_matrix = pd.DataFrame(
        ratio_matrix, index=observed_matrix.index, columns=observed_matrix.columns
    )
    log_ratio_matrix = np.log2(ratio_matrix)
    pval_df = compute_empirical_pvalues(
        observed_matrix, shuffled_matrices, two_sided=True
    )
    pval_df = pval_df.loc[log_ratio_matrix.index, log_ratio_matrix.columns]
    # FDR correction
    _, pval_corrected_df = benjamini_hochberg(pval_df, alpha=alpha)
    return log_ratio_matrix, pval_corrected_df


def shuffle_and_compute_connectivity(
    minimal_cell_barcode_df,
    n_permutations=10000,
    shuffle_starters=False,
    shuffle_presyn=True,
    compute_connectivity=True,
    starter_grouping="area_acronym_ancestor_rank1",
    presyn_grouping="area_acronym_ancestor_rank1",
    output_fraction=False,
):
    """
    Shuffle the barcodes within starter / non-starter cells and compute the connectivity matrix.

    Args:
        minimal_cell_barcode_df (pd.DataFrame): DataFrame of ARA starters data
        n_permutations (int): Number of permutations to perform
        n_concurrent_jobs (int): Number of concurrent jobs to run
        shuffle_starters (bool): Whether to shuffle the starter barcodes
        shuffle_presyn (bool): Whether to shuffle the presynaptic barcodes

    Returns:
        observed_confusion_matrix (pd.DataFrame): Observed confusion matrix
        all_null_matrices (list): List of shuffled confusion matrices
    """

    args = [
        (seed, minimal_cell_barcode_df, shuffle_presyn, shuffle_starters)
        for seed in range(n_permutations)
    ]
    shuffled_cell_barcode_dfs = process_map(
        shuffle_wrapper,
        args,
        max_workers=round(min(cpu_count() / 3, cpu_count() - 2)),
        desc="Shuffling data",
        total=n_permutations,
    )

    if not compute_connectivity:
        return shuffled_cell_barcode_dfs
    # Create a partial function that fixes the extra arguments
    compute_connectivity_matrix_ = partial(
        compute_connectivity_matrix,
        starter_grouping=starter_grouping,
        presyn_grouping=presyn_grouping,
        output_fraction=output_fraction,
    )
    results = process_map(
        compute_connectivity_matrix_,
        shuffled_cell_barcode_dfs,
        max_workers=33,
        desc="Computing connectivity",
        total=n_permutations,
    )
    (
        shuffled_matrices,
        mean_input_fractions,
        starter_input_fractions,
        count_matrices,
    ) = zip(*results)
    return (
        shuffled_cell_barcode_dfs,
        shuffled_matrices,
        mean_input_fractions,
        starter_input_fractions,
        count_matrices,
    )


def filter_matrices(
    observed_cm,
    all_null_matrices,
    row_order=[
        "VISp1",
        "VISp2/3",
        "VISp4",
        "VISp5",
        "VISp6a",
        "VISp6b",
        "VISal",
        "VISl",
        "VISli",
        "VISpm",
        "RSP",
        "AUD",
        "TEa",
        "TH",
    ],
    col_order=[
        "VISp1",
        "VISp2/3",
        "VISp4",
        "VISp5",
        "VISp6a",
        "VISp6b",
        "VISl",
        "VISpm",
    ],
):
    """Filters observed and null connectivity matrices to a specified order and subset.

    This function takes an observed connectivity matrix (`observed_cm`) and a
    corresponding 3D array of null matrices (`all_null_matrices`) and filters
    them based on `row_order` and `col_order`. This is useful for focusing
    on specific areas of interest or standardizing matrix dimensions.

    Args:
        observed_cm (pd.DataFrame): The observed connectivity matrix, where
            rows are presynaptic areas and columns are starter areas.
        all_null_matrices (np.ndarray): A 3D numpy array representing the null
            distribution of connectivity matrices (permutations x rows x columns).
            The row and column order must match `observed_cm`.
        row_order (list, optional): A list of row labels (presynaptic areas)
            to include and their desired order. Defaults to a predefined list.
        col_order (list, optional): A list of column labels (starter areas)
            to include and their desired order. Defaults to a predefined list.
    Returns:
        tuple[pd.DataFrame, np.ndarray]:
            - subset_observed_cm (pd.DataFrame): The filtered observed connectivity matrix.
            - subset_null_array (np.ndarray): The filtered 3D array of null matrices.
    """
    # Determine row/column order
    if row_order is None:
        row_order = list(observed_cm.index)
    if col_order is None:
        col_order = list(observed_cm.columns)
    # If you want to ensure only valid areas are used (i.e., ignoring any that
    # don't exist in observed_cm), you can filter them here:
    row_order = [r for r in row_order if r in observed_cm.index]
    col_order = [c for c in col_order if c in observed_cm.columns]
    # Subset the observed_cm
    subset_observed_cm = observed_cm.loc[row_order, col_order]
    # find the integer row/col indices that correspond to row_order/col_order
    row_indices = [observed_cm.index.get_loc(r) for r in row_order]
    col_indices = [observed_cm.columns.get_loc(c) for c in col_order]
    # Reindex the null distribution:
    #   first index permutations (:), then row_indices, then col_indices
    # null_array[:, row_indices, col_indices] => shape: (N, n_rows, n_cols)
    subset_null_array = all_null_matrices[:, row_indices][:, :, col_indices]
    return subset_observed_cm, subset_null_array


def plot_null_histograms_square(
    observed_cm,
    all_null_matrices,
    bins=30,
    row_label_fontsize=14,
    col_label_fontsize=14,
    x_axis_lims=None,
):
    """
    Plot a grid of square histogram subplots, one for each cell in the observed confusion matrix,
    with optional custom ordering/filtering of rows (presynaptic areas) and columns (starter areas).

    Args:
        observed_cm : pd.DataFrame
            Observed confusion matrix (rows = presyn areas, columns = starter areas).
        all_null_matrices : list of np.ndarray
            List of length N_permutations, each a 2D array (n_rows x n_cols) from a shuffle,
            with the same row/col alignment as observed_cm.
        bins : int
            Number of histogram bins.
        row_label_fontsize : int
            Font size for the row (presyn area) label on the left edge.
        col_label_fontsize : int
            Font size for the column (starter area) label on the top edge.
        row_order : list or None
            List of row labels (presynaptic areas) to include and in which order.
            If None, use observed_cm.index as-is.
        col_order : list or None
            List of column labels (starter areas) to include and in which order.
            If None, use observed_cm.columns as-is.
    Returns:
        subset_null_array:

    """
    # Create the figure and axes grid
    # Decide figure size so each subplot can be roughly square
    n_rows, n_cols = observed_cm.shape
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0), sharex=False, sharey=False
    )

    # Helper to handle 1D/2D indexing of axes
    def get_ax(i, j):
        if n_rows == 1 and n_cols == 1:
            return axes
        elif n_rows == 1:
            return axes[j]
        elif n_cols == 1:
            return axes[i]
        else:
            return axes[i, j]

    # Plot each histogram
    for i, row_label in enumerate(observed_cm.index):
        for j, col_label in enumerate(observed_cm.columns):
            ax = get_ax(i, j)
            # Extract the null distribution for this subset cell
            cell_values = all_null_matrices[:, i, j]
            # Observed value
            observed_val = observed_cm.iloc[i, j]
            # Plot the histogram
            ax.hist(cell_values, bins=bins, density=True, alpha=0.5, color="black")
            # Vertical red line at observed
            ax.axvline(observed_val, color="red", linewidth=2)
            # Remove y-axis ticks
            ax.set_yticks([])
            ax.set_yticklabels([])
            # Put the row label on the left edge for the first column in each row
            if j == 0:
                ax.text(
                    -0.3,
                    0.5,
                    str(row_label),
                    rotation=90,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=row_label_fontsize,
                )

            # Put the column label on top for the first row in each column
            if i == 0:
                ax.text(
                    0.5,
                    1.2,
                    str(col_label),
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=col_label_fontsize,
                )
            # Set x-axis limits
            if x_axis_lims is not None:
                ax.set_xlim(x_axis_lims)
    # Add global labels (optional)
    plt.suptitle("Starter cell area", fontsize=16, y=0.93)
    plt.text(
        0.06,
        0.5,
        "Presynaptic cell area",
        fontsize=16,
        rotation=90,
        ha="center",
        va="center",
        transform=fig.transFigure,
    )
    plt.tight_layout()
    plt.show()


def compute_empirical_pvalues(observed_cm, null_array, two_sided=True):
    """
    Compute empirical p-values for each cell in the observed confusion matrix
    based on the distribution of values in all_null_matrices, using a small-sample correction.

    Parameters
    ----------
    observed_cm : pd.DataFrame
        The observed confusion matrix (n_rows x n_cols).
    all_null_matrices : list of np.ndarray
        A list of length N_permutations, each a 2D array with shape (n_rows, n_cols),
        representing the shuffled null distributions. Must align with observed_cm rows/cols.
    two_sided : bool
        If True, compute two-sided empirical p-values. Otherwise, compute right-tailed p-values.

    Returns
    -------
    pval_df : pd.DataFrame
        A DataFrame (same shape as observed_cm) with the empirical p-values.

    Notes
    -----
    - Small-sample correction: We use (count + 1) / (N + 1) instead of count / N.
    - If the observed_cm cell is NaN, we set the corresponding p-value to NaN.
    """
    # Convert list of 2D arrays to a 3D array: (N_permutations, n_rows, n_cols)
    # null_array = np.array(all_null_matrices)  # shape: (N, n_rows, n_cols)
    n_permutations = null_array.shape[0]
    n_rows, n_cols = observed_cm.shape
    # Prepare an output DataFrame for p-values
    pval_df = pd.DataFrame(
        np.zeros((n_rows, n_cols)),
        index=observed_cm.index,
        columns=observed_cm.columns,
        dtype=float,
    )
    for i in range(n_rows):
        for j in range(n_cols):
            observed_val = observed_cm.iat[i, j]
            # If there's no observed value (NaN), set p-value to NaN and continue
            if pd.isna(observed_val):
                pval_df.iat[i, j] = np.nan
                continue
            # Extract the null distribution for this cell across permutations
            cell_null_vals = null_array[:, i, j]
            # Count how many are >= and <= the observed
            count_ge = np.sum(cell_null_vals >= observed_val)
            count_le = np.sum(cell_null_vals <= observed_val)
            # Small-sample correction uses (count + 1)/(N + 1)
            p_right = (count_ge + 1) / (n_permutations + 1)
            p_left = (count_le + 1) / (n_permutations + 1)
            if two_sided:
                # Two-sided p-value
                p_2sided = 2.0 * min(p_left, p_right)
                # Clamp at 1
                p_val = min(p_2sided, 1.0)
            else:
                # One-sided (right-tailed)
                p_val = p_right
            pval_df.iat[i, j] = p_val
    return pval_df


def benjamini_hochberg(pval_df, alpha=0.05):
    """
    Implement Benjamini–Hochberg (BH) FDR correction for a DataFrame of p-values.

    Parameters
    ----------
    pval_df : pd.DataFrame
        DataFrame of raw p-values.
    alpha : float
        Desired FDR level (commonly 0.05).

    Returns
    -------
    rejected_df : pd.DataFrame (bool)
        A boolean DataFrame indicating which null hypotheses are rejected
        at the desired FDR level (True = significant).
    pval_corrected_df : pd.DataFrame (float)
        The BH-adjusted p-values in the same shape as pval_df.
    """
    # Flatten all p-values into a 1D array (excluding NaNs)
    pvals = pval_df.values.flatten()
    not_nan_mask = ~np.isnan(pvals)
    pvals_nonan = pvals[not_nan_mask]
    m = pvals_nonan.size
    # Sort the p-values in ascending order; keep track of their original indices
    sort_idx = np.argsort(pvals_nonan)
    pvals_sorted = pvals_nonan[sort_idx]
    pvals_bh_sorted = np.empty_like(pvals_sorted)
    # BH procedure:
    # pBH(i) = p_sorted(i) * (m / i)
    factor = m / np.arange(1, m + 1)
    pvals_bh_sorted = pvals_sorted * factor
    # Now enforce monotonicity from largest to smallest
    for i in range(m - 2, -1, -1):
        pvals_bh_sorted[i] = min(pvals_bh_sorted[i], pvals_bh_sorted[i + 1])
    # Create an array for the BH-corrected p-values in their *original* order
    pvals_bh = np.full_like(pvals, np.nan, dtype=float)
    idx_non_nan = np.where(not_nan_mask)[0]
    pvals_bh[idx_non_nan[sort_idx]] = pvals_bh_sorted
    # Determine significance:  pBH(i) <= alpha
    rejected = pvals_bh <= alpha
    # Reshape everything back into the original DataFrame shape
    pval_corrected_df = pd.DataFrame(
        pvals_bh.reshape(pval_df.shape), index=pval_df.index, columns=pval_df.columns
    )
    rejected_df = pd.DataFrame(
        rejected.reshape(pval_df.shape), index=pval_df.index, columns=pval_df.columns
    )

    return rejected_df, pval_corrected_df


def plot_log_ratio_matrix(subset_observed_cm, subset_null_array):
    pval_df = compute_empirical_pvalues(
        subset_observed_cm, subset_null_array, two_sided=True
    )
    mean_null = subset_null_array.mean(axis=0)
    std_null = subset_null_array.std(axis=0)
    z_matrix = (subset_observed_cm.values - mean_null) / (std_null + 1e-9)
    z_matrix = pd.DataFrame(
        z_matrix, index=subset_observed_cm.index, columns=subset_observed_cm.columns
    )
    # Calculate the ratio matrix
    ratio_matrix = subset_observed_cm.values / (mean_null + 1e-9)
    ratio_matrix = pd.DataFrame(
        ratio_matrix, index=subset_observed_cm.index, columns=subset_observed_cm.columns
    )
    # Calculate the log ratio matrix
    log_ratio_matrix = np.log2(ratio_matrix)
    # Mask for zero values (log(1) = 0)
    mask = np.isclose(log_ratio_matrix, 0)
    # Ensure pval_df has the same index/columns as log_ratio_matrix
    pval_df = pval_df.loc[log_ratio_matrix.index, log_ratio_matrix.columns]

    # Function to format p-values in scientific notation (1 decimal before 'e')
    def format_pval(x):
        if x == 0:
            return "0"
        formatted = "{:.1e}".format(x)  # Format with 1 decimal
        base, exp = formatted.split("e")  # Split into base and exponent
        return f"{base}e{int(exp)}"  # Remove leading zeros in exponent

    # Identify significant cells where p-value < 0.05
    significant_cells = pval_df < 0.05
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        log_ratio_matrix,
        cmap="coolwarm",
        center=0,
        annot=False,  # Disable built-in annotations
        square=True,
        mask=mask,
        cbar_kws={"label": "log2 Ratio"},
        vmax=0.30,
        vmin=-0.30,
    )
    # Manually add text annotations
    for i in range(log_ratio_matrix.shape[0]):
        for j in range(log_ratio_matrix.shape[1]):
            log_ratio = log_ratio_matrix.iloc[i, j]
            p_val = pval_df.iloc[i, j]
            p_val_text = format_pval(p_val)
            # If log_ratio is finite (not -inf), display it
            if np.isfinite(log_ratio):
                ax.text(
                    j + 0.5,
                    i + 0.4,
                    f"{log_ratio:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
            # Always display the p-value in smaller text
            ax.text(
                j + 0.5,
                i + 0.7,
                p_val_text,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
            # Add black outline if significant (p < 0.05)
            if significant_cells.iloc[i, j]:
                rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2)
                ax.add_patch(rect)

    # Add grey border around the whole matrix
    ax.axhline(y=0, color="grey", linewidth=5)
    ax.axhline(y=log_ratio_matrix.shape[0], color="grey", linewidth=5)
    ax.axvline(x=0, color="grey", linewidth=5)
    ax.axvline(x=log_ratio_matrix.shape[1], color="grey", linewidth=5)
    # Set the limits to ensure the borders are fully visible
    ax.set_xlim(0, log_ratio_matrix.shape[1])
    ax.set_ylim(log_ratio_matrix.shape[0], 0)
    plt.title(
        "Log Ratio of Observed vs. Shuffled Null with P-values",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Starter area", fontsize=14, fontweight="bold")
    plt.ylabel("Presynaptic area", fontsize=14, fontweight="bold")
    plt.show()


def bubble_plot(
    log_ratio_matrix: pd.DataFrame,
    pval_df: pd.DataFrame,
    alpha: float = 0.05,
    size_scale: float = 800,
    label_fontsize: int = 12,
    tick_fontsize: int = 12,
    ax: plt.Axes = None,
    cbax: plt.Axes = None,
    show_legend: bool = True,
    vmin: float = None,
    vmax: float = None,
    bubble_lw: float = 0.7,
):
    """Generates a bubble plot to visualize log-ratios and p-values.

    This function creates a scatter plot where each bubble represents a
    cell in the input matrices. The size of the bubble corresponds to the
    absolute log-ratio, and the color corresponds to the signed -log2(p-value).
    Significant cells (p-value < alpha) are outlined in black.

    Args:
        log_ratio_matrix: DataFrame of log-ratios, where rows and columns
            represent categories being compared.
        pval_df: DataFrame of p-values corresponding to the log_ratio_matrix.
        alpha: Significance level for highlighting bubbles.
        size_scale: Scaling factor for bubble sizes.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels and colorbar.
        ax: Matplotlib Axes to plot on. If None, uses the current Axes.
        cbax: Matplotlib Axes for the colorbar. If None, no colorbar is drawn.
        show_legend: If True, displays a legend for bubble sizes.
        vmin: Minimum value for the colormap normalization of color_value.
        vmax: Maximum value for the colormap normalization of color_value.
        bubble_lw: Linewidth for the outline of significant bubbles.
    """
    # Reformat input dfs into a long-form DataFrame for plotting
    row_name = log_ratio_matrix.index.name or "row_label"
    col_name = log_ratio_matrix.columns.name or "col_label"
    log_ratio_matrix.index.name = row_name
    log_ratio_matrix.columns.name = col_name
    df_plot = (
        log_ratio_matrix.stack()
        .reset_index()
        .rename(columns={row_name: "y_label", col_name: "x_label", 0: "log_ratio"})
    )
    df_plot["p_value"] = pval_df.stack().values
    df_plot["x"] = pd.Categorical(
        df_plot["x_label"], categories=log_ratio_matrix.columns, ordered=True
    ).codes
    df_plot["y"] = pd.Categorical(
        df_plot["y_label"], categories=log_ratio_matrix.index, ordered=True
    ).codes
    x_categories = log_ratio_matrix.columns
    y_categories = log_ratio_matrix.index
    # Calculate bubble size & color value
    # Bubble size: absolute log ratio * size_scale
    df_plot["bubble_size"] = df_plot["log_ratio"].abs() * size_scale
    # Color value = sign(log_ratio) * -log2(p_value)
    # => Positive log-ratio => red, negative => blue
    df_plot["color_value"] = np.sign(df_plot["log_ratio"]) * -np.log2(
        df_plot["p_value"].clip(lower=1e-300)
    )
    # Main scatter
    sc = ax.scatter(
        x=df_plot["x"],
        y=df_plot["y"],
        s=df_plot["bubble_size"],
        c=df_plot["color_value"],
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )

    # Add black outlines for significant cells
    significant_cells = pval_df < alpha
    is_signif = significant_cells.stack().values
    df_signif = df_plot[is_signif]
    ax.scatter(
        x=df_signif["x"],
        y=df_signif["y"],
        s=df_signif["bubble_size"],
        facecolors="none",
        edgecolors="black",
        linewidths=bubble_lw,
    )
    if cbax:
        plt.colorbar(sc, cax=cbax, ax=ax)
        cbax.set_title("Signed\n$\log_{2}$ p-value", fontsize=tick_fontsize, loc="left")
        cbax.tick_params(
            axis="both",
            which="both",
            pad=2,  # brings tick labels closer
            labelsize=tick_fontsize,
        )

    if show_legend:
        legend_values = [0.3, 0.6, 1.0]
        legend_handles = []
        for val in legend_values:
            size_val = val * size_scale
            h = ax.scatter([], [], s=size_val, c="gray", alpha=0.5, label=val)
            legend_handles.append(h)

        legend = ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(1.05, 0),
            borderaxespad=0.0,
            frameon=False,
            handleheight=3.0,
            fontsize=tick_fontsize,
            title="$|\log_{2}\\frac{\mathrm{observed}}{\mathrm{shuffled}}|$",
        )
        legend.get_title().set_fontsize("6")
    ax.set_xlim([-0.5, len(x_categories) - 0.5])
    ax.set_ylim([-0.5, len(y_categories) - 0.5])

    ax.set_xticks(range(len(x_categories)))
    ax.set_yticks(range(len(y_categories)))
    ax.set_xticklabels(x_categories)
    ax.set_yticklabels(y_categories)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    # ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Invert y-axis so top row is y=0
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        "Starter layer",
        fontsize=label_fontsize,
    )
    ax.set_ylabel("Presynaptic layer", fontsize=label_fontsize)


def _truncate_colormap(
    cmap: mcolors.Colormap, start: float = 0.0, end: float = 1.0, n: int = 256
) -> mcolors.Colormap:
    """
    Return a new Colormap that maps the interval [start, end] of *cmap*
    onto the full [0, 1] range.
    """
    if not (0.0 <= start < end <= 1.0):
        raise ValueError("start and end must satisfy 0 ≤ start < end ≤ 1.")
    colors = cmap(np.linspace(start, end, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_trunc_{start:.2f}_{end:.2f}", colors
    )


def connectivity_diagram_mpl(
    mean_input_fraction: pd.DataFrame,
    lower_df: pd.DataFrame,
    upper_df: pd.DataFrame,
    connection_names: list[str],
    positions: dict[str, tuple[float, float]],
    display_names: list[str] = None,
    ax: plt.Axes = None,
    min_fraction_cutoff: float = 0.2,
    node_style: dict = None,
    arrow_style: dict = None,
    ci_to_alpha: bool = True,
    ci_cmap: str | None = None,
    ci_cmap_start: float = 0.0,
    ci_cmap_end: float = 1.0,
    edge_width_scale: float = 2.0,
    arrow_head_scale: float = 20.0,
    vmin: float = None,
    vmax: float = None,
    colorbar_fontsize=6,
    cax=None,
):
    """
    Generates a Matplotlib diagram representing neural connectivity.

    Nodes are plotted as circles with labels. Edges are arrows whose width
    is proportional to `mean_input_fraction`. Edge color and/or transparency
    can be modulated by the confidence interval width (derived from `lower_df`,
    `upper_df`).

    Args:
        mean_input_fraction: DataFrame of mean input fractions.
            Rows are presynaptic regions, columns are starter regions.
        lower_df: DataFrame of lower confidence bounds for `mean_input_fraction`.
        upper_df: DataFrame of upper confidence bounds for `mean_input_fraction`.
        connection_names: List of columns of `mean_input_fraction` to use to plot
            connections. Must have corresponding entries in `positions`.
        positions: Dictionary of x,y position for nodes, keys must correspond to
            `connection_names`.
        display_names: List of names for nodes, corresponding to `connection_names`. If
            None, will use connection_names
        ax: Matplotlib Axes to plot on. If None, a new figure and axes
            are created.
        min_fraction_cutoff: Minimum input fraction for an edge to be drawn.
        node_style (dict, optional): Styling for nodes. Keys can include:
            'radius' (float): Radius of node circles.
            'facecolor' (str): Face color of node circles.
            'edgecolor' (str): Edge color of node circles.
            'fontsize' (int): Font size for node labels.
            'fontcolor' (str): Color for node labels.
        arrow_style (dict, optional): Styling for arrows.
        ci_to_alpha (bool, optional): If True, edge transparency is inversely
            related to the confidence interval width (narrower CI = more opaque).
        ci_cmap (str | None, optional): Matplotlib colormap name to color edges
            based on confidence interval width. If None, edges are black
            (or colored by `ci_to_alpha` logic if active).
        edge_width_scale (float, optional): Multiplier for connection_strength
            to determine edge linewidth. (e.g., strength * scale = linewidth).
        arrow_head_scale (float, optional): Multiplier for connection_strength
            to determine arrow head size (mutation_scale for FancyArrowPatch).
        vmin (float, optional): Minimum value for colormap normalization.
        vmax (float, optional): Maximum value for colormap normalization.
        colorbar_fontsize (int, optional): Font size for colormap legend.
        cax (mpl.Axis, optional): Matplotlib axis for colorbar, default to None


    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
            The figure and axes used for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))  # Adjust default size as needed
    else:
        fig = ax.get_figure()
    if display_names is None:
        display_names = {n: n for n in connection_names}
    else:
        display_names = {n: d for n, d in zip(connection_names, display_names)}
    # Initialize styles with defaults
    default_node_style = {
        "radius": 0.5,
        "facecolor": "LightGray",
        "edgecolor": "black",
        "fontsize": 7,
        "fontcolor": "black",
    }
    current_node_style = default_node_style.copy()
    if node_style:
        current_node_style.update(node_style)
    default_arrow_style = dict(
        connectionstyle="arc3,rad=0.0",  # Straight line
    )

    current_arrow_style = default_arrow_style.copy()
    if arrow_style:
        current_arrow_style.update(arrow_style)

    node_radius = current_node_style["radius"]

    # Plot nodes
    for name, pos_xy in positions.items():
        circle = plt.Circle(
            pos_xy,
            radius=node_radius,
            facecolor=current_node_style["facecolor"],
            edgecolor=current_node_style["edgecolor"],
            zorder=2,
        )
        ax.add_patch(circle)
        ax.text(
            pos_xy[0],
            pos_xy[1],
            display_names[name],
            ha="center",
            va="center",
            fontsize=current_node_style["fontsize"],
            color=current_node_style["fontcolor"],
            zorder=3,
        )

    # --- Edge plotting preparation ---
    # Calculate confidence interval ranges
    conf_ranges_df = upper_df - lower_df
    valid_mask = mean_input_fraction >= min_fraction_cutoff
    # Flatten valid confidence ranges, remove NaNs, ensure non-negative
    valid_conf_values = conf_ranges_df[valid_mask].values.flatten()

    inv_conf = valid_conf_values
    if vmax is None:
        vmax = np.nanmax(inv_conf) if len(valid_conf_values) > 0 else 1.0
    if vmin is None:
        vmin = np.nanmin(inv_conf)
    if vmax < 1e-9:
        print("Very low max inverse CI")
        vmax = 1.0
    if ci_cmap is not None:
        base_cmap = plt.get_cmap(ci_cmap)
        cmap_obj = _truncate_colormap(base_cmap, start=ci_cmap_start, end=ci_cmap_end)
    # Plot edges
    for starter_name in connection_names:  # Target of the connection
        for presyn_name in connection_names:  # Source of the connection
            # Ensure names are valid matrix keys and have positions
            if not (
                starter_name in mean_input_fraction.columns
                and presyn_name in mean_input_fraction.index
                and presyn_name in positions
                and starter_name in positions
            ):
                continue
            connection_strength = mean_input_fraction.loc[presyn_name, starter_name]
            if connection_strength < min_fraction_cutoff:
                continue

            conf_range = conf_ranges_df.loc[presyn_name, starter_name]
            assert ~np.isnan(conf_range)

            # --- Determine edge color and alpha ---
            edge_plot_color_rgb = (0.0, 0.0, 0.0)  # Default: black
            edge_plot_alpha = 1.0  # Default: opaque

            if ci_cmap is not None:
                norm_cr = (conf_range - vmin) / (vmax - vmin)
                rgba_from_cmap = cmap_obj(norm_cr)
                edge_plot_color_rgb = rgba_from_cmap[:3]  # RGB part
                edge_plot_alpha = rgba_from_cmap[3]  # Alpha from colormap

            if ci_to_alpha:
                current_alpha_val = (
                    0.1  # Default for large/invalid conf_range (mostly transparent)
                )
                if conf_range > 1e-9 and vmax > 1e-9:
                    # Alpha is (1/conf_range) normalized by max(1/conf_range)
                    # Higher alpha for smaller confidence interval
                    current_alpha_val = np.clip(
                        (1.0 / conf_range) / vmax,
                        0.05,  # Minimum alpha to ensure visibility
                        1.0,
                    )
                elif conf_range <= 1e-9:  # Very small (good) confidence range -> opaque
                    current_alpha_val = 1.0

                if (
                    ci_cmap is None
                ):  # If no colormap, base color for this alpha is black
                    edge_plot_color_rgb = (0.0, 0.0, 0.0)
                # If ci_cmap was used, edge_plot_color_rgb is already set.
                edge_plot_alpha = current_alpha_val  # Override alpha

            # --- Get node positions and adjust arrow for node radius ---
            source_pos = positions[presyn_name]
            target_pos = positions[starter_name]

            # Handle self-loops (edge from a node to itself)
            if source_pos == target_pos:
                source_pos = (
                    source_pos[0] - 0.5 * node_radius,
                    source_pos[1] - 1.2 * node_radius,
                )
                target_pos = (
                    target_pos[0] - 1 * node_radius,
                    target_pos[1] + 0.8 * node_radius,
                )
                arrow_style = current_arrow_style.copy()
                arrow_style["connectionstyle"] = "Arc3, rad=-1.5"
                rad2use = 0
            else:
                arrow_style = current_arrow_style.copy()
                rad2use = node_radius

            _draw_arrow_mpl(
                ax,
                source_pos,
                target_pos,
                rad2use,
                connection_strength,
                edge_plot_color_rgb,
                edge_plot_alpha,
                edge_width_scale,
                arrow_head_scale,
                arrow_style,
            )

    # --- Final plot adjustments ---
    all_x_coords, all_y_coords = np.vstack(list(positions.values())).T
    if len(all_x_coords):
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)
        # Add padding based on extent of nodes and node radius
        x_padding = (x_max - x_min) * 0.1 + node_radius
        y_padding = (y_max - y_min) * 0.1 + node_radius
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_aspect("equal", adjustable="box")  # Ensure aspect ratio is equal
    ax.axis("off")  # Turn off axis lines and ticks for a cleaner diagram

    # Add colorbar if ci_cmap was used and a valid range was determined
    if ci_cmap is not None and cmap_obj is not None:
        if not np.isclose(
            vmin, vmax
        ):  # Only add colorbar if there's a meaningful range
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)

            cbar = fig.colorbar(
                sm,
                ax=ax,
                cax=cax,
                orientation="vertical",
                fraction=0.1,
                pad=0.05,
                shrink=0.4,
            )
            cbar.set_label("CI width", fontsize=colorbar_fontsize)
            cbar.ax.tick_params(labelsize=colorbar_fontsize)
        # else: No colorbar if min and max for normalization are the same (no range to show)

    return fig, ax, cbar


def _draw_arrow_mpl(
    ax: plt.Axes,
    source_pos: tuple[float, float],
    target_pos: tuple[float, float],
    node_radius: float,
    connection_strength: float,
    edge_plot_color_rgb: tuple[float, float, float],
    edge_plot_alpha: float,
    edge_width_scale: float,
    arrow_head_scale: float,
    arrow_style_kwargs: dict,
):
    """
    Helper function to draw a directed arrow between two nodes.

    The arrow starts and ends at the perimeters of the node circles.
    Its appearance (width, head size, color, alpha) is determined by input parameters.

    Args:
        ax: Matplotlib Axes object.
        source_pos: (x, y) coordinates of the source node center.
        target_pos: (x, y) coordinates of the target node center.
        node_radius: Radius of the node circles.
        connection_strength: Strength of the connection, used to scale width/head.
        edge_plot_color_rgb: RGB tuple for arrow color.
        edge_plot_alpha: Alpha (transparency) for the arrow.
        edge_width_scale: Scaling factor for linewidth (strength * scale).
        arrow_head_scale: Scaling factor for arrow head size (strength * scale).
        arrow_style_kwargs: Dictionary of base styling for FancyArrowPatch.
    """
    dx, dy = target_pos[0] - source_pos[0], target_pos[1] - source_pos[1]
    dist = np.sqrt(dx**2 + dy**2)

    # Adjust start/end points to originate/terminate at the edge of node circles
    eff_start_x = source_pos[0] + (dx / dist) * node_radius
    eff_start_y = source_pos[1] + (dy / dist) * node_radius
    eff_end_x = target_pos[0] - (dx / dist) * node_radius
    eff_end_y = target_pos[1] - (dy / dist) * node_radius
    props = dict(
        alpha=edge_plot_alpha,
        fc=edge_plot_color_rgb,
        width=connection_strength * edge_width_scale,
        headwidth=connection_strength * arrow_head_scale,
        headlength=connection_strength * arrow_head_scale,
    )
    props.update(arrow_style_kwargs)
    ax.annotate(
        "",
        xytext=(eff_start_x, eff_start_y),
        xy=(eff_end_x, eff_end_y),
        arrowprops=props,
    )
