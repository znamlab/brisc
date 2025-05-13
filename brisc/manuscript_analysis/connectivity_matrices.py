import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from functools import partial


def match_barcodes(cells_df):
    def match_barcodes_(series, barcodes):
        return series.apply(lambda bcs: len(bcs.intersection(barcodes)) > 0)

    starters = cells_df[cells_df["is_starter"] == True]
    connectivity_matrix = cells_df["unique_barcodes"].apply(lambda bcs: match_barcodes_(starters["unique_barcodes"], bcs))
    cells_df["starters"] = connectivity_matrix.apply(lambda row: set(row.index[row]), axis=1)
    cells_df["n_starters"] = cells_df["starters"].apply(len)


def compute_input_fractions(starter_row, presyn_cells, presyn_grouping):
    """
    For a single starter cell row, find all presynaptic cells sharing at least one barcode,
    then compute the counts and input fraction of those presynaptic cells coming from each presyn group.

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
    total_counts_df = counts_df.groupby(starter_grouping).sum().T
    if output_fraction:
        # For each presyn area (each row), divide by the row sum so it sums to 1
        mean_frac_df = total_counts_df.div(total_counts_df.sum(axis=1), axis=0)
    else:
        mean_frac_df = fractions_df.groupby(starter_grouping).mean().T

    return total_counts_df, mean_frac_df, fractions_df, counts_df


def compute_odds_ratio(p_matrix, starter_counts):
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
    cbar_label="Input fraction"
):
    vmin = np.min(connectivity_matrix[connectivity_matrix != -np.inf]) * 0.7
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
        cbax.patch.set_edgecolor('black')
        cbax.patch.set_linewidth(1.5)
        cbax.set_title(cbar_label, fontsize=tick_fontsize, loc="left")     

    # Annotate with appropriate color based on background
    for (i, j), val in np.ndenumerate(connectivity_matrix):
        if odds_ratio:
            q1 = np.percentile(connectivity_matrix, 15)
            q3 = np.percentile(connectivity_matrix, 85)
            text_color = "white" if val <= q1 or val >= q3 else "black"
        else:
            text_color = (
                "white"
                if val < connectivity_matrix.max(axis=None) / 2
                else "black"
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
    ax.set_xlabel("Starter layer", fontsize=label_fontsize)
    ax.set_ylabel("Presynaptic layer", fontsize=label_fontsize)

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
        -.15,
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
    seed, minimal_cell_barcode_df, shuffle_presyn, shuffle_starters = arg
    return shuffle_barcodes(
        seed, minimal_cell_barcode_df, shuffle_presyn, shuffle_starters
    )


def compare_to_shuffle(observed_matrix, shuffled_matrices, alpha=0.05):
    # Compute p-values and log ratio of observed connectivity vs mean for bubble plots
    mean_null = shuffled_matrices.mean(axis=0)
    ratio_matrix = observed_matrix.values / (mean_null + 1e-9)
    ratio_matrix = pd.DataFrame(
        ratio_matrix, index=observed_matrix.index, columns=observed_matrix.columns
    )
    log_ratio_matrix = np.log10(ratio_matrix)
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
        max_workers=round(cpu_count() / 3),
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
    shuffled_matrices, mean_input_fractions, starter_input_fractions, count_matrices = zip(*results)
    return (
        shuffled_cell_barcode_dfs,
        shuffled_matrices,
        mean_input_fractions,
        starter_input_fractions,
        count_matrices
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
    log_ratio_matrix = np.log10(ratio_matrix)
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
        cbar_kws={"label": "Log10 Ratio"},
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
    log_ratio_matrix,
    pval_df,
    alpha=0.05,
    size_scale=800,
    label_fontsize=12,
    tick_fontsize=12,
    ax=None,
    cbax=None,
    show_legend=True,
):
    """
    Create a bubble plot to visualize log-ratios with p-values.
    Args:
    - log_ratio_matrix: pd.DataFrame, shape (n_rows, n_cols)
        DataFrame of log-ratios (log₁₀(observed / expected)).
    - pval_df: pd.DataFrame, shape (n_rows, n_cols)
        DataFrame of p-values for each log-ratio.
    - size_scale: int, default 800
        Scaling factor for bubble sizes.
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
    # Color value = sign(log_ratio) * -log10(p_value)
    # => Positive log-ratio => red, negative => blue
    df_plot["color_value"] = np.sign(df_plot["log_ratio"]) * -np.log10(
        df_plot["p_value"].clip(lower=1e-300)
    )
    # Main scatter
    sc = ax.scatter(
        x=df_plot["x"],
        y=df_plot["y"],
        s=df_plot["bubble_size"],
        c=df_plot["color_value"],
        cmap="coolwarm",
        vmin=np.log10(alpha),
        vmax=-np.log10(alpha),
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
        linewidths=0.5,
    )
    if cbax:
        plt.colorbar(sc, cax=cbax, ax=ax)
        cbax.set_title("Signed\n$\log_{10}$ p-value", fontsize=tick_fontsize, loc="left")
        cbax.tick_params(
            axis="both",
            which="both",
            pad=2,  # brings tick labels closer
            labelsize=tick_fontsize,
        )

    if show_legend:
        legend_values = [0.2, 0.4, 0.6]
        legend_handles = []
        for val in legend_values:
            size_val = val * size_scale
            h = ax.scatter(
                [], [], s=size_val, c="gray", alpha=0.5, label=val
            )
            legend_handles.append(h)

        legend = ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(1.05, 0),
            borderaxespad=0.0,
            frameon=False,
            handleheight=3.0,
            fontsize=tick_fontsize,
            title="$|\log_{10}\\frac{\mathrm{observed}}{\mathrm{shuffled}}|$"
        )
        legend.get_title().set_fontsize('6') 
    ax.set_xlim([-.5, len(x_categories)-.5])
    ax.set_ylim([-.5, len(y_categories)-.5])

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
    ax.set_xlabel("Starter layer", fontsize=label_fontsize, )
    ax.set_ylabel("Presynaptic layer", fontsize=label_fontsize)
