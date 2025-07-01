import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from functools import partial

from brisc.manuscript_analysis.connectivity_matrices import compute_connectivity_matrix


def bootstrap_cells(cell_df, n_samples=None, random_state=None):
    """
    Perform a bootstrap sample from rows where is_starter == True.

    Parameters
    ----------
    cell_df : pd.DataFrame
        The original DataFrame
    n_samples : int, optional
        Number of rows to sample. If None, use the number of rows.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the bootstrap sample.
    """
    if n_samples is None:
        n_samples = len(cell_df)

    bootstrap_sample = cell_df.sample(
        n=n_samples, replace=True, random_state=random_state
    )

    return bootstrap_sample


def hierarchical_bootstrap(
    cell_barcode_df,
    resample_starters=True,
    resample_presynaptic=True,
    random_state=None,
):
    """
    Perform a hierarchical bootstrap on the cell_barcode_df DataFrame,
    creating a *new* DataFrame with bootstrapped samples.

    Parameters
    ----------
    cell_barcode_df : pd.DataFrame
        The original DataFrame containing cell barcodes and their properties.
        Must have columns:
            - 'is_starter': bool
            - 'unique_barcodes': e.g., set or list of barcodes
    resample_starters : bool, optional
        Whether to resample the starter cells. Default is True.
    resample_presynaptic : bool, optional
        Whether to resample the presynaptic cells that share a barcode
        with each sampled starter. Default is True.
    random_state : int, optional
        Random seed for reproducibility.
        Note: Using the same random_state for repeated calls to
        'bootstrap_cells' will produce the same sample each time.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the hierarchical bootstrap sample.
    """
    cell_barcode_df["mask_uid"] = cell_barcode_df.index
    starter_cells = cell_barcode_df[cell_barcode_df["is_starter"]].copy()
    presyn_cells = cell_barcode_df[cell_barcode_df["is_starter"] == False].copy()

    if resample_starters:
        sampled_starters = bootstrap_cells(starter_cells, random_state=random_state)
    else:
        sampled_starters = starter_cells.copy()

    # Collect all dataframes (starters + presyn) to merge later
    result_dfs = [sampled_starters]

    # For each sampled starter cell, find relevant presyn cells
    if resample_presynaptic:
        for _, starter_row in sampled_starters.iterrows():
            starter_barcodes = starter_row["unique_barcodes"]
            # Find presyn cells that share at least one barcode
            presyn_subset = presyn_cells[
                presyn_cells["unique_barcodes"].apply(
                    lambda x: len(starter_barcodes & x) > 0
                )
            ]
            # Bootstrap the presyn subset
            sampled_presyn = bootstrap_cells(presyn_subset, random_state=random_state)
            result_dfs.append(sampled_presyn)
    else:
        # If not resampling presyn cells, just gather them
        for _, starter_row in sampled_starters.iterrows():
            starter_barcodes = starter_row["unique_barcodes"]
            presyn_subset = presyn_cells[
                presyn_cells["unique_barcodes"].apply(
                    lambda x: len(starter_barcodes & x) > 0
                )
            ]
            result_dfs.append(presyn_subset)

    bootstrapped_df = pd.concat(result_dfs, ignore_index=True)

    # Remove duplicate presynaptic cells
    presyn_cells = bootstrapped_df[
        bootstrapped_df["is_starter"] == False
    ].drop_duplicates(subset=["mask_uid"])
    starter_cells = bootstrapped_df[bootstrapped_df["is_starter"]]
    bootstrapped_df = pd.concat([presyn_cells, starter_cells], ignore_index=True)

    return bootstrapped_df


def hierarchical_bootstrap_wrapper(args):
    """
    Thin wrapper to unpack arguments and call hierarchical_bootstrap
    with a specific random_state (seed).
    """
    seed, cell_barcode_df, resample_starters, resample_presynaptic = args
    return hierarchical_bootstrap(
        cell_barcode_df=cell_barcode_df,
        resample_starters=resample_starters,
        resample_presynaptic=resample_presynaptic,
        random_state=seed,
    )


def repeated_hierarchical_bootstrap_in_parallel(
    cell_barcode_df,
    n_permutations=10000,
    resample_starters=True,
    resample_presynaptic=True,
    compute_connectivity=True,
    starter_grouping="area_acronym_ancestor_rank1",
    presyn_grouping="area_acronym_ancestor_rank1",
):
    """
    Run hierarchical_bootstrap n_iterations times in parallel, each with a seed
    from 1..n_iterations, and return the resulting DataFrames.

    Parameters
    ----------
    cell_barcode_df : pd.DataFrame
        Your original data of cells and barcodes (must have "is_starter" and "unique_barcodes").
    n_iterations : int
        Number of bootstrap iterations to run. Defaults to 10,000.
    resample_starters : bool
        Whether to resample (bootstrap) the starter cells in each iteration.
    resample_presynaptic : bool
        Whether to resample the presynaptic cells in each iteration.

    Returns
    -------
    list of pd.DataFrame
        List of bootstrapped DataFrames (one per iteration).
    """

    # Build argument list for all seeds
    args = [
        (seed, cell_barcode_df, resample_starters, resample_presynaptic)
        for seed in range(1, n_permutations + 1)
    ]

    # Use process_map to run in parallel
    bootstrapped_results = process_map(
        hierarchical_bootstrap_wrapper,
        args,
        max_workers=round(cpu_count() / 3),
        desc="Hierarchical Bootstrapping",
        total=n_permutations,
    )

    if compute_connectivity:
        # Create a partial function that fixes the extra arguments
        partial_func = partial(
            compute_connectivity_matrix,
            starter_grouping=starter_grouping,
            presyn_grouping=presyn_grouping,
        )

        # Run it in parallel
        results = process_map(
            partial_func,
            bootstrapped_results,
            max_workers=33,
            desc="Computing connectivity",
            total=n_permutations,
        )

        shuffled_matrices, mean_input_fractions, starter_input_fractions, _ = zip(
            *results
        )

        return (
            bootstrapped_results,
            shuffled_matrices,
            mean_input_fractions,
            starter_input_fractions,
        )
    else:
        return bootstrapped_results


def reorder_dfs(dfs, target):
    """
    Given an observed_cm (DataFrame) and a list of DataFrames, reindex each
    DataFrame so they share the same rows and columns (in the same order) as
    observed_cm. Missing rows/columns are filled with 0.

    Args:
        observed_cm (pd.DataFrame): Reference DataFrame whose row/column order
                                    we want to enforce.
        dfs (list of pd.DataFrame): DataFrames to reindex.

    Returns:
        list of pd.DataFrame: Each reindexed to match observed_cm rows/columns.
    """
    row_order = target.index
    col_order = target.columns
    reindexed_dfs = [
        df.reindex(index=row_order, columns=col_order, fill_value=0) for df in dfs
    ]

    return reindexed_dfs


def compute_percentile_matrices(bootstrap_list, lower_p=2.5, upper_p=97.5):
    """
    Given a list of DataFrames of the same shape (same rows and columns),
    return two DataFrames: lower percentile and upper percentile matrix.
    """

    df = bootstrap_list[0]
    rows = df.index
    cols = df.columns
    lower_df = pd.DataFrame(index=rows, columns=cols, dtype=float)
    upper_df = pd.DataFrame(index=rows, columns=cols, dtype=float)

    # For each cell (row, col), gather the values across all bootstrap DataFrames
    for r in rows:
        for c in cols:
            values = [df.loc[r, c] for df in bootstrap_list]
            lp = np.percentile(values, lower_p)
            up = np.percentile(values, upper_p)
            lower_df.loc[r, c] = lp
            upper_df.loc[r, c] = up

    return lower_df, upper_df


def plot_confidence_intervals(
    mean_input_frac_df,
    lower_df,
    upper_df,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    line_width=1,
    orientation="vertical",
):
    """
    Plot confidence intervals in multiple stacked Axes created via make_axes_locatable.

    Parameters
    ----------
    mean_input_frac_df : pandas.DataFrame
        DataFrame of mean input fractions (index = presyn area, columns = target area).
    lower_df : pandas.DataFrame
        DataFrame of lower confidence bounds matching mean_input_frac_df shape.
    upper_df : pandas.DataFrame
        DataFrame of upper confidence bounds matching mean_input_frac_df shape.
    ax : matplotlib.axes.Axes, optional
        The Axes into which the first subplot is drawn. If None, a new figure & Axes are created.

    Returns
    -------
    (fig, axes) : (matplotlib.figure.Figure, list of matplotlib.axes.Axes)
        The figure and list of axes (one per column in mean_input_frac_df).
    """
    # Determine the presynaptic areas (rows) and target areas (columns)
    presyn_area_order = sorted(mean_input_frac_df.index)
    areas = sorted(mean_input_frac_df.columns)
    num_subplots = len(areas)

    # Create a divider on the initial Axes
    divider = make_axes_locatable(ax)
    axes = [ax]
    direction = "bottom" if orientation == "vertical" else "right"
    for _ in range(num_subplots - 1):
        ax_new = divider.append_axes(direction, size="100%", pad=0.025)
        axes.append(ax_new)

    for i, area in enumerate(areas):
        ax_curr = axes[i]
        m = np.array(mean_input_frac_df.loc[presyn_area_order, area])
        for line in [0.3, 0.6]:
            if orientation == "vertical":
                ax_curr.axhline(line, color="black", linestyle=":", linewidth=0.5)
            else:
                ax_curr.axvline(line, color="black", linestyle=":", linewidth=0.5)
        if orientation == "vertical":
            ax_curr.bar(
                np.arange(len(m)),
                m,
                color="mediumorchid",
                alpha=0.8,
                edgecolor="darkorchid",
                linewidth=line_width,
            )
            # Error bars
            ax_curr.errorbar(
                np.arange(len(m)),
                m,
                np.abs(
                    np.vstack(
                        [
                            lower_df.loc[presyn_area_order, area],
                            upper_df.loc[presyn_area_order, area],
                        ]
                    )
                    - m[None, :]
                ),
                fmt="none",
                markerfacecolor="mediumorchid",
                markeredgecolor="darkorchid",
                markersize=5,
                ecolor="darkorchid",
                elinewidth=line_width,
            )
            ax_curr.set_ylim(0, 0.6)
            ax_curr.set_yticks([0, 0.6])
            ax_curr.set_ylabel(area, fontsize=tick_fontsize)
            ax_curr.set_xticks([])
        else:
            ax_curr.barh(
                np.arange(len(m)),
                m,
                xerr=np.abs(
                    np.vstack(
                        [
                            lower_df.loc[presyn_area_order, area],
                            upper_df.loc[presyn_area_order, area],
                        ]
                    )
                    - m[None, :]
                ),
                color="mediumorchid",
                alpha=0.8,
                edgecolor="darkorchid",
                linewidth=line_width,
            )
            ax_curr.set_xlim(0, 0.6)
            ax_curr.set_xticks([0.0, 0.6], labels=["0", "0.6"])
            for label, x in zip(ax_curr.get_xticklabels(), [0.0, 0.6]):
                if x == 0.0:
                    label.set_ha("left")  # left-align the label
                elif x == 0.6:
                    label.set_ha("right")  # right-align the label
            if i > 0:
                ax_curr.set_yticks([])
            ax_curr.invert_yaxis()
            ax_curr.text(
                0.3,
                -1.0,
                s=area,
                ha="center",
                va="center",
                color="black",
                fontsize=tick_fontsize,
            )
        ax_curr.spines["top"].set_visible(False)
        ax_curr.spines["right"].set_visible(False)
        ax_curr.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    if orientation == "vertical":
        axes[-1].set_xticks(np.arange(len(presyn_area_order)))
        axes[-1].set_xticklabels(presyn_area_order, rotation=0)
        axes[-1].spines["bottom"].set_visible(True)
        axes[-1].set_xlabel("Presynaptic layer", fontsize=label_fontsize)
    else:
        axes[0].set_yticks(np.arange(len(presyn_area_order)), labels=presyn_area_order)
        axes[0].set_ylabel("Presynaptic layer", fontsize=label_fontsize)
        axes[2].set_title("Starter layer", fontsize=label_fontsize, loc="left", pad=15)
        axes[2].set_xlabel("Input fraction", fontsize=label_fontsize, loc="left")
