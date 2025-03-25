import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from brisc.manuscript_analysis.utils import despine


def plot_hist(
    data_df,
    ax,
    col="n_unique_barcodes",
    tick_fontsize=12,
    y_offset=0.05,
    max_val=None,
    show_zero=False,
    show_counts=True,
):
    if max_val is None:
        max_val = data_df[col].max() + 1
    if show_zero:
        min_val = 0
    else:
        min_val = 1
    counts = np.bincount(data_df[col].values, minlength=max_val + 1)
    props = counts / np.sum(counts)
    counts = counts[min_val : max_val + 1]
    props = props[min_val : max_val + 1]
    plt.stairs(
        props,
        np.arange(min_val, max_val + 2) - 0.5,
        fill=True,
        edgecolor="black",
        facecolor="slategray",
        linewidth=0.5,
    )
    if show_counts:
        for i, (count, prop) in enumerate(zip(counts, props)):
            # Annotate unfiltered data with padding if heights are similar
            plt.text(
                i + min_val - 0.2,
                y_offset + prop,
                str(count),
                ha="left",
                fontsize=tick_fontsize,
                color="black",
                alpha=0.8,
                rotation=35,
            )

    ax.set_ylim(0, 1)
    ax.set_xlim(min_val - 0.5, max_val + 0.5)

    ax.set_xticks(np.arange(min_val, max_val + 1, 1))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
    despine(ax)


def plot_presyn_per_barcode(
    barcodes_df,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    max_val=50,
    colors=("darkorange", "dodgerblue"),
):
    cells_with_starter = barcodes_df[barcodes_df["n_starters"] > 0][
        "n_presynaptic"
    ].values
    cells_without_starter = barcodes_df[barcodes_df["n_starters"] == 0][
        "n_presynaptic"
    ].values
    # Regular histogram (left subplot)
    # Non-starters *with* starter
    bin_edges = np.arange(0, max_val + 5, 5)
    bin_edges[-1] = 1e4
    print(f"Bin edges: {bin_edges}")
    for cells, color in zip((cells_without_starter, cells_with_starter), colors):
        counts, _ = np.histogram(cells, bins=bin_edges)
        print(f"Counts: {counts}")
        props = counts / np.sum(counts)
        plt.stairs(
            props,
            bin_edges,
            fill=False,
            linewidth=0.5,
            color=color,
        )

    ax.set_xlim(0, max_val)
    ax.set_xticks(np.arange(0, max_val + 10, 10))

    ax.set_xlabel(
        "mCherry- cells per barcode",
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        "Proportion of barcodes",
        fontsize=label_fontsize,
    )
    ax.legend(
        ["Orphan barcodes", "Non-orphan barcodes"],
        loc="upper right",
        fontsize=tick_fontsize,
        frameon=False,
        bbox_to_anchor=[1.3, 1],
        handlelength=1,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
    despine(ax)

    return ax


def plot_starters_per_barcode(
    starters_per_barcode,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    padding=200,
):
    ax.hist(
        starters_per_barcode,
        bins=np.arange(0, starters_per_barcode.max() + 2, 1) - 0.5,
        histtype="stepfilled",
        edgecolor="black",
        color="lightblue",
        alpha=0.8,
    )

    bin_edges = np.arange(0, starters_per_barcode.max() + 2, 1) - 0.5
    unfiltered_values, _ = np.histogram(starters_per_barcode, bins=bin_edges)

    for i, val_unfiltered in enumerate(unfiltered_values):
        y_offset = val_unfiltered + padding
        ax.text(
            i,
            y_offset + 1,
            str(val_unfiltered),
            ha="center",
            fontsize=label_fontsize,
            color="black",
            alpha=0.8,
        )

    ax.set_ylabel(
        "Number of barcodes",
        fontsize=label_fontsize,
    )
    ax.set_xlabel(
        "Starter cells per barcode",
        fontsize=label_fontsize,
    )
    ax.set_xlim(-0.5, 11.5)
    ax.set_xticks(np.arange(0, 12, 1))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )


def find_singleton_bcs(cells):
    """
    Find singleton barcodes that appear exactly once across all starter cells
    and defines 'unique_barcodes' for all cells that is the intersection of its
    barcodes with singletons

    Args:
        cells (pd.DataFrame): DataFrame of cells

    Returns:
        cells (pd.DataFrame): DataFrame of cells with 'unique_barcodes' column
    """
    starter_cells = cells[cells.is_starter == True]

    # Identify 'singleton' barcodes among starters
    all_starter_barcodes = []
    for bc_list in starter_cells["all_barcodes"]:
        all_starter_barcodes.extend(bc_list)

    barcode_counts = pd.Series(all_starter_barcodes).value_counts()
    singletons = set(barcode_counts.index[barcode_counts == 1])

    # For each cell, define 'unique_barcodes' = intersection of its barcodes with singletons
    cells["unique_barcodes"] = cells["all_barcodes"].apply(
        lambda x: singletons.intersection(x)
    )

    return cells


def load_double_barcode_data(
    data_path,
):
    """
    Load data for double barcodes analysis

    Args:
        data_path (str): Path to the data folder

    Returns:
        result_df (pd.DataFrame): DataFrame with columns:
        - starter_cell_id (str): ID of the starter cell
        - barcode1 (str): First barcode
        - barcode2 (str): Second barcode
        - n_presyn_with_barcode1 (int): Number of presynaptic cells with barcode1
        - n_presyn_with_barcode2 (int): Number of presynaptic cells with barcode2
        - n_presyn_with_both (int): Number of presynaptic cells with both barcodes
    """
    cells = pd.read_pickle(f"{data_path}/cell_barcode_df.pkl")
    # Only keep rows that actually have barcodes
    cells = cells.dropna(subset=["all_barcodes"])

    cells = find_singleton_bcs(cells)

    # Among starter cells, pick out those whose unique_barcodes has length == 2
    double_barcoded_starters = cells[
        (cells["is_starter"] == True) & (cells["unique_barcodes"].apply(len) == 2)
    ].copy()

    double_barcoded_starters["all_barcodes"] = double_barcoded_starters[
        "unique_barcodes"
    ].apply(list)
    presyn_cells = cells[cells["is_starter"] == False]
    presyn_exploded = presyn_cells[["cell_id", "all_barcodes"]].explode("all_barcodes")

    # For each double-barcoded starter, count how many presyn cells have bc1, bc2, or both
    results = []
    for i, row in double_barcoded_starters.iterrows():
        starter_id = row["cell_id"]
        barcodes = row["all_barcodes"]
        b1, b2 = barcodes[0], barcodes[1]

        presyn_with_b1 = set(
            presyn_exploded.loc[presyn_exploded["all_barcodes"] == b1, "cell_id"]
        )
        presyn_with_b2 = set(
            presyn_exploded.loc[presyn_exploded["all_barcodes"] == b2, "cell_id"]
        )

        n_presyn_b1 = len(presyn_with_b1)
        n_presyn_b2 = len(presyn_with_b2)
        n_presyn_both = len(presyn_with_b1.intersection(presyn_with_b2))
        results.append(
            {
                "starter_cell_id": starter_id,
                "barcode1": b1,
                "barcode2": b2,
                "n_presyn_with_barcode1": n_presyn_b1,
                "n_presyn_with_barcode2": n_presyn_b2,
                "n_presyn_with_both": n_presyn_both,
            }
        )
    result_df = pd.DataFrame(results)

    return result_df


def plot_double_barcode_barstack(
    result_df,
    ax_stack=None,
    ax_violin=None,
    label_fontsize=12,
    tick_fontsize=12,
    barcode_proportion=True,
    swarmplot_dotsize=2,
):
    """
    Plot a barstack and violin plot of the number of presynaptic cells with each barcode

    Args:
        ax_stack (matplotlib.axes.Axes): Axes for the barstack plot
        ax_violin (matplotlib.axes.Axes): Axes for the violin plot
        result_df (pd.DataFrame): DataFrame with columns:
            - starter_cell_id (str): ID of the starter cell
            - barcode1 (str): First barcode
            - barcode2 (str): Second barcode
            - n_presyn_with_barcode1 (int): Number of presynaptic cells with barcode1
            - n_presyn_with_barcode2 (int): Number of presynaptic cells with barcode2
            - n_presyn_with_both (int): Number of presynaptic cells with both barcodes
    """

    # Create columns for "barcode1 only", "both", "barcode2 only"
    result_df["b1_only"] = (
        result_df["n_presyn_with_barcode1"] - result_df["n_presyn_with_both"]
    )
    result_df["b2_only"] = (
        result_df["n_presyn_with_barcode2"] - result_df["n_presyn_with_both"]
    )
    result_df["both"] = result_df["n_presyn_with_both"]

    # Compute total presynaptic cells and sort
    result_df["total_presyn"] = (
        result_df["b1_only"] + result_df["both"] + result_df["b2_only"]
    )
    result_df_sorted = result_df.sort_values(
        by="total_presyn", ascending=False
    ).reset_index(drop=True)

    # Compute fraction of the dominant barcode
    union_count = (
        result_df_sorted["n_presyn_with_barcode1"]
        + result_df_sorted["n_presyn_with_barcode2"]
        - result_df_sorted["n_presyn_with_both"]
    ).replace(0, np.nan)
    dominant_count = np.maximum(
        result_df_sorted["n_presyn_with_barcode1"],
        result_df_sorted["n_presyn_with_barcode2"],
    )

    result_df_sorted["frac_dominant"] = (dominant_count / union_count).replace(
        [np.inf, -np.inf], np.nan
    )

    # Compute the proportion of presynaptic cells that have both barcodes
    result_df_sorted["both_prop"] = (
        result_df_sorted["both"] / result_df_sorted["total_presyn"]
    )

    # Sort first by largest "both_prop", then by largest "frac_dominant"
    result_df_sorted = result_df_sorted.sort_values(
        by=["both_prop", "frac_dominant"], ascending=False
    ).reset_index(drop=True)

    b1_only = result_df_sorted["b1_only"]
    b2_only = result_df_sorted["b2_only"]
    both = result_df_sorted["both"]

    # If barcode_proportion is True, convert counts to proportions
    if barcode_proportion:
        b1_only = b1_only / result_df_sorted["total_presyn"]
        b2_only = b2_only / result_df_sorted["total_presyn"]
        both = both / result_df_sorted["total_presyn"]

    b_most = np.maximum(b1_only, b2_only)
    b_least = np.minimum(b1_only, b2_only)

    x = range(len(result_df_sorted))

    # Bottom segment: most abundant
    ax_stack.bar(x, b_most, label="Most Abundant Barcode", color="lightskyblue")

    # Middle: both
    ax_stack.bar(x, both, bottom=b_most, label="Both", color="red")

    # Top: least abundant
    ax_stack.bar(
        x,
        b_least,
        bottom=b_most + both,
        label="Least Abundant Barcode",
        color="lightgrey",  # "#b2df8a",
    )

    # ax_stack.set_xticks([])
    ax_stack.set_ylabel(
        "Fraction of Presynaptic Cells",
        fontsize=label_fontsize,
    )
    ax_stack.set_xlabel(
        "Starter Cell",
        fontsize=label_fontsize,
    )
    ax_stack.legend(
        fontsize=tick_fontsize,
        bbox_to_anchor=(0.8, -0.2),
        handlelength=1,
    )
    ax_stack.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    vals = result_df_sorted["frac_dominant"].dropna()
    parts = ax_violin.violinplot(
        [vals],
        positions=[0],
        widths=0.6,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    # Adjust alpha on the violin body
    for pc in parts["bodies"]:
        pc.set_alpha(0.3)
        pc.set_facecolor("gray")

    # Swarm plot with small jitter
    x_jitter = np.random.uniform(-0.05, 0.05, size=len(vals))
    ax_violin.scatter(
        x_jitter, vals, marker="o", alpha=0.8, edgecolors="black", s=swarmplot_dotsize
    )

    ax_violin.set_xticks([0])
    ax_violin.set_xticklabels([""])
    ax_violin.set_xlim(-0.5, 0.5)
    ax_violin.set_ylim(0.5, 1.05)
    ax_violin.set_ylabel(
        "Fraction of presyn cells\nwith the most abundant barcode",
        fontsize=label_fontsize,
    )
    ax_violin.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
