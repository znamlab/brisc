import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import iss_preprocess as iss
from iss_preprocess.io import get_processed_path

matplotlib.rcParams[
    "pdf.fonttype"
] = 42  # Use Type 3 fonts (TrueType) for selectable text
matplotlib.rcParams["ps.fonttype"] = 42  # For EPS, if relevant


data_path = "becalia_rabies_barseq/BRAC8498.3e/chamber_07"
processed_path = get_processed_path(data_path)

ara_is_starters = pd.read_pickle(
    processed_path.parent / "analysis" / "merged_cell_df_curated_mcherry.pkl"
)
ara_is_starters = ara_is_starters[ara_is_starters["all_barcodes"].notna()]


def shorten_barcodes(barcode_list):
    return [bc[:10] for bc in barcode_list]


ara_is_starters["all_barcodes"] = ara_is_starters["all_barcodes"].apply(
    shorten_barcodes
)


def load_data(data_path="becalia_rabies_barseq/BRAC8498.3e/chamber_07"):
    processed_path = iss.io.get_processed_path(data_path)
    ara_is_starters = pd.read_pickle(
        processed_path.parent / "analysis" / "merged_cell_df_curated_mcherry.pkl"
    )
    ara_is_starters = ara_is_starters[ara_is_starters["all_barcodes"].notna()]

    # Assuming ara_is_starters is your dataframe
    def shorten_barcodes(barcodes):
        return [barcode[:10] for barcode in barcodes]

    ara_is_starters["all_barcodes"] = ara_is_starters["all_barcodes"].apply(
        shorten_barcodes
    )

    # Barcodes per cell
    unfiltered_bar_per_presynaptic_cell = ara_is_starters[
        ara_is_starters["is_starter"] == False
    ]["n_unique_barcodes"].values
    unfiltered_bar_per_is_starter_cell = ara_is_starters[
        ara_is_starters["is_starter"] == True
    ]["n_unique_barcodes"].values

    # Exploding all_barcodes to allow searching in individual barcodes
    exploded_data = ara_is_starters.explode("all_barcodes")
    # Filtering cells with valid barcodes in all_barcodes
    barcoded_cells = exploded_data[exploded_data["all_barcodes"].notna()]
    # Filtering cells where is_starter is False
    non_starter_cells = barcoded_cells[barcoded_cells["is_starter"] == False]
    # Finding all barcodes where is_starter is True
    starter_barcodes = barcoded_cells[barcoded_cells["is_starter"] == True][
        "all_barcodes"
    ].unique()
    # Subset 1: Non-starters with corresponding starters
    non_starter_with_starter = non_starter_cells[
        non_starter_cells["all_barcodes"].isin(starter_barcodes)
    ]
    # Subset 2: Non-starters without corresponding starters
    non_starter_without_starter = non_starter_cells[
        ~non_starter_cells["all_barcodes"].isin(starter_barcodes)
    ]
    # Grouping and counting for both subsets
    counts_with_starter = non_starter_with_starter.groupby("all_barcodes").size()
    counts_without_starter = non_starter_without_starter.groupby("all_barcodes").size()

    # Cells per barcode
    unfiltered_non_is_starter_barcodes = (
        ara_is_starters[ara_is_starters["is_starter"] == False]["all_barcodes"]
        .explode()
        .unique()
    )
    # include barcodes with no presynaptic cells
    unfiltered_is_starter_barcodes = (
        ara_is_starters[ara_is_starters["is_starter"] == True]["all_barcodes"]
        .explode()
        .unique()
    )
    unfiltered_barcodes_not_in_is_starters = unfiltered_non_is_starter_barcodes[
        ~np.isin(unfiltered_non_is_starter_barcodes, unfiltered_is_starter_barcodes)
    ].shape[0]
    unfiltered_is_starter_cells_per_barcode = (
        ara_is_starters[ara_is_starters["is_starter"] == True]["all_barcodes"]
        .explode()
        .value_counts()
        .values
    )
    starters_per_barcode = np.concatenate(
        [
            np.zeros(unfiltered_barcodes_not_in_is_starters),
            unfiltered_is_starter_cells_per_barcode,
        ]
    )

    return (
        unfiltered_bar_per_presynaptic_cell,
        unfiltered_bar_per_is_starter_cell,
        counts_with_starter,
        counts_without_starter,
        starters_per_barcode,
    )


def plot_bc_per_cell_presyn(
    unfiltered_bar_per_presynaptic_cell,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    padding=1500,
):
    ax.hist(
        unfiltered_bar_per_presynaptic_cell,
        bins=np.arange(1, unfiltered_bar_per_presynaptic_cell.max() + 2, 1) - 0.5,
        color="darkslategrey",
        histtype="stepfilled",
        edgecolor="black",
        alpha=0.8,
    )

    bin_edges = np.arange(1, unfiltered_bar_per_presynaptic_cell.max() + 2, 1) - 0.5
    hist_values, _ = np.histogram(unfiltered_bar_per_presynaptic_cell, bins=bin_edges)
    unfiltered_values, _ = np.histogram(
        unfiltered_bar_per_presynaptic_cell, bins=bin_edges
    )

    padding = 1500  # Adjust as needed

    for i, (val_filtered, val_unfiltered) in enumerate(
        zip(hist_values, unfiltered_values)
    ):
        # Annotate unfiltered data with padding if heights are similar
        y_offset = (
            val_filtered + padding
            if abs(val_filtered - val_unfiltered) < padding
            else val_unfiltered
        )
        ax.text(
            i + 1,
            y_offset + 1,
            str(val_unfiltered),
            ha="center",
            fontsize=8,
            color="black",
            alpha=0.8,
        )

    ax.set_xlabel(
        "Unique barcodes per presynaptic cell",
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        "Number of presynaptic cells",
        fontsize=label_fontsize,
    )
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(np.arange(1, 7, 1))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    return ax


def plot_bc_per_cell_starter(
    unfiltered_bar_per_starter_cell,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    padding=1500,
):
    ax.hist(
        unfiltered_bar_per_starter_cell,
        bins=np.arange(1, unfiltered_bar_per_starter_cell.max() + 2, 1) - 0.5,
        color="darkslategrey",
        histtype="stepfilled",
        edgecolor="black",
        alpha=0.8,
    )

    bin_edges = np.arange(1, unfiltered_bar_per_starter_cell.max() + 2, 1) - 0.5
    hist_values, _ = np.histogram(unfiltered_bar_per_starter_cell, bins=bin_edges)
    unfiltered_values, _ = np.histogram(unfiltered_bar_per_starter_cell, bins=bin_edges)

    for i, (val_filtered, val_unfiltered) in enumerate(
        zip(hist_values, unfiltered_values)
    ):
        # Annotate unfiltered data with padding if heights are similar
        y_offset = (
            val_filtered + padding
            if abs(val_filtered - val_unfiltered) < padding
            else val_unfiltered
        )
        ax.text(
            i + 1,
            y_offset + 1,
            str(val_unfiltered),
            ha="center",
            fontsize=8,
            color="black",
            alpha=0.8,
        )

    ax.set_xlabel(
        "Unique barcodes per starter cell",
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        "Number of starter cells",
        fontsize=label_fontsize,
    )
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(np.arange(1, 7, 1))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    return ax


def plot_presyn_per_orphan(
    counts_with_starter,
    counts_without_starter,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    line_width=0.9,
):
    # Regular histogram (left subplot)
    # Non-starters *with* starter
    ax.hist(
        counts_with_starter,
        bins=range(1, 50),
        alpha=0.7,
        edgecolor="black",
        linewidth=line_width,
        histtype="step",
        align="left",
        label="With Starter",
    )

    # Non-starters *without* starter
    ax.hist(
        counts_without_starter,
        bins=range(1, 50),
        alpha=0.7,
        edgecolor="blue",
        linewidth=line_width,
        histtype="step",
        align="left",
        label="Without Starter",
    )

    ax.set_xlim(0, 50)
    ax.set_xlabel(
        "Number of Non-Starter Cells per Barcode",
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        "Frequency",
        fontsize=label_fontsize,
    )
    ax.legend(
        loc="upper right",
        fontsize=tick_fontsize,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    plt.tight_layout()

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

    def shorten_barcodes(barcode_list):
        return [bc[:10] for bc in barcode_list]

    cells["all_barcodes"] = cells["all_barcodes"].apply(shorten_barcodes)
    starter_cells = cells[cells.is_starter == True]
    presyn_cells = cells[cells.is_starter == False]

    # Identify 'singleton' barcodes among starters
    # i.e. barcodes that appear exactly once across all starter cells
    all_starter_barcodes = []
    for bc_list in starter_cells["all_barcodes"]:
        all_starter_barcodes.extend(bc_list)

    barcode_counts = pd.Series(all_starter_barcodes).value_counts()
    singletons = set(barcode_counts.index[barcode_counts == 1])

    # For each cell, define 'unique_barcodes' = intersection of its barcodes with singletons
    cells["unique_barcodes"] = cells["all_barcodes"].apply(
        lambda x: singletons.intersection(x)
    )

    # Among starter cells, pick out those whose unique_barcodes has length == 2
    double_barcoded_starters = cells[
        (cells["is_starter"] == True) & (cells["unique_barcodes"].apply(len) == 2)
    ].copy()

    double_barcoded_starters["all_barcodes"] = double_barcoded_starters[
        "unique_barcodes"
    ].apply(list)
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
