from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tf
from matplotlib import pyplot as plt
import matplotlib
import iss_preprocess as iss

matplotlib.rcParams["pdf.fonttype"] = (
    42  # Use Type 3 fonts (TrueType) for selectable text
)
matplotlib.rcParams["ps.fonttype"] = 42  # For EPS, if relevant


data_path = "becalia_rabies_barseq/BRAC8498.3e/chamber_07"
processed_path = iss.io.get_processed_path(data_path)

ara_is_starters = pd.read_pickle(
    processed_path.parent / "analysis" / "merged_cell_df_curated_mcherry.pkl"
)
ara_is_starters = ara_is_starters[ara_is_starters["main_barcode"].notna()]


def shorten_barcodes(barcode_list):
    return [bc[:10] for bc in barcode_list]


ara_is_starters["all_barcodes"] = ara_is_starters["all_barcodes"].apply(
    shorten_barcodes
)
ara_is_starters["main_barcode"] = ara_is_starters["main_barcode"].apply(
    lambda x: x[:10]
)


def load_data(data_path="becalia_rabies_barseq/BRAC8498.3e/chamber_07"):
    processed_path = iss.io.get_processed_path(data_path)
    ara_is_starters = pd.read_pickle(
        processed_path.parent / "analysis" / "merged_cell_df_curated_mcherry.pkl"
    )
    ara_is_starters = ara_is_starters[ara_is_starters["main_barcode"].notna()]

    # Assuming ara_is_starters is your dataframe
    def shorten_barcodes(barcodes):
        return [barcode[:10] for barcode in barcodes]

    ara_is_starters["all_barcodes"] = ara_is_starters["all_barcodes"].apply(
        shorten_barcodes
    )
    ara_is_starters["main_barcode"] = ara_is_starters["main_barcode"].apply(
        lambda x: x[:10]
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

    return (
        unfiltered_bar_per_presynaptic_cell,
        unfiltered_bar_per_is_starter_cell,
        counts_with_starter,
        counts_without_starter,
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
