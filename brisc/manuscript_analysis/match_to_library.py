from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import functools

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
from brisc.manuscript_analysis.utils import despine

matplotlib.rcParams[
    "pdf.fonttype"
] = 42  # Use Type 3 fonts (TrueType) for selectable text
matplotlib.rcParams["ps.fonttype"] = 42  # For EPS, if relevant
plt.rcParams.update({"mathtext.default": "regular"})  # make math mode also Arial


def _hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


# Define a function to calculate the minimum edit distance
def _calculate_min_edit_distance_worker(insitu_bc, lib_10bp_seq_ref, rv35_library_ref):
    edit_distances = np.fromiter(
        (_hamming_distance(insitu_bc, lib_bc) for lib_bc in lib_10bp_seq_ref), int
    )
    min_edit_distance_idx = np.argmin(edit_distances)
    min_edit_distance = edit_distances[min_edit_distance_idx]
    lib_bc_sequence = rv35_library_ref.loc[min_edit_distance_idx, "10bp_seq"]
    lib_bc_count = rv35_library_ref.loc[min_edit_distance_idx, "counts"]
    return min_edit_distance, lib_bc_sequence, lib_bc_count


def load_data(
    redo=False,
    barseq_path=Path("/nemo/project/proj-znamenp-barseq"),
    main_path=Path("/nemo/lab/znamenskiyp"),
    error_correction_ds_name="BRAC8498.3e_error_corrected_barcodes_26",
):
    processed_path = barseq_path / "processed/becalia_rabies_barseq/BRAC8498.3e/"
    ara_is_starters = pd.read_pickle(
        processed_path / "analysis" / f"{error_correction_ds_name}_cell_barcode_df.pkl"
    )
    #    ara_is_starters = pd.read_pickle(
    #    processed_path / "analysis" / "merged_cell_df_curated_mcherry.pkl"
    # )
    ara_is_starters = ara_is_starters[ara_is_starters["all_barcodes"].notna()]

    barcode_library_sequence_path = (
        main_path
        / "home/shared/projects/barcode_diversity_analysis/collapsed_barcodes/RV35/RV35_bowtie_ed2.txt"
    )
    rv35_library = pd.read_csv(barcode_library_sequence_path, sep="\t", header=None)
    rv35_library["10bp_seq"] = rv35_library[1].str.slice(0, 10)
    rv35_library.rename(columns={0: "counts", 1: "sequence"}, inplace=True)

    lib_10bp_seq = np.array(rv35_library["10bp_seq"])
    in_situ_barcodes_path = (
        processed_path / "analysis" / f"{error_correction_ds_name}_in_situ_barcodes.pkl"
    )
    random_df_path = (
        processed_path / "analysis" / f"{error_correction_ds_name}_random_barcodes.pkl"
    )

    if redo or not (in_situ_barcodes_path.exists() and random_df_path.exists()):
        in_situ_barcodes = ara_is_starters["all_barcodes"].explode().unique()
        in_situ_barcodes = pd.DataFrame(in_situ_barcodes, columns=["sequence"])

        # Create a partial function with fixed library arguments
        partial_worker = functools.partial(
            _calculate_min_edit_distance_worker,
            lib_10bp_seq_ref=lib_10bp_seq,
            rv35_library_ref=rv35_library,
        )
        # Wrap the outer loop with tqdm for progress tracking
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(partial_worker, in_situ_barcodes["sequence"]),
                    total=len(in_situ_barcodes),
                    desc="Calculating edit distances",
                )
            )

        # Extract the results from the list of tuples
        min_edit_distances, lib_bc_sequences, lib_bc_counts = zip(*results)

        # Assign the minimum edit distances, lib_bc sequences, and counts to new columns in in_situ_barcodes
        in_situ_barcodes["ham_min_edit_distance"] = min_edit_distances
        in_situ_barcodes["ham_lib_bc_sequence"] = lib_bc_sequences
        in_situ_barcodes["ham_lib_bc_counts"] = lib_bc_counts

        # Generate random DNA sequences
        num_sequences = in_situ_barcodes.shape[0]
        sequence_length = 10
        random_seed = 42  # add some meaning
        np.random.seed(random_seed)
        random_sequences = np.array(
            [
                "".join(np.random.choice(["A", "C", "G", "T"], size=sequence_length))
                for _ in range(num_sequences)
            ],
            dtype=object,
        )

        # Wrap the outer loop with tqdm for progress tracking
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(partial_worker, random_sequences),
                    total=len(random_sequences),
                    desc="Calculating edit distances",
                )
            )

        # Extract the results from the list of tuples
        min_edit_distances, lib_bc_sequences, lib_bc_counts = zip(*results)

        random_df = pd.DataFrame(random_sequences, columns=["random_sequences"])
        # Assign the minimum edit distances, lib_bc sequences, and counts to new columns in in_situ_barcodes
        random_df["min_edit_distance"] = min_edit_distances
        random_df["lib_bc_sequence"] = lib_bc_sequences
        random_df["lib_bc_counts"] = lib_bc_counts
        random_df.to_pickle(random_df_path)
        in_situ_barcodes.to_pickle(in_situ_barcodes_path)
    else:
        in_situ_barcodes = pd.read_pickle(in_situ_barcodes_path)
        random_df = pd.read_pickle(random_df_path)

    return (
        in_situ_barcodes,
        random_df,
        rv35_library,
    )


def plot_matches_to_library(
    in_situ_barcode_matches,
    random_barcode_matches,
    rv35_library,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    linewidth=0.5,
    hist_edgewith=0.5,
    num_bins=20,
    alpha=1,
):
    # Define bin edges for consistent binning
    bin_edges = np.logspace(0, 6, num=num_bins)
    bin_edges = np.insert(bin_edges, 0, 0)
    in_situ_barcode_matches.loc[
        in_situ_barcode_matches["ham_min_edit_distance"] > 0, "ham_lib_bc_counts"
    ] = 0
    random_barcode_matches.loc[
        random_barcode_matches["min_edit_distance"] > 0, "lib_bc_counts"
    ] = 0

    # Compute histograms
    in_situ_hist, _ = np.histogram(
        in_situ_barcode_matches["ham_lib_bc_counts"].values, bins=bin_edges
    )
    # hist1, _ = np.histogram(data1, bins=bin_edges)
    random_hist, _ = np.histogram(
        random_barcode_matches["lib_bc_counts"].values, bins=bin_edges
    )

    # Normalize histograms (scale max to 1)
    in_situ_hist = in_situ_hist / np.sum(in_situ_hist)
    # hist1 = hist1 / np.max(hist1)
    random_hist = random_hist / np.sum(random_hist)

    # Extract and normalize library sequence data
    sequences = np.flip(rv35_library["counts"])
    edge_positions = sequences.searchsorted(bin_edges)
    counts = np.zeros(len(bin_edges) - 1)

    for i in range(len(bin_edges) - 1):
        parts = edge_positions[i : i + 2]
        counts[i] = sequences[parts[0] : parts[1]].sum()

    counts = counts / np.sum(counts)  # Normalize to max 1

    # Plot normalized histograms as step-line plots
    for this_ax in ax:
        plt.sca(this_ax)
        # Plot normalized library sequence data as a black line
        plt.stairs(
            counts[1:],
            bin_edges[1:],
            linestyle="-",
            linewidth=linewidth,  # hist_edgewith,
            label="Viral library barcodes",
            fill=False,
            edgecolor="black",
            # facecolor="slategray",
        )

        plt.stairs(
            in_situ_hist[1:],
            bin_edges[1:],
            color="dodgerblue",
            linestyle="-",
            linewidth=linewidth,
            label="In situ barcodes",
            alpha=alpha,
        )
        plt.stairs(
            [
                in_situ_hist[0],
            ],
            [0.03, 0.06],
            color="dodgerblue",
            linestyle="-",
            linewidth=linewidth,
            alpha=alpha,
        )
        plt.stairs(
            random_hist[1:],
            bin_edges[1:],
            color="dodgerblue",
            linestyle=(0, (2, 1)),
            linewidth=linewidth,
            label="Random barcodes",
        )
        plt.stairs(
            [
                random_hist[0],
            ],
            [0.03, 0.06],
            color="dodgerblue",
            linestyle=(0, (2, 1)),
            linewidth=linewidth,
        )

        # X-axis log scale
        this_ax.set_xscale("log")

    # X-axis formatting
    ax[0].set_xlabel(
        "Proportion of unique reads in viral library per barcode",
        fontsize=label_fontsize,
    )
    # ax[0].xaxis.set_major_locator(mticker.FixedLocator(locs=np.logspace(0, 6, 5)))
    # ax[0].xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    total_read_in_library = np.sum(rv35_library["counts"])
    xticklab = np.array([1e-8, 1e-5, 1e-2])
    xtick = xticklab * total_read_in_library
    ax[0].set_xticks(
        np.hstack([np.sqrt(0.03 * 0.06), xtick]),
        labels=["$0$", "$10^{-8}$", "$10^{-5}$", "$10^{-2}$"],
    )

    # Y-axis label
    slate4label = np.array(colors.to_rgb("slategray")) * 0  # Keep it black
    twinxs = [ax[0].twinx(), ax[1].twinx()]
    twinxs[0].set_ylabel(
        "             Proportion of unique reads",
        fontsize=label_fontsize,
        color=slate4label,
    )
    ax[0].set_ylabel(
        "             Proportion of barcodes",
        fontsize=label_fontsize,
        color="dodgerblue",
    )
    for iax, x in enumerate([ax, twinxs]):
        x[0].set_ylim(0, 0.2)
        x[1].set_ylim(0.6, 0.65)
        for this_ax in x:
            despine(this_ax)
            this_ax.tick_params(
                axis="both",
                which="major",
                labelsize=tick_fontsize,
            )
        x[0].set_yticks(
            [0, 0.1, 0.2],
            labels=[0, 0.1, 0.2],
            color=slate4label if iax else "dodgerblue",
        )
        x[1].set_yticks(
            [0.6, 0.65], labels=[0.6, 0.65], color=slate4label if iax else "dodgerblue"
        )
        x[1].spines.bottom.set_visible(False)
        x[1].set_xticks([])
        x[0].spines.right.set_visible(True)
        x[1].spines.right.set_visible(True)

    ax[1].legend(loc="upper right", fontsize=tick_fontsize, frameon=False)
