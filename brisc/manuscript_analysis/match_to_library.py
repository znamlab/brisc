from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = (
    42  # Use Type 3 fonts (TrueType) for selectable text
)
matplotlib.rcParams["ps.fonttype"] = 42  # For EPS, if relevant


def load_data():
    processed_path = Path(
        "/nemo/project/proj-znamenp-barseq/processed/becalia_rabies_barseq/BRAC8498.3e/"
    )
    ara_is_starters = pd.read_pickle(
        processed_path / "analysis" / "merged_cell_df_curated_mcherry.pkl"
    )
    ara_is_starters = ara_is_starters[ara_is_starters["main_barcode"].notna()]
    in_situ_barcodes = ara_is_starters["all_barcodes"].explode().unique()
    in_situ_barcodes = pd.DataFrame(in_situ_barcodes, columns=["sequence"])

    barcode_library_sequence_path = Path(
        "/nemo/lab/znamenskiyp/home/shared/projects/barcode_diversity_analysis/collapsed_barcodes/RV35/RV35_bowtie_ed2.txt"
    )
    rv35_library = pd.read_csv(barcode_library_sequence_path, sep="\t", header=None)
    rv35_library["10bp_seq"] = rv35_library[1].str.slice(0, 10)
    rv35_library.rename(columns={0: "counts", 1: "sequence"}, inplace=True)

    def hamming_distance(str1, str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    # Define a function to calculate the minimum edit distance
    def calculate_min_edit_distance(insitu_bc):
        edit_distances = np.fromiter(
            (hamming_distance(insitu_bc, lib_bc) for lib_bc in lib_10bp_seq), int
        )
        min_edit_distance_idx = np.argmin(edit_distances)
        min_edit_distance = edit_distances[min_edit_distance_idx]
        lib_bc_sequence = rv35_library.loc[min_edit_distance_idx, "10bp_seq"]
        lib_bc_count = rv35_library.loc[min_edit_distance_idx, "counts"]
        return min_edit_distance, lib_bc_sequence, lib_bc_count

    redo = False

    lib_10bp_seq = np.array(rv35_library["10bp_seq"])
    if redo:
        # Wrap the outer loop with tqdm for progress tracking
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(
                        calculate_min_edit_distance, in_situ_barcodes["sequence"]
                    ),
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
                    pool.imap(calculate_min_edit_distance, random_sequences),
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
        random_df.to_pickle(
            "/nemo/lab/znamenskiyp/home/users/becalia/data/BRYC65.1d/random_2252_barcodes.pkl"
        )

    else:
        in_situ_barcodes = pd.read_pickle(
            "/nemo/lab/znamenskiyp/home/users/becalia/code/brisc/brisc/cells_and_barcodes/in_situ_barcodes.pkl"
        )
        random_df = pd.read_pickle(
            "/nemo/lab/znamenskiyp/home/users/becalia/data/BRYC65.1d/random_2252_barcodes.pkl"
        )

    def shorten_barcodes(barcode_list):
        return [bc[:10] for bc in barcode_list]

    ara_is_starters["all_barcodes"] = ara_is_starters["all_barcodes"].apply(
        shorten_barcodes
    )
    ara_is_starters["main_barcode"] = ara_is_starters["main_barcode"].apply(
        lambda x: x[:10]
    )

    barcoded_cells = ara_is_starters[ara_is_starters["main_barcode"].notna()]

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

    # Non-starters without corresponding starters
    non_starter_without_starter = non_starter_cells[
        ~non_starter_cells["all_barcodes"].isin(starter_barcodes)
    ]

    # Group and filter for non_starter_without_starter
    grouped_barcodes_without = non_starter_without_starter.groupby("all_barcodes")
    groups_of_size_1_without = grouped_barcodes_without.filter(lambda x: len(x) == 1)

    unique_single_pre_no_starter = groups_of_size_1_without.all_barcodes.unique()

    unique_single_pre_no_starter = pd.DataFrame(
        unique_single_pre_no_starter, columns=["sequence"]
    )

    in_situ_perfect_match = in_situ_barcodes[
        in_situ_barcodes["ham_min_edit_distance"] == 0
    ]
    random_perfect_match = random_df[random_df["min_edit_distance"] == 0]

    return (
        in_situ_perfect_match,
        random_perfect_match,
        rv35_library,
    )


def plot_matches_to_library(
    in_situ_perfect_match,
    random_perfect_match,
    rv35_library,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    line_width=0.9,
):
    # Define bin edges for consistent binning
    bin_edges = np.logspace(0, 6, num=80)

    # Extract histogram data
    data0 = in_situ_perfect_match["ham_lib_bc_counts"].values
    # data1 = unique_single_pre_no_starter["ham_lib_bc_counts"].values
    data2 = random_perfect_match["lib_bc_counts"].values

    # Compute histograms
    hist0, _ = np.histogram(data0, bins=bin_edges)
    # hist1, _ = np.histogram(data1, bins=bin_edges)
    hist2, _ = np.histogram(data2, bins=bin_edges)

    # Normalize histograms (scale max to 1)
    hist0 = hist0 / np.max(hist0)
    # hist1 = hist1 / np.max(hist1)
    hist2 = hist2 / np.max(hist2)

    # Extract and normalize library sequence data
    sequences = np.flip(rv35_library["counts"])
    edge_positions = sequences.searchsorted(bin_edges)
    counts = np.zeros(len(bin_edges) - 1)

    for i in range(len(bin_edges) - 1):
        parts = edge_positions[i : i + 2]
        counts[i] = sequences[parts[0] : parts[1]].sum()

    counts = counts / np.max(counts)  # Normalize to max 1

    # Plot normalized histograms as step-line plots
    ax.step(
        bin_edges[:-1],
        hist0,
        where="post",
        color="#2ca02c",
        linestyle="-",
        linewidth=line_width,
        label="All barcode sequences",
    )

    ax.step(
        bin_edges[:-1],
        hist2,
        where="post",
        color="#1f77b4",
        linestyle="-",
        linewidth=line_width,
        label="Randomly generated",
    )

    # Plot normalized library sequence data as a dashed black line
    ax.step(
        bin_edges[:-1],
        counts,
        where="post",
        color="black",
        linestyle="--",
        linewidth=line_width,
        label="Library sequences",
    )

    # X-axis log scale
    ax.set_xscale("log")

    # X-axis formatting
    ax.set_xlabel(
        "Library barcode abundance",
        fontsize=label_fontsize,
    )
    ax.xaxis.set_major_locator(mticker.FixedLocator(locs=np.logspace(0, 6, 5)))
    ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

    # Y-axis label
    ax.set_ylabel(
        "Normalized Frequency",
        fontsize=label_fontsize,
    )

    # Y-axis scale (0 to 1 since everything is normalized)
    ax.set_xlim(1, 1e6)
    ax.set_ylim(0, 1.05)

    ax.legend(
        loc="upper right",
        fontsize=tick_fontsize,
        bbox_to_anchor=(
            1.2,
            1,
        ),
    )

    # Despine function
    def despine(ax):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    despine(ax)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    return ax
