import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brainglobe_atlasapi import BrainGlobeAtlas
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from functools import partial

from brisc.manuscript_analysis import barcodes_in_cells as bc_cells
from cricksaw_analysis import atlas_utils

bg_atlas = BrainGlobeAtlas("allen_mouse_10um", check_latest=False)

BRAIN_AREA_MAPPING = {}
BRAIN_LAYER_MAPPING = {}

cortical_areas = [
    "AUDp",
    "AUDpo",
    "AUDv",  # Auditory
    "RSPagl",
    "RSPd",
    "RSPv",  # Retrosplenial (note they all map to "RSP")
    "VISal",
    "VISl",
    "VISli",
    "VISp",
    "VISpm",
    "VISpor",  # Visual
    "TEa",
    "ECT",
    "PERI",  # Other
]

# Map all of RSP to one labeled area
cortical_area_mapping = {
    "RSPagl": "RSP",
    "RSPd": "RSP",
    "RSPv": "RSP",
}

layers = ["1", "2/3", "4", "5", "6a", "6b"]

for area in cortical_areas:
    mapped_label = cortical_area_mapping.get(area, area)
    for layer in layers:
        BRAIN_AREA_MAPPING[f"{area}{layer}"] = mapped_label
        BRAIN_LAYER_MAPPING[f"{area}{layer}"] = f"L{layer}"

# Hippocampal-related areas:
hippocampal_areas = [
    "HPF",
    "SUB",
    "POST",
    "ProS",
    "CA1",
    "CA2",
    "CA3",
    "DG-mo",
    "DG-sg",
    "DG-po",
]
for area in hippocampal_areas:
    BRAIN_AREA_MAPPING[area] = "hippocampal"
    BRAIN_LAYER_MAPPING[area] = "hippocampal"

# Thalamic areas:
th_areas = [
    "IGL",
    "LGd-ip",
    "LGd-sh",
    "LGd-co",
    "MG",
    "MGd",
    "MGv",
    "MGm",
    "LAT",
    "IntG",
    "LGd",
    "LGv",
    "VENT",
    "PP",
    "PIL",
    "VPM",
    "VPMpc",
    "VM",
    "POL",
    "SGN",
    "LP",
    "PoT",
    "Eth",
    "TH",
]
for area in th_areas:
    BRAIN_AREA_MAPPING[area] = "TH"
    BRAIN_LAYER_MAPPING[area] = "TH"

# Fiber tracts:
fiber_tracts = [
    "fiber tracts",
    "mlf",
    "optic",
    "ar",
    "fr",
    "ml",
    "rust",
    "cpd",
    "cc",
    "lfbst",
    "cing",
    "hc",
    "scwm",
    "fp",
    "dhc",
    "alv",
    "or",
    "bsc",
    "pc",
    "ec",
    "opt",
    "vtd",
    "mtg",
    "EW",
    "bic",
    "amc",
    "act",
    "st",
    "apd",
    "lab",
    "df",
    "fi",
    "fxpo",
    "mct",
    "fx",
    "vhc",
    "stc",
    "mfbc",
]
for area in fiber_tracts:
    BRAIN_AREA_MAPPING[area] = "fiber_tract"
    BRAIN_LAYER_MAPPING[area] = "fiber_tract"

# Non-cortical areas
non_cortical_areas = [
    "SCzo",
    "SCop",
    "MB",
    "NPC",
    "MPT",
    "PPT",
    "NOT",
    "OP",
    "APN",
    "PAG",
    "SCO",
    "RPF",
    "AQ",
    "MRN",
    "FF",
    "HY",
    "ZI",
    "SCig",
    "SCsg",
    "EPd",
    "SCiw",
    "ND",
    "INC",
    "LT",
    "SNr",
    "SNc",
    "VTA",
    "SCdg",
    "SCdw",
    "csc",
    "RN",
    "MA3",
    "DT",
    "MT",
    "ENTl3",
    "ENTl2",
    "ENTl1",
    "SPFp",
]
for area in non_cortical_areas:
    BRAIN_AREA_MAPPING[area] = "non_cortical"
    BRAIN_LAYER_MAPPING[area] = "non_cortical"


def get_ancestor_rank1(area_acronym):
    """
    Determine the rank 1 ancestor of a given area acronym.

    Args:
        area_acronym (str): Area acronym to find the rank 1 ancestor

    Returns:
        rank1_ancestor (str): Rank 1 ancestor of the given area acronym
    """
    try:
        ancestors = bg_atlas.get_structure_ancestors(area_acronym)
        if "TH" in ancestors:
            return "TH"
        elif "RSP" in ancestors:
            return "RSP"
        elif "TEa" in ancestors:
            return "TEa"
        elif "AUD" in ancestors:
            return "AUD"
        elif "VISp" in ancestors:
            return area_acronym
        elif "VIS" in ancestors:
            return ancestors[-1]
        else:
            return ancestors[1] if len(ancestors) > 1 else "Unknown"
    except KeyError:
        return "Unknown"


def load_cell_barcode_data(
    processed_path,
    area_to_empty="fiber tracts",
    valid_areas=["Isocortex", "TH"],
    distance_threshold=150,
):
    """
    Load the cell barcode data and assign areas and layers to each cell,
    changing cells labeled as being in white matter to the nearest valid area label.

    Args:
        processed_path (Path): Path to the processed data directory
        area_to_empty (str): Area to empty of cells
        valid_areas (list): List of valid areas to move cells to
        distance_threshold (int): Maximum distance to move cells out of fiber tracts

    Returns:
        cell_barcode_df (pd.DataFrame): DataFrame of barcoded cell data with areas and layers assigned
    """
    # Load dataframe
    cell_barcode_df = pd.read_pickle(
        processed_path / "analysis" / "cell_barcode_df.pkl"
    )
    cell_barcode_df = cell_barcode_df[cell_barcode_df["all_barcodes"].notna()]

    # Create unique barcode column containing barcodes found in only 1 starter
    cell_barcode_df = bc_cells.find_singleton_bcs(cell_barcode_df)
    cell_barcode_df = cell_barcode_df[
        cell_barcode_df["unique_barcodes"].apply(lambda x: len(x) > 0)
    ]

    # Move cells out of fiber tracts
    pts = cell_barcode_df[["ara_x", "ara_y", "ara_z"]].values * 1000
    moved = atlas_utils.move_out_of_area(
        pts=pts,
        atlas=bg_atlas,
        areas_to_empty=area_to_empty,
        valid_areas=valid_areas,
        distance_threshold=distance_threshold,
        verbose=True,
    )
    cell_barcode_df["was_in_wm"] = False
    actually_moved = moved.query("moved==True").copy()
    cell_moved = cell_barcode_df.iloc[actually_moved.pts_index].index
    cell_barcode_df.loc[cell_moved, "was_in_wm"] = True
    cell_barcode_df.loc[
        cell_moved, "area_acronym"
    ] = actually_moved.new_area_acronym.values
    cell_barcode_df.loc[cell_moved, "area_id"] = actually_moved.new_area_id.values

    # Assign areas and layers to each cell
    cell_barcode_df["area_acronym_ancestor_rank1"] = cell_barcode_df[
        "area_acronym"
    ].apply(get_ancestor_rank1)

    cell_barcode_df["cortical_area"] = (
        cell_barcode_df["area_acronym"].map(BRAIN_AREA_MAPPING).astype("category")
    )
    cell_barcode_df["cortical_layer"] = (
        cell_barcode_df["area_acronym"].map(BRAIN_LAYER_MAPPING).astype("category")
    )

    return cell_barcode_df


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
        counts_by_area = shared_presyn.groupby(presyn_grouping).size()
        counts_by_area = counts_by_area.reindex(presyn_categories, fill_value=0)
        fraction_by_area = counts_by_area / counts_by_area.sum()

    return fraction_by_area, counts_by_area


def compute_connectivity_matrix(
    cell_barcode_df,
    starter_grouping="area_acronym_ancestor_rank1",
    presyn_grouping="area_acronym_ancestor_rank1",
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
    # 1) Separate starters and presynaptic cells
    starter_cells = cell_barcode_df[cell_barcode_df["is_starter"]].copy()
    presyn_cells = cell_barcode_df[cell_barcode_df["is_starter"] == False].copy()

    # 2) Apply the function to each starter cell to get a table of fractions
    fractions_counts_dfs = starter_cells.apply(
        compute_input_fractions, axis=1, args=(presyn_cells, presyn_grouping)
    )
    fractions_df = pd.DataFrame(
        [t[0] for t in fractions_counts_dfs], index=starter_cells.index
    )
    counts_df = pd.DataFrame(
        [t[1] for t in fractions_counts_dfs], index=starter_cells.index
    )

    # 3) Add a column for a property to group the starter on
    fractions_df[starter_grouping] = starter_cells[starter_grouping].values
    counts_df[starter_grouping] = starter_cells[starter_grouping].values
    # remove rows that sum to 0 (starters with no presynaptic cells)
    fractions_df = fractions_df.loc[
        fractions_df.select_dtypes(include="number").sum(axis=1) > 0
    ]

    # Grouping by starter property, find the mean fraction of each presynaptic grouping
    mean_input_frac_df = fractions_df.groupby(starter_grouping).mean().T
    counts_df = counts_df.groupby(starter_grouping).sum().T

    return counts_df, mean_input_frac_df, fractions_df


def filter_matrix(
    matrix,
    presyn_groups_of_interest=[
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
    starter_groups_of_interest=[
        "VISp1",
        "VISp2/3",
        "VISp4",
        "VISp5",
        "VISp6a",
        "VISp6b",
        "VISal",
        "VISl",
        "VISpm",
    ],
):
    """
    Filter the confusion matrix to include only areas of interest and remove rows and columns with all zeros.

    Args:
        matrix (pd.DataFrame): DataFrame of confusion matrix data

    Returns:
        filtered_confusion_matrix (pd.DataFrame): Filtered confusion matrix
    """
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    # Remove rows and columns with all zeros
    filtered_confusion_matrix = matrix.loc[(matrix != 0).any(axis=1)]
    filtered_confusion_matrix = filtered_confusion_matrix.loc[
        :, (filtered_confusion_matrix != 0).any(axis=0)
    ]

    # Filter to include only areas of interest
    filtered_confusion_matrix = filtered_confusion_matrix.reindex(
        index=presyn_groups_of_interest,
        columns=starter_groups_of_interest,
        fill_value=0,
    )

    filtered_confusion_matrix = filtered_confusion_matrix.loc[
        presyn_groups_of_interest, starter_groups_of_interest
    ]

    return filtered_confusion_matrix


def plot_area_by_area_connectivity(
    average_df,
    counts_df,
    fractions_df,
    input_fraction=False,
    sum_fraction=False,
    ax=None,
):
    if input_fraction:
        # Sort the confusion matrix by index and columns alphabetically
        filtered_confusion_matrix = filter_matrix(average_df)
        counts_matrix = filter_matrix(counts_df)
        if sum_fraction:
            filtered_confusion_matrix = filter_matrix(counts_df)
            filtered_confusion_matrix = (
                filtered_confusion_matrix / filtered_confusion_matrix.sum(axis=0)
            )
        # make a mask that hides nothing
        mask = pd.DataFrame(
            False,
            index=filtered_confusion_matrix.index,
            columns=filtered_confusion_matrix.columns,
        )
        vmax = 0.3
    else:
        filtered_confusion_matrix = filter_matrix(counts_df)
        # Define a mask to hide zero values
        mask = filtered_confusion_matrix == 0
        vmax = 500

    # Plot the heatmap
    sns.heatmap(
        filtered_confusion_matrix,
        cmap="magma_r",
        cbar=False,
        yticklabels=True,
        square=True,
        linewidths=1,
        linecolor="white",
        mask=mask,
        annot=False,
        vmax=vmax,
        ax=ax,
    )

    # Annotate with appropriate color based on background
    for (i, j), val in np.ndenumerate(filtered_confusion_matrix):
        if input_fraction:
            text_color = "white" if val > 0.2 else "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}" if input_fraction else f"{int(val)}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=15,
            )
        else:
            text_color = "white" if val > 300 else "black"
            if not mask.iloc[i, j]:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.2f}" if input_fraction else f"{int(val)}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=15,
                )

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Highlight the diagonal with a black outline
    for i in range(filtered_confusion_matrix.shape[1]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=3))

    # Adjust the limits of the x and y axes to avoid cutting off the outer edges
    ax.set_xlim(-0.5, filtered_confusion_matrix.shape[1] - 0.5 + 1)
    ax.set_ylim(filtered_confusion_matrix.shape[0] - 0.5 + 1, -0.5)

    # add a red vertical line at the 9th column
    ax.axvline(x=6, ymin=0.04, ymax=0.96, color="red", lw=3)
    # add a red horizontal line at the 10th row
    ax.axhline(y=6, xmin=0.05, xmax=0.95, color="red", lw=3)

    ax.add_patch(plt.Rectangle((0, 0), 8, 14, fill=False, edgecolor="black", lw=3))

    for label in ax.get_yticklabels():
        x, y = label.get_position()
        label.set_position((x + 0.025, y))
    ax.tick_params(axis="both", width=0)

    for label in ax.get_xticklabels():
        x, y = label.get_position()
        label.set_position((x, y - 0.025))

    # Add number of starter cells in each area per column on the bottom of the heatmap
    # starter_counts = starters["area_acronym_ancestor_rank1"].value_counts()
    starter_counts = fractions_df.iloc[:, -1].value_counts()
    for i, area in enumerate(filtered_confusion_matrix.columns):
        # if area in starter_counts else put 0
        ax.text(
            i + 0.5,
            filtered_confusion_matrix.shape[0] + 0.5,
            f"{starter_counts.get(area, 0)}",
            ha="center",
            va="center",
            color="black",
            fontsize=15,
        )
    # add a label saying what the sum is
    ax.text(
        filtered_confusion_matrix.shape[1] / 2,
        filtered_confusion_matrix.shape[0] + 1,
        "Total starter cells per area",
        ha="center",
        va="center",
        color="black",
        fontsize=20,
    )

    # Add number of non-starter cells in each area per row on the right of the heatmap
    if not input_fraction:
        non_starter_counts = filtered_confusion_matrix.sum(axis=1)
    else:
        non_starter_counts = counts_matrix.sum(axis=1)

    for i, area in enumerate(filtered_confusion_matrix.index):
        # if area in starter_counts else put 0
        ax.text(
            filtered_confusion_matrix.shape[1] + 0.5,
            i + 0.5,
            f"{non_starter_counts.get(area, 0)}",
            ha="center",
            va="center",
            color="black",
            fontsize=15,
        )
    # add a label saying what the sum is
    ax.text(
        filtered_confusion_matrix.shape[1] + 1,
        filtered_confusion_matrix.shape[0] / 2,
        "Total presynaptic cells per area",
        ha="center",
        va="center",
        color="black",
        fontsize=20,
        rotation=270,
    )

    ax.set_xlabel("Starter cell location", fontsize=20, labelpad=20)
    ax.set_ylabel("Presynaptic cell location", fontsize=20, labelpad=20)

    # set y tick size
    ax.tick_params(axis="y", labelsize=15)
    # set x tick size
    ax.tick_params(axis="x", labelsize=15)


def make_minimal_df(
    cell_barcode_df,
    starter_areas_to_keep=None,
    starter_cell_types_to_keep=None,
    presyn_areas_to_keep=None,
    presyn_cell_types_to_keep=None,
):
    """
    Filter the ARA starters DataFrame to only include the necessary columns and rows.

    Args:
        cell_barcode_df (pd.DataFrame): DataFrame of ARA starters data
        starter_areas_to_keep (list): List of starter areas to include
        starter_cell_types_to_keep (list): List of starter cell types to include
        presyn_areas_to_keep (list): List of presynaptic areas to include
        presyn_cell_types_to_keep (list): List of presynaptic cell types to include

    Returns:
        starters (pd.DataFrame): DataFrame of starter cells with columns:
            - barcode
            - starter_area
            - starter_cell_type
        non_starters (pd.DataFrame): DataFrame of non-starter cells with columns:
            - barcode
            - presyn_area
            - presyn_cell_type
    """

    # Separate out starter vs. non-starter rows
    starters = cell_barcode_df[cell_barcode_df["is_starter"] == True].copy()
    non_starters = cell_barcode_df[cell_barcode_df["is_starter"] == False].copy()

    starters = starters[
        ["unique_barcodes", "area_acronym_ancestor_rank1", "Annotated_clusters"]
    ]
    starters = starters.rename(
        columns={
            "unique_barcodes": "barcode",
            "area_acronym_ancestor_rank1": "starter_area",
            "Annotated_clusters": "starter_cell_type",
        }
    )

    non_starters = non_starters[
        ["unique_barcodes", "area_acronym_ancestor_rank1", "Annotated_clusters"]
    ]
    non_starters = non_starters.rename(
        columns={
            "unique_barcodes": "barcode",
            "area_acronym_ancestor_rank1": "presyn_area",
            "Annotated_clusters": "presyn_cell_type",
        }
    )

    # Keep only certain starter-area rows

    if starter_areas_to_keep:
        starters = starters[starters["starter_area"].isin(starter_areas_to_keep)]
    if starter_cell_types_to_keep:
        starters = starters[
            starters["starter_cell_type"].isin(starter_cell_types_to_keep)
        ]
    if presyn_areas_to_keep:
        non_starters = non_starters[
            non_starters["presyn_area"].isin(presyn_areas_to_keep)
        ]
    if presyn_cell_types_to_keep:
        non_starters = non_starters[
            non_starters["presyn_cell_type"].isin(presyn_cell_types_to_keep)
        ]

    # Filter to only include cells with single starter barcodes
    starters = starters[starters["barcode"].notna()]
    non_starters = non_starters[non_starters["barcode"].notna()]

    return starters, non_starters


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


def shuffle_and_compute_connectivity(
    minimal_cell_barcode_df,
    n_permutations=10000,
    shuffle_starters=False,
    shuffle_presyn=True,
    starter_grouping="area_acronym_ancestor_rank1",
    presyn_grouping="area_acronym_ancestor_rank1",
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
    observed_confusion_matrix, _, _ = compute_connectivity_matrix(
        minimal_cell_barcode_df,
        starter_grouping,
        presyn_grouping,
    )
    args = [
        (seed, minimal_cell_barcode_df, shuffle_presyn, shuffle_starters)
        for seed in range(n_permutations)
    ]
    shuffled_cell_barcode_dfs = process_map(
        shuffle_wrapper, args, max_workers=cpu_count()
    )

    # Create a partial function that fixes the extra arguments
    partial_func = partial(
        compute_connectivity_matrix,
        starter_grouping=starter_grouping,
        presyn_grouping=presyn_grouping,
    )

    # Run it in parallel
    results = process_map(
        partial_func,
        shuffled_cell_barcode_dfs,
        max_workers=cpu_count(),
        desc="Computing connectivity",
    )

    shuffled_matrices, mean_input_fractions, starter_input_fractions = zip(*results)

    return (
        observed_confusion_matrix,
        shuffled_cell_barcode_dfs,
        shuffled_matrices,
        mean_input_fractions,
        starter_input_fractions,
    )


def plot_null_histograms_square(
    observed_cm,
    all_null_matrices,
    bins=30,
    row_label_fontsize=14,
    col_label_fontsize=14,
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
    """
    Plot a grid of square histogram subplots, one for each cell in the observed confusion matrix,
    with optional custom ordering/filtering of rows (presynaptic areas) and columns (starter areas).

    Parameters
    ----------
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
    n_rows, n_cols = subset_observed_cm.shape

    # Subset the null distributions
    null_array = np.array(all_null_matrices)  # shape: (N, orig_n_rows, orig_n_cols)

    # find the integer row/col indices that correspond to row_order/col_order
    row_indices = [observed_cm.index.get_loc(r) for r in row_order]
    col_indices = [observed_cm.columns.get_loc(c) for c in col_order]

    # Reindex the null distribution:
    #   first index permutations (:), then row_indices, then col_indices
    # null_array[:, row_indices, col_indices] => shape: (N, n_rows, n_cols)
    subset_null_array = null_array[:, row_indices][:, :, col_indices]

    # Create the figure and axes grid
    # Decide figure size so each subplot can be roughly square
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
    for i, row_label in enumerate(row_order):
        for j, col_label in enumerate(col_order):
            ax = get_ax(i, j)

            # Extract the null distribution for this subset cell
            cell_values = subset_null_array[:, i, j]

            # Observed value
            observed_val = subset_observed_cm.iloc[i, j]

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

    return subset_null_array, subset_observed_cm


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
    pvals_bh_idx = sort_idx
    pvals_bh[not_nan_mask][pvals_bh_idx] = pvals_bh_sorted

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

    # Make x and y area labels bold
    # ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
    # ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')

    plt.title(
        "Log Ratio of Observed vs. Shuffled Null with P-values",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Starter area", fontsize=14, fontweight="bold")
    plt.ylabel("Presynaptic area", fontsize=14, fontweight="bold")

    plt.show()


def bubble_plot(log_ratio_matrix, pval_df, size_scale=800):
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

    fig, ax = plt.subplots(figsize=(8, 10))

    # Main scatter
    sc = ax.scatter(
        x=df_plot["x"],
        y=df_plot["y"],
        s=df_plot["bubble_size"],
        c=df_plot["color_value"],
        cmap="coolwarm",
        vmin=df_plot["color_value"].min(),
        vmax=df_plot["color_value"].max(),
        edgecolors="none",
    )

    # Add black outlines for significant cells
    significant_cells = pval_df < 0.05
    is_signif = significant_cells.stack().values
    df_signif = df_plot[is_signif]
    ax.scatter(
        x=df_signif["x"],
        y=df_signif["y"],
        s=df_signif["bubble_size"],
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Sign(log₁₀(ratio)) × -log₁₀(p-value)", fontsize=12)

    # Bubble-size legend
    legend_values = [0.1, 0.3, 0.6]
    legend_handles = []
    for val in legend_values:
        size_val = val * size_scale
        h = ax.scatter(
            [], [], s=size_val, c="gray", alpha=0.5, label=f"|log₁₀(ratio)| = {val}"
        )
        legend_handles.append(h)
    ax.legend(
        handles=legend_handles,
        title="Bubble Size Legend",
        loc="upper left",
        bbox_to_anchor=(1.05, -0.05),
        borderaxespad=0.0,
        frameon=True,
        handleheight=4.0,
    )

    ax.set_xticks(range(len(x_categories)))
    ax.set_yticks(range(len(y_categories)))
    ax.set_xticklabels(x_categories, rotation=90)
    ax.set_yticklabels(y_categories)
    # Invert y-axis so top row is y=0
    ax.invert_yaxis()
    plt.xlabel("Starter area", fontsize=13, fontweight="bold")
    plt.ylabel("Presynaptic area", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
