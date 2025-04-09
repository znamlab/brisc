from cricksaw_analysis import atlas_utils
from brainglobe_atlasapi import BrainGlobeAtlas
import pandas as pd


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


def find_singleton_bcs(cells_df):
    """
    Find singleton barcodes that appear exactly once across all starter cells
    and defines 'unique_barcodes' for all cells that is the intersection of its
    barcodes with singletons

    Args:
        cells (pd.DataFrame): DataFrame of cells

    Returns:
        cells (pd.DataFrame): DataFrame of cells with 'unique_barcodes' column
    """
    # Identify 'singleton' barcodes among starters
    starter_barcodes_counts = (
        cells_df[cells_df["is_starter"] == True]["all_barcodes"]
        .explode()
        .value_counts()
    )
    # Identify barcodes that are unique to a single starter cell
    singletons = set(starter_barcodes_counts[starter_barcodes_counts == 1].index)
    # For each cell, define 'unique_barcodes' = intersection of its barcodes with singletons
    cells_df["unique_barcodes"] = cells_df["all_barcodes"].apply(
        lambda x: singletons.intersection(x)
    )

    return cells_df


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
        if area_acronym == "outside":
            return "outside"
        else:
            return "Unknown"


def load_cell_barcode_data(
    processed_path,
    areas_to_empty=["fiber tracts", "outside"],
    valid_areas=["Isocortex", "TH"],
    distance_threshold=150,
):
    """
    Load the cell barcode data and assign areas and layers to each cell,
    changing cells labeled as being in white matter to the nearest valid area label.

    Args:
        processed_path (Path): Path to the processed data directory
        valid_areas (list): List of valid areas to move cells to
        distance_threshold (int): Maximum distance to move cells out of fiber tracts

    Returns:
        cells_df (pd.DataFrame): DataFrame of barcoded cell data with areas and layers assigned
    """
    # Load dataframe
    cells_df = pd.read_pickle(processed_path)
    cells_df = cells_df[cells_df["all_barcodes"].notna()]

    # Create unique barcode column containing barcodes found in only 1 starter
    cells_df = find_singleton_bcs(cells_df)
    cells_df = cells_df[cells_df["unique_barcodes"].apply(lambda x: len(x) > 0)]

    # Define special-case parameters
    chamber = "chamber_09"
    roi = 2
    special_query = f"chamber == '{chamber}' and roi == {roi}"
    cells_df["was_outside"] = cells_df["area_acronym"] == "outside"
    # Process the full DataFrame with default areas_to_empty
    full_pts = cells_df[["ara_x", "ara_y", "ara_z"]].values * 1000
    full_moved = atlas_utils.move_out_of_area(
        pts=full_pts,
        atlas=bg_atlas,
        areas_to_empty=areas_to_empty,
        valid_areas=valid_areas,
        distance_threshold=distance_threshold,
        verbose=True,
    )

    # Apply movement info to the full dataframe
    cells_df["was_in_wm"] = False
    full_actually_moved = full_moved.query("moved == True").copy()
    full_moved_indices = cells_df.iloc[full_actually_moved.pts_index].index
    not_outside_mask = cells_df.loc[full_moved_indices, "area_acronym"] != "outside"
    wm_indices = full_moved_indices[not_outside_mask]
    cells_df.loc[wm_indices, "was_in_wm"] = True
    cells_df.loc[
        full_moved_indices, "area_acronym"
    ] = full_actually_moved.new_area_acronym.values
    cells_df.loc[full_moved_indices, "area_id"] = full_actually_moved.new_area_id.values

    # Apply custom behavior for specific chamber and roi
    special_cells = cells_df.query(special_query).copy()
    if not special_cells.empty:
        special_pts = special_cells[["ara_x", "ara_y", "ara_z"]].values * 1000
        special_moved = atlas_utils.move_out_of_area(
            pts=special_pts,
            atlas=bg_atlas,
            areas_to_empty="fiber tracts",
            valid_areas=valid_areas,
            distance_threshold=distance_threshold,
            verbose=True,
        )

        special_actually_moved = special_moved.query("moved == True").copy()
        special_moved_indices = special_cells.iloc[
            special_actually_moved.pts_index
        ].index
        not_outside_mask = (
            cells_df.loc[special_moved_indices, "area_acronym"] != "outside"
        )
        wm_indices = special_moved_indices[not_outside_mask]
        cells_df.loc[wm_indices, "was_in_wm"] = True
        cells_df.loc[
            special_moved_indices, "area_acronym"
        ] = special_actually_moved.new_area_acronym.values
        cells_df.loc[
            special_moved_indices, "area_id"
        ] = special_actually_moved.new_area_id.values

    # Assign areas and layers to each cell
    cells_df["area_acronym_ancestor_rank1"] = cells_df["area_acronym"].apply(
        get_ancestor_rank1
    )

    cells_df["cortical_area"] = (
        cells_df["area_acronym"].map(BRAIN_AREA_MAPPING).astype("category")
    )
    cells_df["cortical_layer"] = (
        cells_df["area_acronym"].map(BRAIN_LAYER_MAPPING).astype("category")
    )

    return cells_df
