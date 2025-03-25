import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brainglobe_atlasapi import BrainGlobeAtlas
import iss_preprocess as iss
import scanpy as sc
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

bg_atlas = BrainGlobeAtlas("allen_mouse_10um", check_latest=False)


def get_ancestor_rank1(area_acronym):
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
    data_path="becalia_rabies_barseq/BRAC8498.3e/chamber_07",
    filter_by_presynaptics=False,
    filter_shared_bc_starters=True,
    filter_by_annotation=False,
    use_centroid_cluster_mapping=False,
):
    processed_path = iss.io.get_processed_path(data_path)

    # Load data from mCherry curated cells and barcodes with ed2, minimum match ?, ed correction weighting only first 10 bases
    ara_starters = pd.read_pickle(
        processed_path.parent / "analysis" / "cell_barcode_df.pkl"
    )
    ara_starters = ara_starters[ara_starters["all_barcodes"].notna()]
    print("Before filtering:")
    print(f"Number of barcoded cells: {ara_starters.shape[0]}")
    print(
        f"Number of barcodes (unique among all cells): "
        f"{ara_starters.explode('all_barcodes')['all_barcodes'].nunique()}"
    )
    print(
        f"Number of presynaptic cells: {ara_starters[ara_starters['is_starter'] == False].shape[0]}"
    )
    print(
        f"Number of starter cells: {ara_starters[ara_starters['is_starter'] == True].shape[0]}\n"
    )

    # Step 1: Remove starters with shared barcodes
    ara_is_starters = pd.read_pickle(
        processed_path.parent / "analysis" / "cell_barcode_df.pkl"
    )
    ara_is_starters = ara_is_starters[ara_is_starters["all_barcodes"].notna()]

    # Assuming ara_is_starters is your dataframe
    def shorten_barcodes(barcodes):
        return [barcode[:10] for barcode in barcodes]

    ara_is_starters["all_barcodes"] = ara_is_starters["all_barcodes"].apply(
        shorten_barcodes
    )

    if filter_shared_bc_starters:
        # Flatten all barcodes from starter cells to count their occurrences
        starter_barcodes_counts = (
            ara_is_starters[ara_is_starters["is_starter"] == True]["all_barcodes"]
            .explode()
            .value_counts()
        )

        # Identify barcodes that are unique to a single starter cell
        unique_starter_barcodes = starter_barcodes_counts[
            starter_barcodes_counts == 1
        ].index

        # Filter starter cells where all their barcodes are unique among starter cells
        starter_cells_with_unique_barcodes = ara_is_starters[
            (ara_is_starters["is_starter"] == True)
            & (
                ara_is_starters["all_barcodes"].apply(
                    lambda barcodes: all(b in unique_starter_barcodes for b in barcodes)
                )
            )
        ]

        # Filter presynaptic cells that contain at least one of these barcodes
        presynaptic_cells_with_shared_barcodes = ara_is_starters[
            (ara_is_starters["is_starter"] == False)
            & (
                ara_is_starters["all_barcodes"].apply(
                    lambda barcodes: any(b in unique_starter_barcodes for b in barcodes)
                )
            )
        ]
        # Combine the filtered starter cells and the filtered presynaptic cells
        ara_starters = pd.concat(
            [starter_cells_with_unique_barcodes, presynaptic_cells_with_shared_barcodes]
        )
    ara_starters = ara_starters.rename(columns={"is_starter": "starter"})

    print("After shared starter cell count filtering:")
    print(f"Number of barcoded cells: {ara_starters.shape[0]}")
    print(
        f"Number of barcodes (unique among all cells): "
        f"{ara_starters.explode('all_barcodes')['all_barcodes'].nunique()}"
    )
    print(
        f"Number of presynaptic cells: {ara_starters[ara_starters['starter'] == False].shape[0]}"
    )
    print(
        f"Number of starter cells: {ara_starters[ara_starters['starter'] == True].shape[0]}\n"
    )

    # Step 2: Identify barcodes found in more than 4 non-starter cells
    if filter_by_presynaptics:
        if False:
            non_starter_barcodes_counts = (
                ara_starters[ara_starters["starter"] == False]["all_barcodes"]
                .explode()
                .value_counts()
            )
            barcodes_to_keep = non_starter_barcodes_counts[
                non_starter_barcodes_counts > 4
            ].index.values
            ara_starters = ara_starters[
                ara_starters["all_barcodes"].apply(
                    lambda x: any([b in barcodes_to_keep for b in x])
                )
            ]

        # Step 1: Identify barcodes found in more than 4 non-starter cells
        non_starter_barcodes_counts = (
            ara_starters[ara_starters["starter"] == False]["all_barcodes"]
            .explode()
            .value_counts()
        )
        barcodes_to_keep = non_starter_barcodes_counts[
            non_starter_barcodes_counts > 4
        ].index.values

        # Step 2: Remove barcodes that aren't present in 5 or more non-starter cells from all_barcodes
        def filter_barcodes_and_update_counts(row):
            filtered_barcodes = []
            filtered_counts = []
            total_removed = 0

            for barcode, count in zip(row["all_barcodes"], row["n_spots_per_barcode"]):
                if barcode in barcodes_to_keep:
                    filtered_barcodes.append(barcode)
                    filtered_counts.append(count)
                else:
                    total_removed += count

            row["all_barcodes"] = filtered_barcodes
            row["n_spots_per_barcode"] = filtered_counts
            row["total_n_spots"] -= total_removed

            return row

        ara_starters = ara_starters.apply(filter_barcodes_and_update_counts, axis=1)
        ara_starters = ara_starters[ara_starters["total_n_spots"] > 0]
        print("After presynaptic cell count filtering:")
        print(f"Number of barcoded cells: {ara_starters.shape[0]}")
        print(
            f"Number of barcodes (unique among all cells): "
            f"{ara_starters.explode('all_barcodes')['all_barcodes'].nunique()}"
        )
        print(
            f"Number of presynaptic cells: {ara_starters[ara_starters['starter'] == False].shape[0]}"
        )
        print(
            f"Number of starter cells: {ara_starters[ara_starters['starter'] == True].shape[0]}\n"
        )

    # Step 2: Add cell type annotations and cell correlations to cluster centroids
    adata = sc.read_h5ad(processed_path.parent / "analysis" / "adata_annotated.h5ad")
    ara_starters["Annotated_clusters"] = adata.obs.Annotated_clusters
    ara_starters["gene_total_counts"] = adata.obs.total_counts
    ara_starters["n_genes"] = adata.obs.n_genes_by_counts
    correlation_cells = pd.read_csv(
        "/nemo/project/proj-znamenp-barseq/processed/becalia_rabies_barseq/BRAC8498.3e/analysis/correlation_scores_all_barcoded.csv",
        index_col=0,
    )
    ara_starters = ara_starters.join(correlation_cells)

    if use_centroid_cluster_mapping:
        # Step 3: See how many cells would be left if we immediately filter to cells that already have an annotated cluster
        ara_over25_genes = ara_starters.dropna(subset=["Annotated_clusters"])
        print("After annotated cluster filtering:")
        print(f"Number of barcoded cells: {ara_over25_genes.shape[0]}")
        print(f"Number of barcodes: {ara_over25_genes['main_barcode'].nunique()}")
        print(
            f"Number of presynaptic cells: {ara_over25_genes[ara_over25_genes['starter'] == False].shape[0]}"
        )
        print(
            f"Number of starter cells: {ara_over25_genes[ara_over25_genes['starter'] == True].shape[0]}\n"
        )

        # Filter to just cells with a high correlation to a cell type cluster centroid
        ara_starters.dropna(subset=["best_cluster"], inplace=True)
        ara_starters = ara_starters[ara_starters.best_cluster != "Zero_correlation"]
        ara_starters = ara_starters[ara_starters.best_score > 0.2]
        ara_starters["Clusters"] = ara_starters["Annotated_clusters"].fillna(
            ara_starters["best_cluster"]
        )
        print("After cell type cluster centroid filtering:")
        print(f"Number of barcoded cells: {ara_starters.shape[0]}")
        print(
            f"Number of barcodes (unique among all cells): "
            f"{ara_starters.explode('all_barcodes')['all_barcodes'].nunique()}"
        )
        print(
            f"Number of presynaptic cells: {ara_starters[ara_starters['starter'] == False].shape[0]}"
        )
        print(
            f"Number of starter cells: {ara_starters[ara_starters['starter'] == True].shape[0]}\n"
        )

    else:
        if filter_by_annotation:
            ara_starters = ara_starters.dropna(subset=["Annotated_clusters"])
            print("After annotated cluster filtering:")
            print(f"Number of barcoded cells: {ara_starters.shape[0]}")
            print(
                f"Number of barcodes (unique among all cells): "
                f"{ara_starters.explode('all_barcodes')['all_barcodes'].nunique()}"
            )
            print(
                f"Number of presynaptic cells: {ara_starters[ara_starters['starter'] == False].shape[0]}"
            )
            print(
                f"Number of starter cells: {ara_starters[ara_starters['starter'] == True].shape[0]}\n"
            )

    ara_starters["area_acronym_ancestor_rank1"] = ara_starters["area_acronym"].apply(
        get_ancestor_rank1
    )

    cortical_areas = {
        #'outside': "outside",
        #'root': "outside",
        ### Auditory primary
        "AUDp1": "AUDp",
        "AUDp2/3": "AUDp",
        "AUDp4": "AUDp",
        "AUDp5": "AUDp",
        "AUDp6a": "AUDp",
        "AUDp6b": "AUDp",
        ### Auditory posterior
        "AUDpo1": "AUDpo",
        "AUDpo2/3": "AUDpo",
        "AUDpo4": "AUDpo",
        "AUDpo5": "AUDpo",
        "AUDpo6a": "AUDpo",
        "AUDpo6b": "AUDpo",
        ### Auditory ventral
        "AUDv1": "AUDv",
        "AUDv2/3": "AUDv",
        "AUDv4": "AUDv",
        "AUDv5": "AUDv",
        "AUDv6a": "AUDv",
        "AUDv6b": "AUDv",
        ### Retrosplenial lateral agranular
        "RSPagl1": "RSP",
        "RSPagl2/3": "RSP",
        "RSPagl5": "RSP",
        "RSPagl6a": "RSP",
        "RSPagl6b": "RSP",
        ### Retrosplenial dorsal
        "RSPd1": "RSP",
        "RSPd2/3": "RSP",
        "RSPd5": "RSP",
        "RSPd6a": "RSP",
        "RSPd6b": "RSP",
        ### Retrosplenial ventral
        "RSPv1": "RSP",
        "RSPv2/3": "RSP",
        "RSPv5": "RSP",
        "RSPv6a": "RSP",
        ### Visual antero-lateral
        "VISal1": "VISal",
        "VISal2/3": "VISal",
        "VISal4": "VISal",
        "VISal5": "VISal",
        "VISal6a": "VISal",
        "VISal6b": "VISal",
        ### Visual lateral
        "VISl1": "VISl",
        "VISl2/3": "VISl",
        "VISl4": "VISl",
        "VISl5": "VISl",
        "VISl6a": "VISl",
        "VISl6b": "VISl",
        ### Visual laterointermediate
        "VISli1": "VISli",
        "VISli2/3": "VISli",
        "VISli4": "VISli",
        "VISli5": "VISli",
        "VISli6a": "VISli",
        "VISli6b": "VISli",
        ### Visual primary
        "VISp1": "VISp",
        "VISp2/3": "VISp",
        "VISp4": "VISp",
        "VISp5": "VISp",
        "VISp6a": "VISp",
        "VISp6b": "VISp",
        ### Visual posteromedial
        "VISpm1": "VISpm",
        "VISpm2/3": "VISpm",
        "VISpm4": "VISpm",
        "VISpm5": "VISpm",
        "VISpm6a": "VISpm",
        "VISpm6b": "VISpm",
        ### TEa
        "TEa1": "TEa",
        "TEa2/3": "TEa",
        "TEa4": "TEa",
        "TEa5": "TEa",
        "TEa6a": "TEa",
        "TEa6b": "TEa",
        ### Hippocampal Areas
        "HPF": "hippocampal",
        "SUB": "hippocampal",
        "POST": "hippocampal",
        "ProS": "hippocampal",
        "CA1": "hippocampal",
        "CA2": "hippocampal",
        "CA3": "hippocampal",
        "DG-mo": "hippocampal",
        "DG-sg": "hippocampal",
        "DG-po": "hippocampal",
        ### Thalamus
        "IGL": "TH",
        "LGd-ip": "TH",
        "LGd-sh": "TH",
        "LGd-co": "TH",
        "MG": "TH",
        "MGd": "TH",
        "MGv": "TH",
        "LAT": "TH",
        "IntG": "TH",
        "LGd": "TH",
        "LGv": "TH",
        "VENT": "TH",
        "PP": "TH",
        "PIL": "TH",
        "VPM": "TH",
        "VPMpc": "TH",
        "VM": "TH",
        "POL": "TH",
        "SGN": "TH",
        "LP": "TH",
        "PoT": "TH",
        "Eth": "TH",
        "TH": "TH",
        "MGm": "TH",
        ### Fiber Tracts
        "fiber tracts": "fiber_tract",
        "mlf": "fiber_tract",
        "optic": "fiber_tract",
        "ar": "fiber_tract",
        "fr": "fiber_tract",
        "ml": "fiber_tract",
        "rust": "fiber_tract",
        "cpd": "fiber_tract",
        "cc": "fiber_tract",
        "lfbst": "fiber_tract",
        "cing": "fiber_tract",
        "hc": "fiber_tract",
        "scwm": "fiber_tract",
        "fp": "fiber_tract",
        "dhc": "fiber_tract",
        "alv": "fiber_tract",
        "or": "fiber_tract",
        "bsc": "fiber_tract",
        "pc": "fiber_tract",
        "ec": "fiber_tract",
        "opt": "fiber_tract",
        "vtd": "fiber_tract",
        "mtg": "fiber_tract",
        "EW": "fiber_tract",
        "bic": "fiber_tract",
        "amc": "fiber_tract",
        "act": "fiber_tract",
        "st": "fiber_tract",
        "apd": "fiber_tract",
        "lab": "fiber_tract",
        "df": "fiber_tract",
        "fi": "fiber_tract",
        "fxpo": "fiber_tract",
        "mct": "fiber_tract",
        "fx": "fiber_tract",
        "vhc": "fiber_tract",
        "stc": "fiber_tract",
        "mfbc": "fiber_tract",
        ### Non-cortical areas
        "SCzo": "non_cortical",
        "SCop": "non_cortical",
        "MB": "non_cortical",
        "NPC": "non_cortical",
        "MPT": "non_cortical",
        "PPT": "non_cortical",
        "NOT": "non_cortical",
        "OP": "non_cortical",
        "APN": "non_cortical",
        "PAG": "non_cortical",
        "SCO": "non_cortical",
        "RPF": "non_cortical",
        "AQ": "non_cortical",
        "MRN": "non_cortical",
        "FF": "non_cortical",
        "HY": "non_cortical",
        "ZI": "non_cortical",
        "SCig": "non_cortical",
        "SCsg": "non_cortical",
        "EPd": "non_cortical",
        "SCiw": "non_cortical",
        "ND": "non_cortical",
        "INC": "non_cortical",
        "LT": "non_cortical",
        "SNr": "non_cortical",
        "SNc": "non_cortical",
        "VTA": "non_cortical",
        "SCdg": "non_cortical",
        "SCdw": "non_cortical",
        "csc": "non_cortical",
        "RN": "non_cortical",
        "MA3": "non_cortical",
        "DT": "non_cortical",
        "MT": "non_cortical",
        "ENTl3": "non_cortical",
        "ENTl2": "non_cortical",
        "ENTl1": "non_cortical",
        "SPFp": "non_cortical",
    }

    cortical_layers = {
        #'outside': "outside",
        #'root': "outside",
        ### Layer 1
        "RSPd1": "L1",
        "RSPv1": "L1",
        "RSPagl1": "L1",
        "VISpm1": "L1",
        "VISp1": "L1",
        "VISal1": "L1",
        "VISl1": "L1",
        "VISli1": "L1",
        "TEa1": "L1",
        "AUDpo1": "L1",
        "AUDp1": "L1",
        "AUDv1": "L1",
        "ECT1": "L1",
        "PERI1": "L1",
        ### Layer 2/3
        "RSPd2/3": "L2/3",
        "RSPv2/3": "L2/3",
        "RSPagl2/3": "L2/3",
        "VISpm2/3": "L2/3",
        "VISp2/3": "L2/3",
        "VISal2/3": "L2/3",
        "VISl2/3": "L2/3",
        "VISli2/3": "L2/3",
        "TEa2/3": "L2/3",
        "AUDpo2/3": "L2/3",
        "AUDp2/3": "L2/3",
        "AUDv2/3": "L2/3",
        "ECT2/3": "L2/3",
        "PERI2/3": "L2/3",
        ### Layer 4
        "VISpm4": "L4",
        "VISp4": "L4",
        "VISal4": "L4",
        "VISl4": "L4",
        "VISli4": "L4",
        "TEa4": "L4",
        "AUDpo4": "L4",
        "AUDp4": "L4",
        "AUDv4": "L4",
        "ECT4": "L4",
        ### Layer 5
        "RSPd5": "L5",
        "RSPv5": "L5",
        "RSPagl5": "L5",
        "VISpm5": "L5",
        "VISp5": "L5",
        "VISal5": "L5",
        "VISl5": "L5",
        "VISli5": "L5",
        "TEa5": "L5",
        "AUDpo5": "L5",
        "AUDp5": "L5",
        "AUDv5": "L5",
        "ECT5": "L5",
        "PERI5": "L5",
        "ENTl5": "L5",
        ### Layer 6a
        "RSPv6a": "L6a",
        "RSPd6a": "L6a",
        "RSPagl6a": "L6a",
        "VISpm6a": "L6a",
        "VISp6a": "L6a",
        "VISal6a": "L6a",
        "VISl6a": "L6a",
        "VISli6a": "L6a",
        "TEa6a": "L6a",
        "AUDpo6a": "L6a",
        "AUDp6a": "L6a",
        "AUDv6a": "L6a",
        "ECT6a": "L6a",
        "ENTl6a": "L6a",
        "PERI6a": "L6a",
        ### Layer 6b
        "RSPv6b": "L6b",
        "RSPd6b": "L6b",
        "RSPagl6b": "L6b",
        "VISpm6b": "L6b",
        "VISp6b": "L6b",
        "VISal6b": "L6b",
        "VISl6b": "L6b",
        "VISli6b": "L6b",
        "TEa6b": "L6b",
        "AUDpo6b": "L6b",
        "AUDp6b": "L6b",
        "AUDv6b": "L6b",
        "ECT6b": "L6b",
        "PERI6b": "L6b",
        "VISpor6b": "L6b",
        ### Hippocampal Areas
        "HPF": "hippocampal",
        "SUB": "hippocampal",
        "POST": "hippocampal",
        "ProS": "hippocampal",
        "CA1": "hippocampal",
        "CA2": "hippocampal",
        "CA3": "hippocampal",
        "DG-mo": "hippocampal",
        "DG-sg": "hippocampal",
        "DG-po": "hippocampal",
        ### Thalamus
        "IGL": "TH",
        "LGd-ip": "TH",
        "LGd-sh": "TH",
        "LGd-co": "TH",
        "MG": "TH",
        "MGd": "TH",
        "MGv": "TH",
        "LAT": "TH",
        "IntG": "TH",
        "LGd": "TH",
        "LGv": "TH",
        "VENT": "TH",
        "PP": "TH",
        "PIL": "TH",
        "VPM": "TH",
        "VPMpc": "TH",
        "VM": "TH",
        "POL": "TH",
        "SGN": "TH",
        "LP": "TH",
        "PoT": "TH",
        "Eth": "TH",
        "TH": "TH",
        "MGm": "TH",
        ### Fiber Tracts
        "fiber tracts": "fiber_tract",
        "mlf": "fiber_tract",
        "optic": "fiber_tract",
        "ar": "fiber_tract",
        "fr": "fiber_tract",
        "ml": "fiber_tract",
        "rust": "fiber_tract",
        "cpd": "fiber_tract",
        "cc": "fiber_tract",
        "lfbst": "fiber_tract",
        "cing": "fiber_tract",
        "hc": "fiber_tract",
        "scwm": "fiber_tract",
        "fp": "fiber_tract",
        "dhc": "fiber_tract",
        "alv": "fiber_tract",
        "or": "fiber_tract",
        "bsc": "fiber_tract",
        "pc": "fiber_tract",
        "ec": "fiber_tract",
        "opt": "fiber_tract",
        "vtd": "fiber_tract",
        "mtg": "fiber_tract",
        "EW": "fiber_tract",
        "bic": "fiber_tract",
        "amc": "fiber_tract",
        "act": "fiber_tract",
        "st": "fiber_tract",
        "apd": "fiber_tract",
        "lab": "fiber_tract",
        "df": "fiber_tract",
        "fi": "fiber_tract",
        "fxpo": "fiber_tract",
        "mct": "fiber_tract",
        "fx": "fiber_tract",
        "vhc": "fiber_tract",
        "stc": "fiber_tract",
        "mfbc": "fiber_tract",
        ### Non-cortical areas
        "SCzo": "non_cortical",
        "SCop": "non_cortical",
        "MB": "non_cortical",
        "NPC": "non_cortical",
        "MPT": "non_cortical",
        "PPT": "non_cortical",
        "NOT": "non_cortical",
        "OP": "non_cortical",
        "APN": "non_cortical",
        "PAG": "non_cortical",
        "SCO": "non_cortical",
        "RPF": "non_cortical",
        "AQ": "non_cortical",
        "MRN": "non_cortical",
        "FF": "non_cortical",
        "HY": "non_cortical",
        "ZI": "non_cortical",
        "SCig": "non_cortical",
        "SCsg": "non_cortical",
        "EPd": "non_cortical",
        "SCiw": "non_cortical",
        "ND": "non_cortical",
        "INC": "non_cortical",
        "LT": "non_cortical",
        "SNr": "non_cortical",
        "SNc": "non_cortical",
        "VTA": "non_cortical",
        "SCdg": "non_cortical",
        "SCdw": "non_cortical",
        "csc": "non_cortical",
        "RN": "non_cortical",
        "MA3": "non_cortical",
        "DT": "non_cortical",
        "MT": "non_cortical",
        "ENTl3": "non_cortical",
        "ENTl2": "non_cortical",
        "ENTl1": "non_cortical",
        "SPFp": "non_cortical",
    }

    ara_starters["cortical_area"] = ara_starters["area_acronym"].map(cortical_areas)
    ara_starters["cortical_layer"] = ara_starters["area_acronym"].map(cortical_layers)

    return ara_starters


def plot_raw_area_by_area_connectivity(
    ara_starters,
    ax=None,
):
    # Filtering data
    starters = ara_starters[ara_starters["starter"] == True]
    non_starters = ara_starters[ara_starters["starter"] == False]

    # Creating the confusion matrix
    confusion_matrix = pd.DataFrame(
        0,
        index=non_starters["area_acronym_ancestor_rank1"].unique(),
        columns=starters["area_acronym_ancestor_rank1"].unique(),
    )

    for _, starter_row in starters.iterrows():
        main_barcode = starter_row["main_barcode"]
        starter_area = starter_row["area_acronym_ancestor_rank1"]

        linked_non_starters = non_starters[non_starters["main_barcode"] == main_barcode]
        for _, non_starter_row in linked_non_starters.iterrows():
            non_starter_area = non_starter_row["area_acronym_ancestor_rank1"]
            confusion_matrix.loc[non_starter_area, starter_area] += 1

    # Sort the confusion matrix by index and columns alphabetically
    confusion_matrix = confusion_matrix.sort_index(axis=0).sort_index(axis=1)

    # Remove rows and columns with all zeros
    filtered_confusion_matrix = confusion_matrix.loc[
        (confusion_matrix != 0).any(axis=1)
    ]
    filtered_confusion_matrix = filtered_confusion_matrix.loc[
        :, (filtered_confusion_matrix != 0).any(axis=0)
    ]

    # Define the labels to drop
    labels_to_drop = ["Unknown", "fiber tracts", "grey", "ECT"]
    filtered_confusion_matrix = filtered_confusion_matrix.drop(
        labels=labels_to_drop, axis=0, errors="ignore"
    )
    filtered_confusion_matrix = filtered_confusion_matrix.drop(
        labels=["Unknown", "fiber tracts"], axis=1, errors="ignore"
    )
    filtered_confusion_matrix = filtered_confusion_matrix.drop(
        labels=[
            "VISli",
            "VISal",
            "AUD",
            "RSP",
            "TEa",
            "TH",
        ],
        axis=1,
        errors="ignore",
    )

    # Filter to include only areas of interest
    areas_of_interest = [
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
    ]
    filtered_confusion_matrix = filtered_confusion_matrix.reindex(
        index=areas_of_interest, columns=areas_of_interest, fill_value=0
    )

    filtered_confusion_matrix = filtered_confusion_matrix.loc[
        areas_of_interest, areas_of_interest
    ]
    filtered_confusion_matrix = filtered_confusion_matrix.drop(
        labels=[
            "VISli",
            "VISal",
            "AUD",
            "RSP",
            "TEa",
            "TH",
        ],
        axis=1,
        errors="ignore",
    )
    # Plotting the fully normalized confusion matrix using seaborn heatmap
    plt.figure(figsize=(20, 18), dpi=80)

    mask_zeroes = True
    if mask_zeroes:
        # Define a mask to hide zero values
        mask = filtered_confusion_matrix == 0
    else:
        # make a mask that hides nothing
        mask = pd.DataFrame(
            False,
            index=filtered_confusion_matrix.index,
            columns=filtered_confusion_matrix.columns,
        )

    # Plot the heatmap with zero values masked
    ax = sns.heatmap(
        filtered_confusion_matrix,
        cmap="magma_r",
        cbar=False,
        yticklabels=True,
        square=True,
        linewidths=1,
        linecolor="white",
        mask=mask,
        annot=False,
        vmax=390,
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Annotate with appropriate color based on background
    for (i, j), val in np.ndenumerate(filtered_confusion_matrix):
        if not mask.iloc[i, j]:
            text_color = "white" if val > 300 else "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=15,
            )

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
    starter_counts = starters["area_acronym_ancestor_rank1"].value_counts()
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
    non_starter_counts = filtered_confusion_matrix.sum(
        axis=1
    )  # non_starters['area_acronym_ancestor_rank1'].value_counts()
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

    plt.xlabel("Starter cell location", fontsize=20, labelpad=20)
    plt.ylabel("Presynaptic cell location", fontsize=20, labelpad=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)


def make_minimal_df(ara_starters):
    starters = ara_starters[ara_starters["starter"] == True]
    non_starters = ara_starters[ara_starters["starter"] == False]
    # make df which is only columns "main_barcode" and "area_acronym_ancestor_rank1",
    # but rename "area_acronym_ancestor_rank1" to "starter_area"
    starters = starters[["main_barcode", "area_acronym_ancestor_rank1"]]
    starters = starters.rename(columns={"area_acronym_ancestor_rank1": "starter_area"})
    non_starters = non_starters[["main_barcode", "area_acronym_ancestor_rank1"]]
    non_starters = non_starters.rename(
        columns={"area_acronym_ancestor_rank1": "presyn_area"}
    )
    starter_areas_to_keep = ["VISp5", "VISp4", "VISp2/3", "VISp6a", "VISp1", "VISp6b"]
    starters = starters[starters["starter_area"].isin(starter_areas_to_keep)]
    starters_barcodes = starters.main_barcode.unique()
    non_starters = non_starters[non_starters["main_barcode"].isin(starters_barcodes)]

    return starters, non_starters


def compute_observed_confusion_matrix(starters, non_starters):
    """
    Vectorized computation of the *observed* confusion matrix.

    We merge on 'main_barcode' (starters <-> non_starters),
    group by (presyn_area, starter_area) to count connections,
    and pivot into a DataFrame.
    """
    merged = pd.merge(
        non_starters[["presyn_area", "main_barcode"]],
        starters[["starter_area", "main_barcode"]],
        on="main_barcode",
        how="inner",
    )
    grouped = (
        merged.groupby(["presyn_area", "starter_area"]).size().reset_index(name="count")
    )
    confusion_df = grouped.pivot_table(
        index="presyn_area", columns="starter_area", values="count", fill_value=0
    )
    return confusion_df


def shuffle_cm_chunk(
    non_starters_arr, starters_arr, row_index, col_index, n_permutations, seed_offset=0
):
    """
    Perform *n_permutations* random shuffles in one process.
    Return a list of confusion-matrix arrays, each reindexed
    to match (row_index, col_index).

    Parameters
    ----------
    non_starters_arr : dict with keys {'presyn_area', 'barcodes'}
    starters_arr : dict with keys {'starter_area', 'barcodes'}
    row_index : pd.Index for the final matrix rows
    col_index : pd.Index for the final matrix columns
    n_permutations : int
    seed_offset : int, optional random seed offset
    """
    np.random.seed(seed_offset)
    results = []

    for _ in range(n_permutations):
        # Shuffle
        shuffled_barcodes = np.random.permutation(non_starters_arr["barcodes"])

        # Build a merged DataFrame with vectorized approach
        ns_df = pd.DataFrame(
            {
                "presyn_area": non_starters_arr["presyn_area"],
                "main_barcode": shuffled_barcodes,
            }
        )
        st_df = pd.DataFrame(
            {
                "starter_area": starters_arr["starter_area"],
                "main_barcode": starters_arr["barcodes"],
            }
        )

        merged = pd.merge(ns_df, st_df, on="main_barcode", how="inner")

        # groupby -> pivot
        grouped = (
            merged.groupby(["presyn_area", "starter_area"])
            .size()
            .reset_index(name="count")
        )
        shuffle_cm = grouped.pivot_table(
            index="presyn_area", columns="starter_area", values="count", fill_value=0
        )

        # Reindex to match the final shape (row_index, col_index)
        shuffle_cm = shuffle_cm.reindex(
            index=row_index, columns=col_index, fill_value=0
        )

        # Append the numpy array version
        results.append(shuffle_cm.values)

    return results


def main_parallel_shuffling(starters, non_starters, n_permutations=10000, n_jobs=8):
    """
    Orchestrates the parallel shuffling procedure:
    1) Compute observed confusion matrix (vectorized).
    2) Prepare minimal data structures for pickling.
    3) Chunk the total permutations into fewer tasks.
    4) Parallelize with ProcessPoolExecutor.
    5) Return list of all null matrices (arrays).
    """

    # Compute observed confusion matrix
    observed_confusion_matrix = compute_observed_confusion_matrix(
        starters, non_starters
    )

    # Save row/col indices for consistent reindexing
    row_index = observed_confusion_matrix.index
    col_index = observed_confusion_matrix.columns

    # Prepare minimal data for pickling
    non_starters_arr = {
        "presyn_area": non_starters["presyn_area"].values,
        "barcodes": non_starters["main_barcode"].values,
    }
    starters_arr = {
        "starter_area": starters["starter_area"].values,
        "barcodes": starters["main_barcode"].values,
    }

    # Chunk permutations into fewer tasks
    all_null_matrices = []
    chunk_size = max(n_permutations // n_jobs, 1)
    remainder = n_permutations % n_jobs
    tasks = []
    for i in range(n_jobs):
        this_chunk = chunk_size + (1 if i < remainder else 0)
        tasks.append(this_chunk)

    # Parallel loop
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        seed_offset = 42
        for chunk in tasks:
            # Each task does `chunk` permutations
            if chunk > 0:
                future = executor.submit(
                    shuffle_cm_chunk,
                    non_starters_arr,
                    starters_arr,
                    row_index,
                    col_index,
                    chunk,
                    seed_offset,
                )
                futures.append(future)
                seed_offset += 1

        # Wrap as_completed in tqdm for progress
        for f in tqdm(as_completed(futures), total=len(futures), desc="Shuffling"):
            chunk_result = f.result()  # list of matrix arrays
            all_null_matrices.extend(chunk_result)

    # Return results
    print(f"Done! Generated {len(all_null_matrices)} shuffled matrices.")
    return observed_confusion_matrix, all_null_matrices


def run_connectivity_parallel_shuffling(starters, non_starters):
    n_permutations = 100000
    n_jobs = 250
    observed_cm, all_nulls = main_parallel_shuffling(
        starters, non_starters, n_permutations=n_permutations, n_jobs=n_jobs
    )

    return observed_cm, all_nulls


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
