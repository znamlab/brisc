import numpy as np
import pandas as pd
import scanpy as sc
import iss_preprocess as iss
import flexiznam as flz
from iss_preprocess.io import get_processed_path, get_roi_dimensions
from iss_preprocess.segment import get_cell_masks
from iss_analysis.pipeline import make_cell_dataframe, spots_ara_infos, segment_spots
from iss_analysis.segment import get_barcode_in_cells, match_starter_to_barcodes
from manuscript_analysis.load import get_ancestor_rank1, assign_areas_layers
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm


def assign_cell_barcodes(
    project: str = "becalia_rabies_barseq",
    mouse: str = "BRAC8498.3e",
    valid_chambers: list = ["chamber_07", "chamber_08", "chamber_09", "chamber_10"],
    error_correction_ds_name: str = "BRAC8498.3e_error_corrected_barcodes_10",
    redo_barcode_assignment: bool = True,
    redo_barcode_ara: bool = False,
    redo_gene_assignment: bool = False,
):
    """Orchestrates the assignment of barcodes and genes to individual cells.

    This function integrates data from rabies barcoding, mCherry starter cell
    identification, and in situ gene expression profiling to generate a
    comprehensive DataFrame (`cell_barcode_df`). Each row in this DataFrame
    represents a cell, annotated with its properties, assigned barcodes,
    gene expression, starter status, and anatomical location.

    The pipeline involves several steps:
    1.  **Barcode Assignment**:
        - Calls `iss_analysis.segment.get_barcode_in_cells` to process
          error-corrected barcode spots.
        - Assigns barcodes to cell masks and retrieves ARA (Allen Reference Atlas)
          information for these rabies-positive cells and spots.
        - Results (`rab_spot_df`, `rab_cells_properties`) are saved or loaded
          from CSV files.
    2.  **Starter Cell (mCherry) Identification**:
        - Calls `iss_analysis.segment.match_starter_to_barcodes` to identify
          mCherry-positive starter cells.
        - Matches these starter cells to the barcoded cells using mask-based
          proximity of rabies spots to mCherry signals.
        - Updates `rabies_cell_properties` with starter cell information
          (e.g., `is_starter`, `mcherry_uid`).
    3.  **Gene-to-Cell Assignment**:
        - Calls `assign_genes_to_cells` (which uses `process_chamber`) to
          segment in situ gene expression spots and assign them to cell masks
          across the specified chambers.
        - This yields `fused_df` (gene spots with cell assignments) and
          `cell_df` (cell properties including ARA from cell masks and gene counts).
        - Results are saved or loaded from pickle files.
    4.  **Data Fusion**:
        - Filters `cell_df` and `fused_df` to ensure consistency, using
          `mask_uid` (a unique identifier for each cell: chamber_roi_label)
          as the key.
        - Merges the gene expression data from `filtered_cell_df` with the
          barcode and starter cell information from `rabies_cell_properties`.
          This forms the primary `cell_barcode_df`.
    5.  **Anatomical and Cluster Annotation**:
        - Assigns broader anatomical areas and layers using
          `manuscript_analysis.load.assign_areas_layers` and `get_ancestor_rank1`.
        - Adds pre-computed cell type cluster annotations by loading an
          `AnnData` object (`adata_annotated.h5ad`).
    6.  **Flatmap Coordinates (NOT IMPLEMENTED)**:
        - A section (currently disabled with `if False`) exists to add
          flatmap coordinates.

    The function includes flags (`redo_*`) to control whether certain
    computationally intensive steps are re-executed or if previously saved
    results are loaded.

    Args:
        project (str, optional): The name of the project.
            Defaults to "becalia_rabies_barseq".
        mouse (str, optional): The mouse identifier.
            Defaults to "BRAC8498.3e".
        valid_chambers (list[str], optional): A list of chamber names to process.
            Defaults to ["chamber_07", "chamber_08", "chamber_09", "chamber_10"].
        error_correction_ds_name (str, optional): The name of the dataset
            containing error-corrected barcode sequences.
            Defaults to "BRAC8498.3e_error_corrected_barcodes_10".
        redo_barcode_assignment (bool, optional): If True, re-runs the
            barcode-to-cell assignment and ARA annotation for rabies spots.
            Otherwise, attempts to load from saved CSV files. Defaults to True.
        redo_barcode_ara (bool, optional): If True (and
            `redo_barcode_assignment` is True), forces recalculation of ARA
            information for rabies spots within `get_barcode_in_cells`.
            Defaults to False.
        redo_gene_assignment (bool, optional): If True, re-runs the
            gene-to-cell assignment. Otherwise, attempts to load from saved
            pickle files. Defaults to False.

    Returns:
        pd.DataFrame:
            `cell_barcode_df`: A DataFrame where each row corresponds to a cell
            detected in the gene expression analysis. It includes columns for:
            - Basic cell properties (e.g., 'x', 'y', 'label', 'mask_uid',
              'chamber', 'roi').
            - Gene expression counts for various genes (columns like 'gene_GENENAME').
            - ARA information derived from cell masks (e.g., 'area_acronym',
              'area_id').
            - Rabies barcode information (if the cell was barcoded, e.g.,
              'main_barcode', 'all_barcodes', 'total_n_spots', 'n_unique_barcodes').
              These columns will be NaN for cells not found in `rabies_cell_properties`.
            - Starter cell status (e.g., 'is_starter', 'mcherry_uid').
            - Derived anatomical annotations (e.g.,
              'area_acronym_ancestor_rank1', 'cortical_area', 'cortical_layer').
            - Annotated cell type clusters ('Annotated_clusters').
            - Potentially flatmap coordinates (if the relevant code section is enabled).
            The DataFrame is indexed by `mask_uid`.
    """
    processed_path = get_processed_path(Path(project / mouse / "analysis"))
    # Barcode assignment takes about 30 mins with ara redo, shorter without
    target = processed_path / f"{error_correction_ds_name}_rabies_spots.csv"
    if redo_barcode_assignment or not target.exists():
        print("Merging barcodes assingment from all ROIs...")
        rab_spot_df, rab_cells_barcodes, rab_cells_properties = get_barcode_in_cells(
            project=project,
            mouse=mouse,
            error_correction_ds_name=error_correction_ds_name,
            valid_chambers=valid_chambers,
            save_folder=None,
            verbose=True,
            add_ara_properties=True,
            redo=redo_barcode_ara,
        )

        rab_spot_df.to_csv(target, index=False)
        rab_cells_properties.to_csv(
            processed_path / "rabies_cells_properties.csv", index=False
        )

    else:
        print("Loading existing barcode assignments...")
        rab_spot_df = pd.read_csv(target)
        rab_cells_properties = pd.read_csv(
            processed_path / "rabies_cells_properties.csv"
        )

    # mcherry assignment
    (
        mcherry_cell_properties,
        rab_spot_df,
        rabies_cell_properties,
    ) = match_starter_to_barcodes(
        project=project,
        mouse=mouse,
        rabies_cell_properties=rab_cells_properties,
        rab_spot_df=rab_spot_df,
        mcherry_cells=None,  # USING CURATED MCHERRY CELLS BY DEFAULT AND USING MASK METHOD NOT SPOT
        which="curated",
        method="masks",
        verbose=True,
        max_starter_distance=0.5,  # distance to mask borders in um
        min_spot_number=3,  # minimum number of spots to consider a mcherry cells starter
        min_percentage_in_mask=10,  # minimum percentage of spots in mask to consider a mcherry cells starter
        mcherry_prefix="mCherry_1",  # acquisition prefix for mCherry images
    )

    assert (
        rabies_cell_properties.x.isna().sum() == 0
    ), "Error got an old rabies_cell_properties"

    # Perform gene assignments
    if redo_gene_assignment:
        print("Assigning genes to cells...")
        cell_barcode_df = assign_genes_to_cells(
            chambers=valid_chambers,
            save_assignments=False,
        )
    else:
        print("Loading existing gene assignments...")
        fused_df = pd.read_pickle(processed_path / "new_fused_df.pkl")
        cell_df = pd.read_pickle(processed_path / "new_cell_df.pkl")

    # Fusing cell mask with gene assignment dataframes
    cell_index = cell_df.mask_uid
    fused_index = fused_df.mask_uid
    filtered_fused_df = fused_df[fused_df.mask_uid.isin(cell_index)]
    filtered_cell_df = cell_df[cell_df.mask_uid.isin(fused_index)]
    filtered_fused_df.index = filtered_fused_df.mask_uid
    filtered_cell_df.index = filtered_cell_df.mask_uid
    filtered_fused_df = filtered_fused_df.drop(columns=["mask_uid"])
    filtered_cell_df = filtered_cell_df.drop(columns=["mask_uid"])
    filtered_fused_df = filtered_fused_df.drop(
        columns=[
            "unassigned1",
            "unassigned2",
            "unassigned3",
            "unassigned4",
            "unassigned5",
            "chamber",
            "roi",
        ]
    )

    cell_barcode_df = filtered_cell_df.merge(
        rabies_cell_properties[
            [
                "cell_id",
                "max_n_spots",
                "main_barcode",
                "n_unique_barcodes",
                "total_n_spots",
                "all_barcodes",
                "n_spots_per_barcode",
                "mcherry_uid",
                "is_starter",
            ]
        ],
        left_index=True,
        right_index=True,
        how="left",  # Keep all rows from the first DataFrame
    )

    # Assign areas and layers to each cell
    cell_barcode_df["area_acronym_ancestor_rank1"] = cell_barcode_df[
        "area_acronym"
    ].apply(get_ancestor_rank1)
    cell_barcode_df = assign_areas_layers(cell_barcode_df)

    # Add annotated clusters from BRAC8498.3e clustering
    adata = sc.read_h5ad(processed_path / "adata_annotated.h5ad")
    cell_barcode_df["Annotated_clusters"] = adata.obs.Annotated_clusters

    if False:
        # TODO: we might want to save flatmap coordinates
        # Add flatmap coordinates
        coords_flatmap = pd.read_pickle(processed_path / "rab_cell_properties.pkl")
        cell_barcode_df = cell_barcode_df.merge(
            coords_flatmap[
                [
                    "unnormalised_depth",
                    "normalised_depth",
                    "normalised_layers",
                    "flatmap_dorsal_x",
                    "flatmap_dorsal_y",
                ]
            ],
            left_index=True,
            right_index=True,
            how="left",
        )

    return cell_barcode_df


def assign_genes_to_cells(
    chambers=["chamber_07", "chamber_08", "chamber_09", "chamber_10"],
    save_assignments: bool = False,
):
    """Processes multiple chambers in parallel to assign genes to cells.

    This function takes a list of chamber names and processes each one using
    the `process_chamber` function. The processing is done in parallel using
    a multiprocessing Pool.

    After all chambers are processed, the resulting DataFrames (one for
    fused spots with gene assignments, and one for cell properties) from
    each chamber are concatenated into two final DataFrames. These final
    DataFrames can optionally be saved to disk.

    Args:
        chambers (list[str], optional): A list of chamber names to process.
            Defaults to ["chamber_07", "chamber_08", "chamber_09", "chamber_10"].
        save_assignments (bool, optional): If True, the final concatenated
            DataFrames (fused_df and cell_df) will be saved to pickle files
            in the analysis directory of the "becalia_rabies_barseq/BRAC8498.3e/"
            project. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - fused_df (pd.DataFrame): A DataFrame containing all segmented
              spots with their assigned genes from all processed chambers.
            - cell_df (pd.DataFrame): A DataFrame containing properties for all
              detected cells from all processed chambers.
    """
    # Use multiprocessing to process chambers in parallel

    # Initialize multiprocessing Pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_chamber, chambers)

    # Collect the results from all chambers
    fused_spot_dfs = [result[0] for result in results]
    cell_dfs = [result[1] for result in results]

    # Concatenate all results into final dataframes
    fused_df = pd.concat(fused_spot_dfs)
    cell_df = pd.concat(cell_dfs)

    # Save final dataframes
    processed_path = get_processed_path("becalia_rabies_barseq/BRAC8498.3e/")
    if save_assignments:
        pd.to_pickle(fused_df, processed_path / "analysis" / "new_fused_df.pkl")
        pd.to_pickle(cell_df, processed_path / "analysis" / "new_cell_df.pkl")
    return fused_df, cell_df


def process_chamber(
    chamber,
    save_assignments: bool = False,
):
    """Processes a single chamber to extract cell and spot data.

    For a given chamber, this function iterates through all its regions of
    interest (ROIs). In each ROI, it loads expanded cell masks, calculates
    cell mask properties (including ARA information), and segments spots to
    assign them to genes.

    The resulting cell data and fused spot data (spots with assigned genes)
    are collected for all ROIs within the chamber. Optionally, these
    DataFrames can be saved to disk.

    Args:
        chamber (str): The name of the chamber to process (e.g., "chamber_07").
        save_assignments (bool, optional): If True, the resulting DataFrames
            (fused_df and cell_df) for the chamber will be saved to pickle
            files in the analysis directory. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - fused_df (pd.DataFrame): A DataFrame containing all segmented
              spots with their assigned genes, x, y coordinates, chamber, ROI,
              and a unique mask_uid (chamber_roi_label).
            - cell_df (pd.DataFrame): A DataFrame containing properties for all
              detected cells, including their label, x, y coordinates, ARA info,
              chamber, ROI, and a unique mask_uid.
    """
    cell_dfs = []
    fused_spot_dfs = []

    data_path = f"becalia_rabies_barseq/BRAC8498.3e/{chamber}"
    processed_path = get_processed_path(data_path)
    roi_dims = get_roi_dimensions(data_path)

    for roi in tqdm(roi_dims[:, 0], total=len(roi_dims[:, 0])):
        # Load expanded cell masks
        print(f"Loading {chamber} ROI {roi}")
        big_masks = get_cell_masks(
            data_path, roi, projection="corrected", mask_expansion=5
        )

        print("Calculating cell mask properties")
        cell_df = make_cell_dataframe(
            data_path,
            roi=roi,
            masks=big_masks,
            mask_expansion=None,
        )
        cell_df = spots_ara_infos(
            data_path,
            spots=cell_df,
            atlas_size=10,
            roi=roi,
            acronyms=True,
            inplace=True,
        )

        _, fused_spot_df = segment_spots(
            data_path,
            roi,
            masks=big_masks,
            barcode_df=None,
            barcode_dot_threshold=None,
            spot_score_threshold=0.1,
            hyb_score_threshold=0.8,
            load_genes=True,
            load_hyb=True,
            load_barcodes=False,
        )
        fused_spot_df["chamber"] = chamber
        fused_spot_df["roi"] = roi
        fused_spot_df["mask_uid"] = fused_spot_df.apply(
            lambda row: f"{row['chamber']}_{row['roi']}_{int(row.name)}", axis=1
        )
        fused_spot_dfs.append(fused_spot_df)

        # Prepare cell_df for collation
        cell_df["chamber"] = chamber
        cell_df["roi"] = roi
        cell_df["mask_uid"] = cell_df.apply(
            lambda row: f"{row['chamber']}_{row['roi']}_{int(row['label'])}", axis=1
        )
        cell_dfs.append(cell_df)

    # Concatenate the results for the chamber
    fused_df = pd.concat(fused_spot_dfs)
    cell_df = pd.concat(cell_dfs)

    # Save data
    if save_assignments:
        pd.to_pickle(
            fused_df, processed_path.parent / "analysis" / f"{chamber}_fused_df.pkl"
        )
        pd.to_pickle(
            cell_df, processed_path.parent / "analysis" / f"{chamber}_cell_df.pkl"
        )

    return fused_df, cell_df
