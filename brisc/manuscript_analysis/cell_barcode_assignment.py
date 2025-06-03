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


def process_chamber(
    chamber,
    save_assignments: bool = False,
):
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


def assign_genes_to_cells(
    chambers=["chamber_07", "chamber_08", "chamber_09", "chamber_10"],
    save_assignments: bool = False,
):
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


def assign_cell_barcodes(
    project: str = "becalia_rabies_barseq",
    mouse: str = "BRAC8498.3e",
    valid_chambers: list = ["chamber_07", "chamber_08", "chamber_09", "chamber_10"],
    error_correction_ds_name: str = "BRAC8498.3e_error_corrected_barcodes_10",
    redo_barcode_assignment: bool = True,
    redo_barcode_ara: bool = False,
    redo_gene_assignment: bool = False,
):
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
