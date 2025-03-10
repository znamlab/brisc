import iss_preprocess as issp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import expand_labels
from skimage import measure
from tqdm import tqdm


def load_data():
    project = "becalia_rabies_barseq"
    mouse = "BRAC8780.3f"
    chamber = "chamber_08"
    prefix = "hybridisation_round_1_1"
    data_path = f"{project}/{mouse}/{chamber}"
    rab_genes_channel = 1
    padlock_channel = 3
    chans = [rab_genes_channel, padlock_channel]

    ops = issp.io.load_ops(data_path)
    roi_dims = issp.io.load.get_roi_dimensions(data_path, prefix=prefix)

    example_tile, bad_px = issp.pipeline.load_and_register_tile(
        data_path,
        prefix=prefix,
        tile_coors=(10, 0, 0),
        projection="max-median",
        correct_illumination=False,
        filter_r=False,
    )
    # keep only round 1
    example_tile = example_tile[..., 0]

    th = 20
    r1 = 9
    r2 = 37
    footprint = 10
    threshold = 100

    def detect_rab_cells(
        stack,
        rab_genes_channel=1,
        padlock_channel=3,
        r1=9,
        r2=41,
        threshold=100,
        footprint=10,
    ):
        rab_genes = stack[..., rab_genes_channel]
        chans = [rab_genes_channel, padlock_channel]
        binary = issp.pipeline.segment._filter_mcherry_masks(
            rab_genes, r1, r2, threshold, footprint=footprint
        )
        labeled_image = measure.label(binary)
        labeled_image = expand_labels(labeled_image, distance=footprint)
        labeled_image, props_df = issp.segment.cells.label_image(
            binary, example_tile[..., chans]
        )
        return labeled_image, props_df

    rab_cells, spots_dfs = [], []
    rabies_stack = None
    labeled_images = None
    for roi in tqdm(roi_dims[:, 0], total=len(roi_dims)):
        print(f"Processing roi {roi}")
        stack, bad_px = issp.pipeline.load_and_register_tile(
            data_path,
            prefix=prefix,
            tile_coors=(roi, 0, 0),
            projection="max-median",
            correct_illumination=False,
            filter_r=False,
        )
        stack = stack[..., 0]
        labeled_image, props_df = detect_rab_cells(
            stack,
            rab_genes_channel=rab_genes_channel,
            padlock_channel=padlock_channel,
            r1=r1,
            r2=r2,
            threshold=threshold,
            footprint=footprint,
        )
        labeled_image = expand_labels(labeled_image, int(5 / 0.23))
        padlock = stack[..., padlock_channel]
        filtered = issp.image.filter_stack(padlock, r1=2, r2=4)
        spots = issp.segment.detect_spots(filtered, threshold=th, spot_size=2)
        spots["mask_id"] = labeled_image[spots.y, spots.x]
        spots["roi"] = roi
        props_df["roi"] = roi

        rab_cells.append(props_df)
        spots_dfs.append(spots)
        if rabies_stack is None:
            rabies_stack = np.zeros((stack.shape[0], stack.shape[1], 2, len(roi_dims)))
            labeled_images = np.zeros((stack.shape[0], stack.shape[1], len(roi_dims)))
        rabies_stack[..., roi - 1] = stack[..., chans].copy()
        labeled_images[..., roi - 1] = labeled_image

    rab_cells = pd.concat(rab_cells)
    spots_df = pd.concat(spots_dfs)
    ops = issp.io.load_ops(data_path)
    good_cells = issp.pipeline.segment._filter_masks(
        ops, rab_cells, labeled_image=None
    ).copy()
    good_cells["cell_uid"] = (
        good_cells["roi"].astype(str) + "_" + good_cells["label"].astype(str)
    )
    spots_df["cell_uid"] = (
        spots_df["roi"].astype(str) + "_" + spots_df["mask_id"].astype(str)
    )
    spots_df["valid_cell"] = spots_df["cell_uid"].isin(good_cells["cell_uid"])
    spots_df["valid_cell"].value_counts()

    print(f"Number of cells: {len(good_cells)}")
    good_cells["spot_count"] = 0
    for cell_uid, count in spots_df["cell_uid"].value_counts().items():
        good_cells.loc[good_cells["cell_uid"] == cell_uid, "spot_count"] = count

    return good_cells


def plot_histogram_bc_per_cell(
    good_cells,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
):
    bars = ax.hist(good_cells["spot_count"], bins=np.arange(40) - 0.5, color="grey")

    # Change the color of the first three bars to red
    for bar in bars[2][:3]:
        bar.set_facecolor("red")

    for n in [1, 3, 5]:
        lt_n = good_cells["spot_count"].lt(n).sum()
        print(
            f"Cells with less than {n} spots: {lt_n}, {lt_n/len(good_cells)*100:.2f}%"
        )
    ax.set_xlabel(
        "Barcode spots per cell",
        fontsize=label_fontsize,
    )
    ax.set_xlim(-0.5, 40)
    ax.set_ylabel(
        "Number of Cells",
        fontsize=label_fontsize,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    # axes[1].set_xlabel("Barcode spots per cell")
    # axes[1].set_ylabel("Cumulative fraction")
    # axes[1].set_xlim(0, 20)
    # axes[1].set_ylim(0, 1)
