import iss_preprocess as issp

import numpy as np
import pandas as pd
from skimage.segmentation import expand_labels
from tqdm import tqdm


def detect_rab_cells(
    stack,
    rab_genes_channel=1,
    padlock_channel=3,
    r1=9,
    r2=37,
    threshold=100,
    footprint=10,
    mask_expansion_um=5,
    pixel_size=0.23,
):
    rab_genes = stack[..., rab_genes_channel]
    binary = issp.pipeline.segment._filter_mcherry_masks(
        rab_genes, r1, r2, threshold, footprint=footprint
    )
    chans = [rab_genes_channel, padlock_channel]
    labeled_image, props_df = issp.segment.cells.label_image(binary, stack[..., chans])
    labeled_image = expand_labels(labeled_image, int(mask_expansion_um / pixel_size))
    return labeled_image, props_df


def load_data(
    project="becalia_rabies_barseq",
    mouse="BRAC8780.3f",
    chamber="chamber_08",
    prefix="hybridisation_round_1_1",
    rab_genes_channel=1,
    padlock_channel=3,
    th=20,
    r1=9,
    r2=37,
    footprint=10,
    mask_expansion_um=5,
    spot_size=2,
    threshold=100,
):
    data_path = f"{project}/{mouse}/{chamber}"
    ops = issp.io.load_ops(data_path)
    roi_dims = issp.io.load.get_roi_dimensions(data_path, prefix=prefix)
    chans = [rab_genes_channel, padlock_channel]
    pixel_size = issp.io.get_pixel_size(data_path, prefix)

    rab_cells, spots_dfs = [], []
    rabies_stack = None
    labeled_images = None
    for roi in tqdm(roi_dims[:, 0], total=len(roi_dims)):
        stack, _ = issp.pipeline.register.load_and_register_tile(
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
            mask_expansion_um=mask_expansion_um,
            pixel_size=pixel_size,
        )
        padlock = stack[..., padlock_channel]
        filtered = issp.image.filter_stack(padlock)
        spots = issp.segment.detect_spots(filtered, threshold=th, spot_size=spot_size)
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

    return good_cells, labeled_images, rabies_stack, spots_df


def plot_histogram_bc_per_cell(
    good_cells,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    hist_bins=40,
):
    bars = ax.hist(
        good_cells["spot_count"], bins=np.arange(hist_bins) - 0.5, color="grey"
    )

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


def plot_cells_spots(
    good_cells,
    rabies_stack,
    labeled_images,
    ax=None,
    roi_of_interest=5,
    min_y=450,
    max_y=1100,
    min_x=400,
    max_x=1000,
    colors=[(1, 0, 0), (0, 1, 1)],
    vmaxs=[800, 200],
    linewidth=0.9,
):
    """Plot the cells and spots for the fixed bounding box of the ROI of interest.

    Args:
        good_cells (pd.DataFrame): DataFrame with the cell properties.
        rabies_stack (np.ndarray): 4D array with the rabies stack.
        labeled_images (np.ndarray): 3D array with the labeled images.
        roi_of_interest (int): ROI to plot.
        min_y (int): Minimum y coordinate.
        max_y (int): Maximum y coordinate.
        min_x (int): Minimum x coordinate.
        max_x (int): Maximum x coordinate.
        colors (list): List with the colors for the channels.
        vmaxs (list): List with the vmax values for the channels.
    """
    # Filter cells for ROI 5
    no_spot = good_cells[good_cells["spot_count"] <= 2]
    roi_5_no_spot_cells = no_spot[no_spot["roi"] == roi_of_interest]

    # Extract the data and label slices for the fixed bounding box
    data = issp.vis.utils.get_stack_part(
        rabies_stack[..., roi_of_interest - 1],
        [min_x, max_x],
        [min_y, max_y],
    )
    labels = issp.vis.utils.get_stack_part(
        labeled_images[..., roi_of_interest - 1],
        [min_x, max_x],
        [min_y, max_y],
    )

    # Convert to RGB for display
    rgb = issp.vis.to_rgb(data, colors=colors, vmin=[0, 0], vmax=vmaxs)

    # Plot
    ax.imshow(rgb)

    # (a) Outline all cells in black
    ax.contour(
        labels, levels=np.arange(labels.max()) + 0.5, colors="k", linewidths=linewidth
    )

    # (b) Highlight the no_spot cells in white
    if not roi_5_no_spot_cells.empty:
        no_spot_label_ids = roi_5_no_spot_cells["label"].unique()
        for lbl_id in no_spot_label_ids:
            ax.contour(
                labels == lbl_id, levels=[0.5], colors="white", linewidths=linewidth
            )
    ax.set_xticks([])
    ax.set_yticks([])
