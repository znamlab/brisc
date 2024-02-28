"""Helper functions for the petr_talk_202402 module."""
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from itertools import cycle
import matplotlib as mpl
from xml.etree import ElementTree
import iss_preprocess as iss


def get_barcodes(
    data_path,
    mean_intensity_threshold=0.01,
    dot_product_score_threshold=0.2,
    mean_score_threshold=0.75,
):
    """
    Get barcode spots from the given data path.

    Args:
        data_path (str): The path to the data.
        mean_intensity_threshold (float): The threshold for mean intensity. Default is 0.01.
        dot_product_score_threshold (float): The threshold for dot product score. Default is 0.2.
        mean_score_threshold (float): The threshold for mean score. Default is 0.75.

    Returns:
        barcode_spots (pd.DataFrame): DataFrame containing the barcode spots.
        gmm (GaussianMixture): The trained Gaussian Mixture Model.
        all_barcode_spots (pd.DataFrame): DataFrame containing all barcode spots.
    """
    data_folder = iss.io.get_processed_path(data_path)
    ops = iss.io.load.load_ops(data_path)
    all_barcode_spots = []
    for roi in ops["use_rois"]:
        barcode_spots = pd.read_pickle(data_folder / f"barcode_round_spots_{roi}.pkl")
        barcode_spots["roi"] = roi
        all_barcode_spots.append(barcode_spots)
    all_barcode_spots = pd.concat(all_barcode_spots, ignore_index=True)

    # filter the utter crap
    all_barcode_spots = all_barcode_spots[
        (all_barcode_spots.mean_intensity > mean_intensity_threshold)
        & (all_barcode_spots.dot_product_score > dot_product_score_threshold)
        & (all_barcode_spots.mean_score > mean_score_threshold)
    ].copy()

    # Do a GMM on the 4 metrics dot_product_score, spot_score, mean_intensity, mean_score,
    # with just 2 clusters
    # Extract the four metrics from the dataframe
    metrics = ["dot_product_score", "spot_score", "mean_intensity", "mean_score"]
    skip = len(all_barcode_spots) // 10000
    data = all_barcode_spots[metrics].values[::skip]

    # Perform GMM with two clusters
    means_init = np.nanpercentile(data, [1, 99], axis=0)
    gmm = GaussianMixture(n_components=2, means_init=means_init)
    gmm.fit(data)

    labels = gmm.predict(all_barcode_spots[metrics].values)
    barcode_spots = all_barcode_spots[labels == 1].copy()
    return barcode_spots, gmm, all_barcode_spots


LAYERS = (
    "Cux2",
    "Dgkb",
    "Lamp5",
    "Necab1",
    "Pcp4",
    "Pde1a",
    "Rorb",
    "Serpine2",
    "Spock3",
    "Foxp2",
    "Igfbp4",
    "Crym",
    "Ctgf",
)
# Then other that have interesting expression patterns
PATTERN = ("Chodl", "Cxcl14", "Fxyd6", "Id2", "Lypd1", "Matn2", "Pdyn", "Penk", "Synpr")
# Define the genes we're going to plot
# First the ones with expression in specific layers, more or less sorted by layer

# Other genes I'm not interested in (too broad or too sparse)
USELESS = (
    "Aldoc",
    "Calb2",
    "Cartpt",
    "Cdh9",
    "Chgb",
    "Cox6a2",
    "Cplx1",
    "Cplx3",
    "Crh",
    "Arpp21",
    "Necab2",
    "Crhbp",
    "Enpp2",
    "Gap43",
    "Gpx3",
    "Hpcal1",
    "Kcnab1",
    "Kcnc2",
    "Kcnip1",
    "Pvalb",
    "Calb1",
    "Cnr1",
    "Cpne6",
    "Kcnip4",
    "Luzp2",
    "Lypd6",
    "Mt1",
    "Ndnf",
    "Nefl",
    "Nos1",
    "Npy",
    "Nr4a2",
    "Cck",
    "Zcchc12",
    "Cd24a",
    "Nrip3",
    "Nrn1",
    "Nxph1",
    "Pak1",
    "Pcdh8",
    "Pcp4l1",
    "Prdx5",
    "Pthlh",
    "Cdh13",
    "Rab3b",
    "Rbp4",
    "Rrad",
    "Tac2",
    "Th",
    "Vstm2a",
)
USELESS = tuple([f"unassigned{i}" for i in range(1, 5)] + list(USELESS))
# Some that are biased toward some layers but a bit too broad
BAD_LAYERS = (
    "Fezf2",
    "Rgs4",
    "Ptn",
    "Car4",
    "Crtac1",
    "Cxcl12",
    "Fam19a1",
    "Nov",
    "Myl4",
    "Nnat",
    "Spon1",
)


def plot_gene_image(ax, genes_spots, layers=LAYERS, ok=PATTERN, **kwargs):
    """Plot the genes in the genes_spots dataframe on the ax.

    The genes are colored by layer, and the genes in the ok list are plotted smaller.

    Args:
        ax: a matplotlib axis
        genes_spots: a dataframe with columns x, y, gene, layer
        layers: ordered list of genes
        ok: other genes to plot
        **kwargs: additional keyword arguments to pass to ax.scatter

    Returns:
        None

    """
    default_kwargs = dict(s=1, marker="o", alpha=0.2)
    default_kwargs.update(kwargs)
    cm = mpl.colormaps["terrain"]
    for il, gene_name in enumerate(layers):
        spots = genes_spots[genes_spots.gene == gene_name]
        ax.scatter(
            spots.x,
            spots.y,
            color=cm(il / len(layers)),
            label=gene_name,
            **default_kwargs,
        )
    colors = cycle(
        [
            "mediumseagreen",
            "gold",
            "orchid",
            "mediumblue",
            "yellow",
            "slateblue",
            "cornflowerblue",
            "moccasin",
            "pink",
            "teal",
            "plum",
            "azure",
            "indianred",
        ]
    )
    default_kwargs["alpha"] /= 2
    default_kwargs["s"] /= 2
    for gene_name in ok:
        spots = genes_spots[genes_spots.gene == gene_name]
        ax.scatter(
            spots.x, spots.y, color=next(colors), label=gene_name, **default_kwargs
        )


def get_mcherry_cells(data_path, roi):
    """
    Retrieves the coordinates of manually clicked mCherry cells from an XML file.

    Args:
        data_path (str): The path to the data.
        roi (int): The region of interest.

    Returns:
        numpy.ndarray: An array of shape (N, 2) containing the coordinates of the mCherry cells.

    Raises:
        AssertionError: If the XML file does not exist or has an unexpected format.
    """

    reg_folder = iss.io.load.get_processed_path(data_path) / "reg"
    manual_click_folder = reg_folder.parent / "manual_mcherry_cell_detection"
    manual_xml = manual_click_folder / f"CellCounter_mCherry_1_roi{roi}_registered.xml"
    assert manual_xml.exists(), f"File {manual_xml} does not exist"
    tree = ElementTree.parse(manual_xml)
    root = tree.getroot()
    marker_data = root[1]
    marker_1 = marker_data[1]
    print(marker_1.tag)
    mcherry_cells = []
    for marker in marker_1:
        if marker.tag != "Marker":
            continue
        assert marker[0].tag == "MarkerX"
        assert marker[1].tag == "MarkerY"
        mcherry_cells.append([int(marker[0].text), int(marker[1].text)])
    mcherry_cells = np.array(mcherry_cells)
    print(f"Found {mcherry_cells.shape[0]} manually clicked cells")
    return mcherry_cells


def get_raw_data(data_path, roi, mcherry_channel=2):
    """Loads raw data for a specific region of interest (ROI).

    This is slow, about 15 minutes.

    Args:
        data_path (str): The path to the data.
        roi (int): The ROI number.
        mcherry_channel (int, optional): The channel number for the mCherry channel. Defaults to 2.

    Returns:
        stack: A 3D numpy array of shape (N, M, 3) containing the raw data.
        mask: A 2D numpy array containing the mask for the ROI.
    """

    ops = iss.io.load.load_ops(data_path)
    reg_folder = iss.io.load.get_processed_path(data_path) / "reg"
    tform = np.load(reg_folder / f"mCherry_1_roi{roi}_tform_to_ref.npz")
    print("Stitching the mCherry channel")
    stitched_reg = iss.pipeline.stitch_tiles(
        data_path,
        "mCherry_1",
        suffix="max",
        roi=roi,
        ich=mcherry_channel,
        correct_illumination=True,
    )
    print("Transforming the mCherry channel")
    stitched_reg = iss.reg.util.transform_image(
        stitched_reg, scale=tform["scale"], angle=tform["angle"], shift=tform["shift"]
    )
    stitched_reg -= stitched_reg.min()
    stitched_reg *= (2**16 - 1) / stitched_reg.max()
    stitched_reg = stitched_reg.astype("uint16")

    print("Stitching the reference channel")
    stitched_stack_reference = iss.pipeline.stitch.stitch_registered(
        data_path,
        prefix="genes_round_4_1",
        roi=roi,
        channels=[0, 1, 2, 3],
        ref_prefix="genes_round",
        filter_r=ops["filter_r"],
    )
    stitched_stack_reference = np.nanstd(stitched_stack_reference, axis=-1)
    stitched_stack_reference -= stitched_stack_reference.min()
    stitched_stack_reference *= (2**16 - 1) / stitched_stack_reference.max()
    stitched_stack_reference = stitched_stack_reference.astype("uint16")

    print("Stitching the rabies channel")
    stitched_stack_rabies = iss.pipeline.stitch.stitch_registered(
        data_path,
        prefix="barcode_round_1_1",
        roi=roi,
        channels=[0, 1, 2, 3],
        ref_prefix="genes_round",
        filter_r=ops["filter_r"],
    )

    stitched_stack_rabies = np.nanstd(stitched_stack_rabies, axis=-1)
    stitched_stack_rabies -= stitched_stack_rabies.min()
    stitched_stack_rabies *= (2**16 - 1) / stitched_stack_rabies.max()
    stitched_stack_rabies = stitched_stack_rabies.astype("uint16")

    print("Cropping to the smallest size")
    # crop to the smallest size
    shapes = [
        stitched_reg.shape,
        stitched_stack_reference.shape,
        stitched_stack_rabies.shape,
    ]
    smallest = np.min(shapes, axis=0)
    stitched_stack_reference = stitched_stack_reference[: smallest[0], : smallest[1]]
    stitched_stack_rabies = stitched_stack_rabies[: smallest[0], : smallest[1]]
    stitched_reg = stitched_reg[: smallest[0], : smallest[1]]

    # Make a stack and load masks
    stack = np.stack(
        [stitched_reg, stitched_stack_rabies, stitched_stack_reference], axis=2
    )

    print("Loading the mask")
    mask = np.load(iss.io.get_processed_path(data_path) / f"masks_{roi}.npy")
    mask = mask[: smallest[0], : smallest[1]]
    return stack, mask


def plot_stack_part(
    ax, ctr, w, st, mask=None, sh=(0, 0), show_mask=False, show_contours=True
):
    """
    Plots a part of the stack with optional mask and contours.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        ctr (tuple): The center coordinates of the part to plot.
        w (int): The width of the part to plot.
        st (numpy.ndarray): The stack to plot the part from.
        mask (numpy.ndarray): The mask to overlay on the part.
        sh (tuple, optional): The shift in x and y directions for the red channel. Defaults to (0, 0).
        show_mask (bool, optional): Whether to show the mask overlay. Defaults to False.
        show_contours (bool, optional): Whether to show the contours. Defaults to True.
    """

    part = st[ctr[1] - w : ctr[1] + w, ctr[0] - w : ctr[0] + w, :].copy()
    part[:, :, 0] = st[
        ctr[1] - w - sh[1] : ctr[1] + w - sh[1],
        ctr[0] - w - sh[0] : ctr[0] + w - sh[0],
        0,
    ]
    vmin = np.percentile(part, 0.1, axis=(0, 1))
    vmax = np.percentile(part, 99.99, axis=(0, 1))
    # vmax[2] = np.percentile(part[:,:,2], 99)
    rgb = iss.vis.to_rgb(
        part, colors=([1, 0, 0], [0, 1, 0], [1, 0, 1]), vmin=vmin, vmax=vmax
    )
    ax.imshow(rgb)
    if show_contours or show_mask:
        if mask is None:
            raise ValueError("Mask is required for contours or mask overlay")
        mpart = mask[ctr[1] - w : ctr[1] + w, ctr[0] - w : ctr[0] + w]
    if show_contours:
        ax.contour(mpart != 0, levels=[0.5], colors="pink", linewidths=1)
    if show_mask:
        mpart = np.array(mpart, dtype=float)
        mpart[mpart == 0] = np.nan
        ax.imshow(mpart, cmap="prism", alpha=0.3)


def get_spots(data_path, roi, barcode_spots):
    # get spots
    genes_spots = pd.read_pickle(
        iss.io.get_processed_path(data_path) / f"genes_round_spots_{roi}.pkl"
    )
    roi_barcode_spots = barcode_spots[barcode_spots.roi == roi]

    return genes_spots, roi_barcode_spots


def select_spots(spots, xlim, ylim):
    return spots[
        (spots.x > xlim[0])
        & (spots.x < xlim[1])
        & (spots.y > ylim[0])
        & (spots.y < ylim[1])
    ]
