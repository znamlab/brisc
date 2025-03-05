from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tf
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.ndimage import map_coordinates
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = (
    42  # Use Type 3 fonts (TrueType) for selectable text
)
matplotlib.rcParams["ps.fonttype"] = 42  # For EPS, if relevant


def normalize(data, vmin, vmax):
    return np.clip((data - vmin) / (vmax - vmin), 0, 1)


def mask_points(pts, mask):
    """
    Function to mask points based on the 3D mask array

    Args:
        pts (pandas.DataFrame): DataFrame with columns 'x', 'y', 'z' for the
            coordinates of the points.
        mask (numpy.ndarray): 3D mask array.

    Returns:
        pandas.DataFrame: DataFrame with the points that are within the mask
    """
    # Extract the floating-point coordinates
    coords = pts.iloc[:, :3].values.T  # Transpose to match map_coordinates format
    # Interpolate the mask at the floating-point coordinates
    interpolated_values = map_coordinates(
        mask, coords, order=1, mode="constant", cval=0
    )

    # Filter points where the interpolated mask value is greater than 0
    return pts[interpolated_values > 0]


def plot_rv_coronal_slice(
    project="rabies_barcoding",
    mouse="BRYC64.2i",
    processed=Path("/nemo/lab/znamenskiyp/home/shared/projects/"),
    inj_center=np.array([673, 205, 890]),
    label_fontsize=12,
    ax=None,
):
    """

    Returns:
        matplotlib.axes.Axes: Axes object with the plot.
        matplotlib.axes.Axes: Axes object with the secondary x-axis.
    """

    mcherry_file = (
        processed
        / project
        / mouse
        / "cellfinder_results_010/registration/downsampled_channel_0.tiff"
    )
    background_file = (
        processed
        / project
        / mouse
        / "cellfinder_results_010/registration/downsampled.tiff"
    )
    reg_folder = processed / project / mouse / "cellfinder_results_010/registration"
    points_file = (
        processed / project / mouse / "cellfinder_results_010/points/downsampled.points"
    )
    atlas = tf.imread(reg_folder / "registered_atlas.tiff")
    pts = pd.read_hdf(points_file)
    mcherry = tf.imread(mcherry_file)
    background = tf.imread(background_file)

    # Load the Allen Brain Atlas
    atlas_obj = BrainGlobeAtlas(
        "allen_mouse_10um"
    )  # Use the appropriate atlas for your data

    # Get all areas in the isocortex
    isocortex_regions = atlas_obj.get_structure_descendants("Isocortex")

    # Convert acronyms to IDs
    isocortex_ids = [
        atlas_obj.structures[acronym]["id"] for acronym in isocortex_regions
    ]

    # Create a binary mask
    mask = np.isin(atlas, isocortex_ids).astype(np.uint8)

    masked_points = mask_points(pts, mask)

    # Normalize and create RGB for the second subplot
    cropped_mcherry = mcherry[620:750, :, :]
    cropped_mcherry_normalized = normalize(cropped_mcherry, 0, 3000)
    cropped_background_channel = normalize(background[620:750, :, :], 0, 1500)

    rgb2 = np.zeros((*cropped_mcherry_normalized.max(axis=0).shape, 3))
    rgb2[..., 0] = cropped_mcherry_normalized.max(axis=0)  # Red channel for mCherry
    rgb2[..., 1] = cropped_background_channel.max(
        axis=0
    )  # Green channel for background
    rgb2[..., 2] = cropped_background_channel.max(axis=0)  # Blue channel for background

    ax.imshow(rgb2)
    plot_cells = False
    if plot_cells:
        ax.scatter(
            masked_points.iloc[:, 2],
            masked_points.iloc[:, 1],
            s=0.1,
            alpha=0.3,
            color="white",
        )
    ax.scatter(inj_center[2], inj_center[1], s=5, color="white")
    circle2 = plt.Circle(
        (inj_center[2], inj_center[1]),
        50,
        color="lightblue",
        linewidth=1,
        fill=False,
        alpha=0.6,
    )
    ax.add_artist(circle2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add scale bar to the second subplot
    scalebar2 = plt.Line2D(
        [rgb2.shape[1] - 110, rgb2.shape[1] - 10],
        [rgb2.shape[0] - 20, rgb2.shape[0] - 20],
        color="white",
        linewidth=4,
    )
    ax.add_line(scalebar2)
    ax.text(
        rgb2.shape[1] - 60,
        rgb2.shape[0] - 30,
        "1 mm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=label_fontsize,
    )

    return ax


def plot_rabies_density(
    inj_center=np.array([673, 205, 890]),
    project="rabies_barcoding",
    mouse="BRYC64.2i",
    processed=Path("/nemo/lab/znamenskiyp/home/shared/projects/"),
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
):
    """_summary_"""
    points_file = (
        processed / project / mouse / "cellfinder_results_010/points/downsampled.points"
    )
    pts = pd.read_hdf(points_file)
    reg_folder = processed / project / mouse / "cellfinder_results_010/registration"
    atlas = tf.imread(reg_folder / "registered_atlas.tiff")
    # Load the Allen Brain Atlas
    atlas_obj = BrainGlobeAtlas(
        "allen_mouse_10um"
    )  # Use the appropriate atlas for your data

    # Get all areas in the isocortex
    isocortex_regions = atlas_obj.get_structure_descendants("Isocortex")
    # Convert acronyms to IDs
    isocortex_ids = [
        atlas_obj.structures[acronym]["id"] for acronym in isocortex_regions
    ]
    mask = np.isin(atlas, isocortex_ids).astype(np.uint8)
    dst_to_center_pts = np.sqrt(np.sum((pts - inj_center) ** 2, axis=1))
    dst_to_center_pts = np.sort(dst_to_center_pts) / 100
    masked_points = mask_points(pts, mask)
    cells_masked = masked_points.copy()
    dst_to_center_masked = np.sqrt(np.sum((cells_masked - inj_center) ** 2, axis=1))
    dst_to_center_masked = np.sort(dst_to_center_masked) / 100

    bins = np.arange(0, 10, 0.1)
    volumes = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    hist_masked, _ = np.histogram(dst_to_center_masked, bins=bins)
    hist_pts, _ = np.histogram(dst_to_center_pts, bins=bins)
    density_masked = hist_masked / volumes
    density_pts = hist_pts / volumes
    ax.plot(
        bins[:-1],
        density_masked,
        color="k",
        alpha=0.7,
        label="masked_points",
        linewidth=2,
    )
    ax.plot(bins[:-1], density_pts, color="r", alpha=0.7, label="pts", linewidth=2)
    ax.set_xlabel(
        "Distance to injection (mm)",
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        "Cell density (cells/mm$^3$)",
        fontsize=label_fontsize,
    )
    ax.set_xlim(0, 2)
    ax.legend(
        fontsize=tick_fontsize,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    return ax
