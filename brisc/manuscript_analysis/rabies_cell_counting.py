from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tf
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.ndimage import map_coordinates
from matplotlib import pyplot as plt
import matplotlib
from .utils import despine, get_path

matplotlib.rcParams[
    "pdf.fonttype"
] = 42  # Use Type 3 fonts (TrueType) for selectable text
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
    injection_center,
    ax,
    mcherry,
    background,
    z_proj_size=100,
    vlim_mcherry=(0, 3000),
    vlim_background=(0, 1500),
    zoom_row_lims=(-90, 380),
    zoom_col_lims=(-310, 290),
    use_downsampled=False,
):
    """
    Plot a coronal slice of the brain with the injection site.

    Args:
        injection_center (np.ndarray): Injection center in voxel coordinates (Z, Y, X).
        ax (matplotlib.axes._axes.Axes or None): Axes on which to plot.
        mcherry (np.ndarray): mCherry channel data.
        background (np.ndarray): Background channel data.
        z_proj_size (int): Size of the z-projection (ignored when use_downsampled=True).
        vlim_mcherry (tuple): (vmin, vmax) for mCherry normalization.
        vlim_background (tuple): (vmin, vmax) for background normalization.
        zoom_row_lims (tuple): (min_offset, max_offset) for cropping the zoom plot along rows.
        zoom_col_lims (tuple): (min_offset, max_offset) for cropping the zoom plot along columns.
        use_downsampled (bool): If True, treat mcherry/background as 3-D registered
            downsampled volumes and take a single coronal slice at injection_center[0]
            instead of computing a Z-projection over z_proj_size slices.

    Returns:
        matplotlib.axes.Axes: Axes object with the plot.
        matplotlib.axes.Axes: Axes object with the secondary x-axis.
    """

    # Parameters for BRYC64.2h (mouse used for initial submission)
    # z_proj_size=100,
    # vlim_mcherry=(0, 3000),
    # vlim_background=(0, 1500),
    # zoom_row_lims=(-90, 380),
    # zoom_col_lims=(-310, 290),

    # Parameters for BRAC11816.2e (revision experiment)

    # Load the Allen Brain Atlas
    # atlas_obj = BrainGlobeAtlas("allen_mouse_10um")

    # Get all areas in the isocortex
    # isocortex_regions = atlas_obj.get_structure_descendants("Isocortex")

    # Convert acronyms to IDs
    # isocortex_ids = [
    #     atlas_obj.structures[acronym]["id"] for acronym in isocortex_regions
    # ]

    # Create a binary mask
    # mask = np.isin(atlas, isocortex_ids).astype(np.uint8)

    # masked_points = mask_points(pts, mask)

    # Normalize and create RGB for the second subplot[673, 205, 890]
    z, row, col = injection_center
    if use_downsampled:
        nz = int(z_proj_size / 2)
        cropped_mcherry = mcherry[z - nz : z + nz, :, :].astype(float)
        cropped_background_channel = background[z - nz : z + nz, :, :].astype(float)
        cropped_mcherry_normalized = normalize(
            cropped_mcherry, vlim_mcherry[0], vlim_mcherry[1]
        ).max(axis=0)
        cropped_background_normalized = normalize(
            cropped_background_channel, vlim_background[0], vlim_background[1]
        ).max(axis=0)
    else:
        # full res volumes are already projected;
        pixel_size = 2
        cropped_mcherry_normalized = normalize(
            mcherry.astype(float), vlim_mcherry[0], vlim_mcherry[1]
        )
        cropped_background_normalized = normalize(
            background.astype(float), vlim_background[0], vlim_background[1]
        )

    rgb2 = np.zeros((*cropped_mcherry_normalized.shape, 3))
    rgb2[..., 0] = cropped_mcherry_normalized  # Red channel for mCherry
    rgb2[..., 1] = cropped_background_normalized  # Green channel for background
    rgb2[..., 2] = cropped_background_normalized  # Blue channel for background
    ax_zoom, ax_overview = ax
    ax_overview.imshow(rgb2)
    ax_overview.set_aspect("equal")
    ax_overview.set_xticks([])
    ax_overview.set_yticks([])
    for spine in ax_overview.spines.values():
        spine.set_edgecolor("white")

    rgb_zoom = rgb2[
        max(0, row + zoom_row_lims[0]) : row + zoom_row_lims[1],
        max(0, col + zoom_col_lims[0]) : col + zoom_col_lims[1],
        :,
    ]
    ax_zoom.imshow(rgb_zoom)
    ax_zoom.set_aspect("equal")
    ax_zoom.axis("off")

    if use_downsampled:
        xdata = [rgb_zoom.shape[1] - 240, rgb_zoom.shape[1] - 140]
        ydata = [rgb_zoom.shape[0] - 50, rgb_zoom.shape[0] - 50]
    else:
        xdata = [100, 100 + 1000 / pixel_size]
        ydata = [rgb_zoom.shape[0] - 100, rgb_zoom.shape[0] - 100]
    scalebar2 = plt.Line2D(
        xdata,
        ydata,
        color="white",
        linewidth=4,
    )
    ax_zoom.add_line(scalebar2)


def mask_rounded_points(points, mask):
    """
    Return only those points (Nx3) whose integer voxel coordinates fall in mask > 0.
    Here, `mask` should be the same shape as the volume from which `points` were extracted.
    """
    # Round coords to nearest integer, ensure inside volume bounds
    rounded = np.round(points).astype(int)
    Z, Y, X = mask.shape
    in_bounds = (
        (0 <= rounded[:, 0])
        & (rounded[:, 0] < Z)
        & (0 <= rounded[:, 1])
        & (rounded[:, 1] < Y)
        & (0 <= rounded[:, 2])
        & (rounded[:, 2] < X)
    )
    valid_points = rounded[in_bounds]
    # Now keep only those where mask>0
    keep_idx = mask[valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]] > 0
    # Return the *original floating coordinates* that correspond to valid mask
    return points[in_bounds][keep_idx]


def plot_rabies_density(
    inj_center,
    project="rabies_barcoding",
    mouse="BRYC64.2h",
    data_root=None,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    max_radius_mm=2.0,
    n_points=200,
    linewidth=0.5,
    voxel_distances_sorted=None,
    cell_distances_sorted=None,
):
    """
    Plots the cumulative density of labeled cells in isocortex as a function
    of distance from the injection site. Distances and volumes are computed
    strictly within the isocortex mask.

    Parameters
    ----------
    inj_center : np.ndarray
        Injection center in voxel coordinates (Z, Y, X).
    project : str
        Project folder name.
    mouse : str
        Mouse folder name.
    data_root : Path, optional
        Base path to data. Defaults to None (auto-find in lab).
    ax : matplotlib.axes._axes.Axes or None
        Axes on which to plot.
    label_fontsize : int
        Fontsize for the x/y labels.
    tick_fontsize : int
        Fontsize for the tick labels.
    max_radius_mm : float
        Maximum radius (in mm) to plot.
    n_points : int
        Number of radius increments.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    # -------------------------------------------------------------------------
    if voxel_distances_sorted is None or cell_distances_sorted is None:
        # Load points and atlas
        voxel_distances_sorted, cell_distances_sorted = rv_cortical_cell_distances(
            inj_center, project, mouse, data_root
        )

    # -------------------------------------------------------------------------
    # Compute cumulative density in isocortex
    # For radius r, #voxels inside r = index in voxel_distances_sorted
    # Volume = (#voxels) * (0.01 mm)^3
    # #cells inside r similarly from cell_distances_sorted
    # density(r) = #cells(r) / volume(r)

    radii = np.linspace(0, max_radius_mm, n_points)
    densities = []
    voxel_volume = 0.01**3  # mm^3 per voxel

    for r in radii:
        # How many voxels are within r?
        # np.searchsorted gives index where r would be inserted to keep sorted order
        # side="right" => we get the count of elements <= r
        idx_vox = np.searchsorted(voxel_distances_sorted, r, side="right")
        # Number of isocortex cells within r
        idx_cells = np.searchsorted(cell_distances_sorted, r, side="right")

        if idx_vox == 0:
            densities.append(0.0)
            continue

        volume_r = idx_vox * voxel_volume  # mm^3 in isocortex
        cell_count = idx_cells
        density_r = cell_count / volume_r
        densities.append(density_r)

    # -------------------------------------------------------------------------
    # Plot
    ax.plot(
        radii,
        densities,
        linewidth=linewidth,
        color="black",
        label="Cumulative isocortex density",
    )
    ax.set_xlabel("Distance to\ninjection site (mm)", fontsize=label_fontsize)
    ax.set_ylabel("Cell density\n(cells / mm$^3$)", fontsize=label_fontsize)
    ax.set_xlim(0, max_radius_mm)
    # ax.legend(fontsize=tick_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    despine(ax)


def rv_cortical_cell_distances(inj_center, project, mouse, data_root=None):
    points_file = (
        get_path(project, data_root)
        / mouse
        / "cellfinder_results_010/points/downsampled.points"
    )

    pts = pd.read_hdf(points_file).values  # Nx3
    reg_folder = (
        get_path(project, data_root) / mouse / "cellfinder_results_010/registration"
    )
    atlas = tf.imread(reg_folder / "registered_atlas.tiff")

    # -------------------------------------------------------------------------
    # BrainGlobe atlas and mask for isocortex
    atlas_obj = BrainGlobeAtlas("allen_mouse_10um")
    isocortex_regions = atlas_obj.get_structure_descendants("Isocortex")
    isocortex_ids = [
        atlas_obj.structures[acronym]["id"] for acronym in isocortex_regions
    ]

    # Create isocortex mask (same shape as atlas)
    mask = np.isin(atlas, isocortex_ids).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Get voxel coordinates in isocortex
    # mask.shape = (Z, Y, X)
    # voxel_coords: N_voxels x 3 (Z, Y, X)
    voxel_coords = np.argwhere(mask > 0)

    # Convert injection center to mm (1 voxel = 0.01 mm)
    inj_center_mm = inj_center * 0.01

    # Distances of each isocortex voxel to the injection center
    voxel_coords_mm = voxel_coords * 0.01
    voxel_distances = np.sqrt(np.sum((voxel_coords_mm - inj_center_mm) ** 2, axis=1))
    voxel_distances_sorted = np.sort(voxel_distances)

    # -------------------------------------------------------------------------
    # Get only isocortex cells
    cells_masked = mask_rounded_points(pts, mask)
    # Distances of each isocortex cell to injection center
    cells_masked_mm = cells_masked * 0.01
    cell_distances = np.sqrt(np.sum((cells_masked_mm - inj_center_mm) ** 2, axis=1))
    cell_distances_sorted = np.sort(cell_distances)
    return voxel_distances_sorted, cell_distances_sorted
