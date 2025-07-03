from iss_preprocess.diagnostics.diag_stitching import plot_single_overview
from iss_preprocess.vis import round_to_rgb, to_rgb, add_bases_legend
from iss_preprocess.vis.utils import get_stack_part
from iss_preprocess.pipeline.sequencing import basecall_tile
from iss_preprocess.io import load_stack

import matplotlib.image as mpimg
import matplotlib.patches as patches
from skimage.measure import block_reduce
import numpy as np
from pathlib import Path


def run_plot_overview(
    roi=3,
    chamber=10,
    prefix="barcode_round_1_1",
    downsample_factor=25,
    ch=[0, 1, 2, 3],
    channel_colors=([0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0]),
    vmin=[63, 30, 63, 72],
    vmax=[1738, 722, 1284, 1077],
    filter_r=(2, 4),
):
    filter_r = True
    if filter_r:
        print("Filtering R")
    data_path = f"becalia_rabies_barseq/BRAC8498.3e/chamber_{chamber}"
    fig = plot_single_overview(
        data_path,
        prefix,
        roi,
        ch,
        nx=None,
        ny=None,
        plot_grid=False,
        downsample_factor=downsample_factor,
        save_raw=True,
        correct_illumination=True,
        filter_r=filter_r,
        channel_colors=channel_colors,
        vmin=vmin,
        vmax=vmax,
    )

    return fig


def plot_rabies_raw(ax, img, crop_top=50, crop_bottom=350, crop_left=50, crop_right=50):
    """
    Load a .png file from image_path, rotate it 90 degrees left,
    crop it according to the given parameters,
    and display it on the given Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to display the image.
    image_path : str or pathlib.Path
        The path to the .png image file.
    crop_top : int, optional
        Number of pixels to crop from the top. Default is 50.
    crop_bottom : int, optional
        Number of pixels to crop from the bottom. Default is 350.
    crop_left : int, optional
        Number of pixels to crop from the left. Default is 50.
    crop_right : int, optional
        Number of pixels to crop from the right. Default is 50.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axis with the image displayed.
    """
    # Load the image data
    img = mpimg.imread(img)
    img_rotated = np.rot90(img)
    img_cropped = img_rotated[crop_top:-crop_bottom, crop_left:-crop_right]
    ax.imshow(img_cropped[300:1500])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    return ax


def load_rv_bc_rounds(
    data_path,
    tile_coords=(5, 2, 7),
    center=None,
    window=200,
):
    stack, spot_sign_image, spots = basecall_tile(
        data_path, tile_coords, save_spots=False
    )
    if center is None:
        # Find the place with the highest density of spots
        x, y = spots["x"].values, spots["y"].values
        # Create a grid of potential disk centers

        x_grid, y_grid = np.meshgrid(
            np.arange(window, stack.shape[1] - window, 50),
            np.arange(window, stack.shape[0] - window, 50),
        )
        # Compute the Euclidean distance from each spot to each potential center
        distances = np.sqrt(
            (x[:, None, None] - x_grid) ** 2 + (y[:, None, None] - y_grid) ** 2
        )
        # Count the number of spots within a 200px radius for each potential center
        counts = np.sum(distances <= 100, axis=0)

        center = np.unravel_index(counts.argmax(), counts.shape)
        center = (x_grid[center], y_grid[center])
        center = np.array(center)
    lims = np.vstack([center - window, center + window]).astype(int)
    lims = np.clip(lims, 0, np.array([stack.shape[1], stack.shape[0]]) - 1)
    stack_part = get_stack_part(stack, lims[:, 0], lims[:, 1])

    return stack_part


def plot_selected_rounds(
    axes,
    stack_part,
    selected_rounds=[1, 6, 10],
    fontsize=14,
    vmin=None,
    vmax=None,
):
    """
    Here `axes` is an array/list of Axes, one for each round you want to plot.
    """
    channel_colors = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])

    for ax, iround in zip(axes, selected_rounds):
        rgb_stack = round_to_rgb(
            stack_part,
            iround - 1,
            extent=None,
            channel_colors=channel_colors,
            vmin=vmin,
            vmax=vmax,
        )
        # rotate stack 90 degrees left
        rgb_stack = np.rot90(rgb_stack, 1, (0, 1))
        ax.imshow(rgb_stack)
        # put title on bottom
        ax.set_title(f"Round {iround}", fontsize=fontsize, y=-0.25)
        if iround == 1:
            add_bases_legend(channel_colors, ax.transAxes, fontsize=fontsize)

        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")


def make_downsampled_rgb(
    processed_path: Path,
    downsample_factor: int = 20,
    channel_colors: list = ([0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0]),
    vmax: tuple = (15000, 6000, 6000, 6000),
    vmin: tuple = (700, 200, 200, 200),
):
    """
    Create a downsampled RGB image from the processed data.

    Args:
        processed_path (Path): path to processed data.
        downsample_factor (int, optional): Downsampling factor. Defaults to 20.
        channel_colors (list, optional): RGB colors. Defaults to ([0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0]).
        vmax (tuple, optional): vmax per channel. Defaults to (15000,6000,6000,6000).
        vmin (tuple, optional): vmin per channel. Defaults to (700,200,200,200).

    Returns:
        rgb: Downsampled RGB image.
    """
    stack = load_stack(processed_path)
    small_stack = None
    for ch in range(stack.shape[-1]):
        small_stack_ch = block_reduce(stack[:, :, ch], downsample_factor, np.max)
        if small_stack is None:
            small_stack = np.zeros(
                small_stack_ch.shape + (stack.shape[-1],), dtype="uint16"
            )
        small_stack[:, :, ch] = small_stack_ch.astype("uint16")

    rgb = to_rgb(
        small_stack,
        colors=channel_colors,
        vmax=vmax,
        vmin=vmin,
    )
    # rotate 90 degrees left
    rgb = np.rot90(rgb, 1, (0, 1))

    return rgb


def add_scalebar(
    ax,
    downsample_factor,
    length_um,
    *,
    pixel_size_um=0.231,  # µm per raw pixel
    bar_height_px=10,  # bar thickness in display pixels
    margin_px=15,  # margin from right & bottom in display pixels
    color="white",
):
    """Draw a horizontal scalebar of physical length *length_um* (µm)."""
    # physical length to data pixels
    disp_px_size_um = pixel_size_um * downsample_factor  # um per data pixel
    length_data_px = length_um / disp_px_size_um  # data-px

    # renderer size adjustment
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer)
    ax_w, ax_h = bbox.width, bbox.height  # screen-px

    # data-px to screen-px
    xlim = ax.get_xlim()
    W_data = abs(xlim[1] - xlim[0])  # data-px width
    length_screen_px = length_data_px * ax_w / W_data  # screen-px

    # px to Axes-fraction
    length_frac = length_screen_px / ax_w
    height_frac = bar_height_px / ax_h
    margin_x_frac = margin_px / ax_w
    margin_y_frac = margin_px / ax_h

    # draw rectangle
    x0 = 1.0 - margin_x_frac - length_frac  # right-hand side
    y0 = margin_y_frac  # bottom margin

    rect = patches.Rectangle(
        (x0, y0),
        length_frac,
        height_frac,
        transform=ax.transAxes,
        linewidth=0,
        facecolor=color,
        edgecolor=color,
        clip_on=False,
        zorder=10,
    )
    ax.add_patch(rect)


def downsample_xy(rgb_img: np.ndarray, factor: int = 5) -> np.ndarray:
    """
    Down-sample an (H, W, C) RGB image in x and y by an integer *factor*
    without interpolation, using np.max over each block.

    Args:
        rgb_img : ndarray  (H, W, 3)  or (H, W, 4)  uint8 / float
        factor  : int      down-sampling factor in x and y (default 5)
    Returns:
        ndarray (H/f, W/f, C)  same dtype as input
    """
    if factor <= 1:
        return rgb_img
    reducer_shape = (factor, factor, 1)
    return block_reduce(rgb_img, reducer_shape, np.max)


def print_image_stats(name, img, *, pixel_size_um, downsample_factor):
    """
    Print width of `img` in pixels and µm plus the effective pixel size.

    Args:
        name : str
            Friendly label (e.g. "rab").
        img : np.ndarray
            RGB image (HxWx3).
        pixel_size_um : float
            Raw microscope pixel size (µm / raw-pixel).
        downsample_factor : int
            How many raw pixels → one displayed pixel.
    """
    disp_px_size_um = pixel_size_um * downsample_factor  # µm per displayed pixel
    H, W = img.shape[:2]
    width_um = W * disp_px_size_um
    print(
        f"[{name:8s}]  width = {W:5d} px   |   "
        f"pixel size = {disp_px_size_um:6.3f} µm/px   |   "
        f"width = {width_um:8.1f} µm"
    )
