from iss_preprocess.diagnostics.diag_stitching import plot_single_overview
from iss_preprocess.vis import round_to_rgb
from iss_preprocess.vis import add_bases_legend
from iss_preprocess.vis.utils import get_stack_part
from iss_preprocess.pipeline.sequencing import basecall_tile
import matplotlib.image as mpimg

import numpy as np


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
        ax.imshow(rgb_stack)
        # put title on bottom
        ax.set_title(f"Round {iround}", fontsize=fontsize, y=-0.45)
        if iround == 1:
            add_bases_legend(channel_colors, ax.transAxes, fontsize=fontsize)

        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
