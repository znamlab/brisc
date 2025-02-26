# Script version of rabies_all_cells_plots notebook to run for all bc
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ccf_streamlines.projection as ccfproj
import bg_atlasapi as bga
import flexiznam as flz

from cricksaw_analysis import atlas_utils
from iss_preprocess.pipeline.stitch import load_tile_ref_coors
from iss_preprocess.pipeline.segment import get_cell_masks
from iss_preprocess.io.load import load_ops
from iss_preprocess.vis.diagnostics import plot_all_rounds
from iss_preprocess.vis.vis import to_rgb
from iss_preprocess.call import BASES
from znamutils import slurm_it


project = "becalia_rabies_barseq"
mouse = "BRAC8498.3e"

ARA_PROJECTION = "flatmap_dorsal"
ATLAS_SIZE = 10
# We can highlight some areas, we define that has a dict with the area name as key
# and the color as value
area_colors = {
    "VISp": "green",
    "VISpm": "blue",
    "VISam": "red",
    "VISal": "orange",
    "VISl": "purple",
    "VISrl": "yellow",
    "VISa": "cyan",
    "VISli": "pink",
    "VISpor": "bisque",
    "AUDp": "teal",
    "AUDd": "magenta",
    "AUDpo": "brown",
    "AUDv": "orchid",
    "TEa": "royalblue",
}

CTX_TABLE = atlas_utils.create_ctx_table()


def find_l1_pixels(atlas_img, ctx_table=CTX_TABLE):
    """Return the coordinates of the surface pixel in cortex

    Args:
        atlas_img: 2D array with the atlas image
        ctx_table: table with the cortex regions

    Returns:
        dv, ml: the dorsal and mediolateral coordinates of the layer 1 pixels
    """
    layer1_ids = ctx_table[ctx_table["layer"] == "1"].id.values
    layer1_mask = np.isin(atlas_img, layer1_ids)
    return np.where(layer1_mask)


def get_projector(ara_projection=ARA_PROJECTION):
    """Get a projector object to project coordinates to the flat map

    Returns:
        ccf_coord_proj: a projector object
    """
    project_folder = Path(flz.PARAMETERS["data_root"]["processed"]).parent
    ccf_streamlines_folder = project_folder / "resources" / "ccf_streamlines"
    # Get a projector object
    ccf_coord_proj = ccfproj.IsocortexCoordinateProjector(
        projection_file=ccf_streamlines_folder / f"{ara_projection}.h5",
        surface_paths_file=ccf_streamlines_folder / "surface_paths_10_v3.h5",
        closest_surface_voxel_reference_file=ccf_streamlines_folder
        / "closest_surface_voxel_lookup.h5",
        streamline_layer_thickness_file=ccf_streamlines_folder
        / "cortical_layers_10_v2.h5",
    )

    return ccf_coord_proj


def get_midline(atlas_size=ATLAS_SIZE):
    """Get the midline of the atlas

    Args:
        atlas_size: size of the atlas in microns

    Returns:
        midline: the midline of the atlas in voxel
    """
    atlas = bga.bg_atlas.BrainGlobeAtlas(f"allen_mouse_{atlas_size}um")
    midline = int(np.where(np.diff(atlas.hemispheres[0, 0, :]) != 0)[0])
    return midline


def plot_line(flat_ax, atlas, cor_plane=800, hemisphere="right", **kwargs):
    """Plot a line on the flat map along a coronal plane

    Args:
        flat_ax: axis to plot the line
        atlas: atlas object
        cor_plane: coronal plane to plot the line
        hemisphere: hemisphere to plot the line
        **kwargs: additional arguments to pass to the plot

    """
    ccf_coord_proj = get_projector()
    midline = get_midline()
    line_kwargs = dict(color="grey", ls="--")
    if hemisphere == "left":
        cor_atlas = atlas.annotation[cor_plane, :, :midline]
    elif hemisphere == "right":
        cor_atlas = atlas.annotation[cor_plane, :, midline:]
    else:
        cor_atlas = atlas.annotation[cor_plane, :, :]

    line_kwargs.update(kwargs)
    dv, ml = find_l1_pixels(cor_atlas)
    if hemisphere == "right":
        ml = midline + ml
    coor_plane_coords = np.vstack((np.array([cor_plane] * len(dv)), dv, ml)).T
    subsample = max(1, len(coor_plane_coords) // 100)
    coor_plane_coords = coor_plane_coords[::subsample]
    flat_coords = ccf_coord_proj.project_coordinates(
        coor_plane_coords * ATLAS_SIZE,
        drop_voxels_outside_view_streamlines=False,
        view_space_for_other_hemisphere=ARA_PROJECTION,
        hemisphere=hemisphere,
    )
    # subsample to keep about 10 points
    ordering = np.argsort(flat_coords[:, 0])
    subsample = max(1, len(coor_plane_coords) // 100)
    flat_coords = flat_coords[ordering][::subsample]
    flat_ax.plot(flat_coords[:, 0], flat_coords[:, 1], **line_kwargs)


def plot_background(
    cor_ax,
    flat_ax,
    cor_plane=800,
    hemisphere="right",
    area_colors=area_colors,
    lines_to_add=None,
    alpha=0.1,
    atlas=None,
):
    if atlas is None:
        atlas = bga.bg_atlas.BrainGlobeAtlas(f"allen_mouse_{ATLAS_SIZE}um")

    # Coronal view
    atlas_utils.plot_coronal_view(
        cor_ax,
        cor_plane,
        atlas,
        hemisphere=hemisphere,
        area_colors=area_colors,
        alpha=alpha,
    )
    atlas_utils.plot_flatmap(
        flat_ax,
        hemisphere=hemisphere,
        area_colors=area_colors,
        alpha=alpha,
        ccf_streamlines_folder=None,
    )

    # add a line to show the plane
    plot_line(flat_ax, atlas, cor_plane=cor_plane, hemisphere=hemisphere, lw=1)
    if lines_to_add is not None:
        for line in lines_to_add:
            flat_ax.plot(*line, color="black", lw=1)

    flat_ax.set_aspect("equal")
    flat_ax.set_ylim(1380, 600)
    flat_ax.set_xlim(50, 1150)
    cor_ax.set_xlim(-10, 600)
    cor_ax.set_ylim(500, -10)


def plot_spots(cor_ax, flat_ax, df, **kwargs):
    midline = get_midline()
    cor_coors = np.vstack(
        [
            df["ara_z"].values * 1000 / ATLAS_SIZE - midline,
            df["ara_y"].values * 1000 / ATLAS_SIZE,
        ]
    )
    cor_ax.scatter(*cor_coors, **kwargs)
    flat_coords = df[["ara_x", "ara_y", "ara_z"]].values
    bad = np.any(flat_coords < 0, axis=1)
    # also remove NaN
    bad = np.logical_or(bad, np.any(np.isnan(flat_coords), axis=1))
    if np.any(bad):
        print(f"Found {np.sum(bad)} bad coordinates")
        flat_coords = flat_coords[~bad]
    ccf_coord_proj = get_projector()
    flat_coords = ccf_coord_proj.project_coordinates(
        flat_coords * 1000,
        drop_voxels_outside_view_streamlines=False,
        view_space_for_other_hemisphere=ARA_PROJECTION,
        hemisphere="right",
    )

    flat_ax.scatter(flat_coords[:, 0], flat_coords[:, 1], **kwargs)
    return cor_coors, flat_coords


def plot_starter_raw(starter, bc, fig, rnd_axes, spot_axis=None, mcherry_axis=None):
    """Plot the raw data for a starter cell

    Args:
        starter (pd.Series): the starter cell
        bc (str): the barcode to plot
        fig (plt.Figure): the figure to plot
        rnd_axes (list of plt.Axes): the axes to plot the rounds
        spot_axis (plt.Axes): the axis to plot the spots
        mcherry_axis (plt.Axes): the axis to plot the mcherry channel

    """
    chamber = starter.chamber
    roi = starter.roi
    data_path = f"{project}/{mouse}/{chamber}"

    flm_sess = flz.get_flexilims_session(project_id=project)
    ds = flz.Dataset.from_flexilims(
        name=f"{mouse}_{chamber}_barcodes_mask_assignment_roi{roi}_0",
        flexilims_session=flm_sess,
    )
    corrected_ds_name = ds.extra_attributes["error_correction_ds_name"]
    assignment = pd.read_pickle(ds.path_full)
    corrected_spots = pd.read_pickle(
        flz.Dataset.from_flexilims(
            name=corrected_ds_name, flexilims_session=flm_sess
        ).path_full
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # find the spots assigned to this mask
    spot_assigned = assignment[assignment["mask"] == int(starter.cell_id)]
    spot_df = corrected_spots.loc[spot_assigned.spot]

    tile = spot_df.tile.value_counts().idxmax()
    tile_coors = [int(i) for i in tile.split("_")]
    print(f"Loading tile {tile} for data path {data_path}")
    stack, bad_pixels = load_tile_ref_coors(
        data_path,
        tile_coors=tile_coors,
        prefix="barcode_round",
        correct_illumination=False,
    )

    print(f"Loaded stack with {stack.shape[-1]} rounds")

    channel_colors = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])
    coords = np.vstack(spot_df[["x_tile", "y_tile"]].values)
    abs_coords = np.vstack(spot_df[["x", "y"]].values)
    minabs = np.min(abs_coords, axis=0).astype(int) - 50
    maxabs = np.max(abs_coords, axis=0).astype(int) + 50
    min_c = np.min(coords, axis=0).astype(int) - 50
    max_c = np.max(coords, axis=0).astype(int) + 50
    vmin, vmax = np.percentile(stack[..., 0], (0.1, 99.99), axis=(0, 1))
    ops = load_ops(data_path)
    stack_part = stack[
        min_c[1] : max_c[1], min_c[0] : max_c[0], np.argsort(ops["camera_order"]), ...
    ]

    fig, _ = plot_all_rounds(
        stack_part,
        view=None,
        channel_colors=channel_colors,
        grid=False,
        round_labels=None,
        fig=fig,
        axes=rnd_axes,
        vmin=vmin,
        vmax=vmax,
        legend_kwargs={"fontsize": 10},
    )

    for i, ax in enumerate(rnd_axes):
        ax.set_title(None)
        base_index = list(BASES).index(bc[i])
        ax.text(
            0.9,
            0.9,
            bc[i],
            ha="center",
            va="center",
            fontsize=10,
            color=channel_colors[base_index],
            transform=ax.transAxes,
        )
        ax.axis("off")

    if spot_axis is not None:
        dapi_mask = get_cell_masks(data_path, roi=roi)
        dapi_mask = dapi_mask[minabs[1] : maxabs[1], minabs[0] : maxabs[0]].astype(
            float
        )
        dapi_mask[dapi_mask == 0] = np.nan
        std_stack = np.sum(np.nanstd(stack_part, axis=-1), axis=-1)
        spot_axis.imshow(std_stack, cmap="viridis")
        spot_axis.imshow(dapi_mask % 20, cmap="tab20", alpha=0.3, interpolation=None)
        spot_axis.scatter(
            coords[:, 0] - min_c[0],
            coords[:, 1] - min_c[1],
            ec="w",
            fc="none",
            s=20,
            marker="o",
        )
        sequences = spot_df.bases.value_counts()
        seq_txt = "\n".join([f"{k}={v}" for k, v in sequences.items()])
        spot_axis.text(
            0.1,
            0.1,
            seq_txt,
            ha="left",
            va="bottom",
            fontsize=8,
            color="white",
            transform=spot_axis.transAxes,
        )
        spot_axis.axis("off")
    if mcherry_axis is not None:
        print(f"Loading mcherry {tile} for data path {data_path}")
        full_stack, bad_pixels = load_tile_ref_coors(
            data_path,
            tile_coors=tile_coors,
            prefix="mCherry_1",
            correct_illumination=False,
            filter_r=False,
        )
        print(f"Loaded stack with {full_stack.shape[-1]}")
        mch_ch = [ops["mcherry_signal_channel"], ops["mcherry_background_channel"]]
        mcherry_stack = full_stack[min_c[1] : max_c[1], min_c[0] : max_c[0], mch_ch, 0]
        mcherry_mask = get_cell_masks(
            data_path, roi=roi, prefix="mCherry_1", curated=True
        )
        mcherry_mask = mcherry_mask[minabs[1] : maxabs[1], minabs[0] : maxabs[0]]
        print(mcherry_mask.shape)
        vmax = np.nanpercentile(mcherry_stack, 99.9)
        rgb = to_rgb(mcherry_stack, colors=[(1, 0, 0), (0, 1, 0)], vmin=0, vmax=vmax)
        mcherry_axis.imshow(rgb)
        mcherry_axis.contour(mcherry_mask % 20, cmap="tab20", alpha=0.5)
        mcherry_axis.scatter(
            coords[:, 0] - min_c[0],
            coords[:, 1] - min_c[1],
            ec="w",
            fc="none",
            s=20,
            marker="o",
        )
        mcherry_axis.axis("off")


@slurm_it(conda_env="iss-preprocess", print_job_id=True)
def plot_barcode(bc, save_folder=None, fig=None, verbose=True, plot_raw_data=True):
    """Plot the cells for a barcode

    Args:
        bc (str): the barcode to plot
        save_folder (Path): the folder to save the figure
        fig (plt.Figure): the figure to plot
        verbose (bool): whether to print information
        plot_raw_data (bool): whether to plot the raw data
    """
    if isinstance(save_folder, str):
        save_folder = Path(save_folder)
    df_file = flz.get_processed_path(f"{project}/{mouse}/analysis/cell_barcode_df.pkl")
    full_df = pd.read_pickle(df_file)
    barcoded_cells = full_df.query("main_barcode.notna()")

    exploded = barcoded_cells.all_barcodes.explode()
    cell_id_with_bc = exploded[exploded == bc].copy()
    bc_cells = full_df.loc[cell_id_with_bc.index]
    starters = bc_cells.query("is_starter == True")
    presynaptic = bc_cells.query("is_starter == False")
    nstarters = len(starters)
    if verbose:
        print(f"Found {len(presynaptic)} presynaptic cells, and {nstarters} starters.")
    cor_plane = int(np.nanmedian(bc_cells.ara_x.values) * 1000 / ATLAS_SIZE)

    if fig is None:
        fig = plt.figure()
    fig.clear()

    if plot_raw_data:
        n_rows = nstarters * 2 + 3
        cor_ax = plt.subplot2grid((n_rows, 8), (0, 0), colspan=4, rowspan=3)
        flat_ax = plt.subplot2grid((n_rows, 8), (0, 4), colspan=4, rowspan=3)
        starters_axis = {}
        for i in range(nstarters):
            axes = {}
            axes["spot_axis"] = plt.subplot2grid(
                (n_rows, 8), (i * 2 + 3, 0), colspan=1, rowspan=1
            )
            axes["mcherry_axis"] = plt.subplot2grid(
                (n_rows, 8), (i * 2 + 3 + 1, 0), colspan=1, rowspan=1
            )
            axes["rnd_axes"] = [
                plt.subplot2grid((n_rows, 8), (i * 2 + 3 + k, 1 + j))
                for k in range(2)
                for j in range(7)
            ]

            starters_axis[i] = axes
        fig.set_size_inches(18, 2.5 * n_rows)
    else:
        cor_ax = fig.add_subplot(1, 2, 1)
        flat_ax = fig.add_subplot(1, 2, 2)
    plot_background(cor_ax, flat_ax, cor_plane=cor_plane, alpha=0.1)
    if len(presynaptic) > 0:
        plot_spots(
            cor_ax,
            flat_ax,
            presynaptic,
            c="royalblue",
            s=10,
            marker="o",
            alpha=0.8,
            zorder=10,
        )
    else:
        if verbose:
            print("No presynaptic cells found.")
    c, f = plot_spots(
        cor_ax, flat_ax, starters, c="darkred", s=50, marker="*", alpha=0.8, zorder=20
    )
    flat_ax.axis("off")
    cor_ax.axis("off")
    supt = f"Barcode {bc} with {len(presynaptic)} presynaptic cells and {len(starters)} starters"
    fig.suptitle(supt)

    if plot_raw_data:
        for istarter in range(nstarters):
            starter = starters.iloc[istarter]
            plot_starter_raw(starter, bc, fig, **starters_axis[istarter])
    fig.tight_layout()
    if save_folder is not None:
        save_folder.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_folder / f"barcode_{bc}.png")
        print(f"Saved figure to {save_folder / f'barcode_{bc}.png'}")
    print(f"Done plotting barcode {bc}")
