import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import ccf_streamlines.projection as ccfproj

import flexiznam as flz


ARA_PROJECTION = "flatmap_dorsal"
ATLAS_SIZE = 10


def get_avg_layer_depth():
    project_folder = Path(flz.PARAMETERS["data_root"]["processed"]).parent
    ccf_streamlines_folder = project_folder / "resources" / "ccf_streamlines"
    with open(ccf_streamlines_folder / "avg_layer_depths.json", "r") as f:
        layer_tops = json.load(f)
    return layer_tops


def get_projector(ara_projection=ARA_PROJECTION):
    """Get a projector object to project coordinates to the flat map

    Returns:
        ccf_coord_proj: a projector object
    """
    project_folder = Path(flz.PARAMETERS["data_root"]["processed"]).parent
    ccf_streamlines_folder = project_folder / "resources" / "ccf_streamlines"

    layer_tops = get_avg_layer_depth()
    layer_borders = np.hstack([0, np.sort(np.hstack(list(layer_tops.values())))])
    layer_thicknesses = {
        f"Isocortex layer {lay}": float(th)
        for lay, th in zip(["1", "2/3", "4", "5", "6a", "6b"], np.diff(layer_borders))
    }
    # Get a projector object
    ccf_coord_proj = ccfproj.IsocortexCoordinateProjector(
        projection_file=ccf_streamlines_folder / f"{ara_projection}.h5",
        surface_paths_file=ccf_streamlines_folder / "surface_paths_10_v3.h5",
        closest_surface_voxel_reference_file=ccf_streamlines_folder
        / "closest_surface_voxel_lookup.h5",
        streamline_layer_thickness_file=ccf_streamlines_folder
        / "cortical_layers_10_v2.h5",
        layer_thicknesses=layer_thicknesses,
    )

    return ccf_coord_proj


def compute_flatmap_coors(
    df,
    col_prefix="ara_",
    col_suffix="",
    projection="flatmap_dorsal",
    distance_cutoff=150,
    thickness_type="unnormalized",
):
    """Compute coordinates on ARA flatmap

    Args:
        df: dataframe with coordinates
        col_prefix: prefix for column names. Default: "ara_".
        col_suffix: suffix for column names. Default: "".
        projection: projection to use. Default: "flatmap_dorsal"
        distance_cutoff (float): Points for which projection is not defined but that
            have a valid point `distance_cutoff` (in um) away for which it is defined,
            will be projected like their closest valid neighbour. Slow. Put to 0 to skip
        thickness_type (str): Normalisation of Z axis for ccf projector, one of
            "unnormalized", "normalized_full", "normalized_layers". Default: "unnormalized"



    Returns:
        flat_coors: coordinates on ARA flatmap
    """
    if thickness_type not in ["unnormalized", "normalized_full", "normalized_layers"]:
        raise ValueError(f"Unknown thickness_type: {thickness_type}")

    # make sure we do not change anything in place
    df = df.copy()
    columns = [f"{col_prefix}{i}{col_suffix}" for i in ["x", "y", "z"]]
    ori_coors = df[columns].values
    flat_coors = np.zeros_like(ori_coors)
    bad = np.any(ori_coors <= 0, axis=1)
    # also remove NaN
    bad = np.logical_or(bad, np.any(np.isnan(ori_coors), axis=1))
    if np.any(bad):
        print(f"Found {np.sum(bad)} bad coordinates")
        flat_coors[bad, :] = np.nan
    ccf_coord_proj = get_projector(projection)
    if projection == "flatmap_dorsal":
        view_space = "flatmap_dorsal"
    elif projection == "top":
        view_space = False
    else:
        print(f"Warning unknown projections: {projection}.")
        view_space = False
    flat_coors[~bad, :] = ccf_coord_proj.project_coordinates(
        ori_coors[~bad, :] * 1000,
        drop_voxels_outside_view_streamlines=False,
        view_space_for_other_hemisphere=view_space,
        hemisphere="right",
        thickness_type=thickness_type,
    )
    if distance_cutoff is None or distance_cutoff <= 0:
        if thickness_type == "normalized_layers":
            # Until https://github.com/AllenInstitute/ccf_streamlines/issues/10 is fixed
            # we need to renormalise
            layer_tops = get_avg_layer_depth()
            norm_factor = layer_tops["wm"] / 2000.00
            flat_coors[:, 2] *= norm_factor

        return flat_coors

    # Not in streamlines, the point is below the bottom of the cortex
    to_move = np.isnan(flat_coors[:, 0]) & (~bad) & (~df.cortical_area.isna())
    print(f"Found {np.sum(to_move)} points in cortex with no flatmap coordinates")
    # Get voxels for which projection is possible
    valid_voxels = ccf_coord_proj.closest_surface_voxels[:, 0]
    valid_voxels = np.vstack(
        np.unravel_index(valid_voxels, np.array(ccf_coord_proj.volume_shape))
    ).T
    # Find voxels of points to move
    ori_voxel = (ori_coors[to_move, :] * 1000 / ccf_coord_proj.resolution).astype(int)
    z_size = ccf_coord_proj.volume_shape[2]
    z_midline = z_size / 2
    # reflect to left hemisphere
    to_reflect = ori_voxel[:, 2] > z_midline
    ori_voxel[to_reflect, 2] = z_size - ori_voxel[to_reflect, 2]
    # limit the search around the part of the atlas we have
    min_v, max_v = ori_voxel.min(axis=0), ori_voxel.max(axis=0)
    dst_cutoff = distance_cutoff / ccf_coord_proj.resolution[0]
    lower_bounds = min_v - dst_cutoff
    upper_bounds = max_v + dst_cutoff
    is_within_bounds = np.all(
        (valid_voxels >= lower_bounds) & (valid_voxels <= upper_bounds), axis=1
    )
    valid_voxels = valid_voxels[is_within_bounds, :]
    distances, closest_voxels = np.zeros(ori_voxel.shape[0]), np.zeros_like(ori_voxel)
    for i, voxel in tqdm(
        enumerate(ori_voxel),
        total=ori_voxel.shape[0],
        desc="Find closest voxel that can be projected to flatmap",
    ):
        # keep valid_voxels in a dst_cutoff cube around voxel
        upper = voxel + dst_cutoff
        lower = voxel - dst_cutoff
        is_within = np.all((valid_voxels >= lower) & (valid_voxels <= upper), axis=1)
        vvox = valid_voxels[is_within, :]
        dst = np.linalg.norm(vvox - voxel, axis=-1)
        if not len(dst):
            distances[i] = np.inf
            continue
        distances[i] = dst.min()
        closest_voxels[i] = vvox[dst.argmin(), :]
    close_enough = distances < dst_cutoff
    print(f"Found {np.sum(close_enough)}/{ori_voxel.shape[0]} close enough")
    # put them back in the right hemisphere if needed
    closest_reflected = closest_voxels.copy()
    closest_reflected[to_reflect, 2] = z_size - closest_reflected[to_reflect, 2]
    # and reproject the one that are close enough (keep the rest to NaN)
    reprojected = np.empty(closest_voxels.shape) + np.nan
    reprojected[close_enough, :] = ccf_coord_proj.project_coordinates(
        closest_reflected[close_enough, :] * ccf_coord_proj.resolution[0],
        drop_voxels_outside_view_streamlines=False,
        view_space_for_other_hemisphere=view_space,
        hemisphere="right",
        thickness_type=thickness_type,
    )
    flat_coors[to_move, :] = reprojected

    if thickness_type == "normalized_layers":
        # Until https://github.com/AllenInstitute/ccf_streamlines/issues/10 is fixed
        # we need to renormalise by the longest path, which 200 voxels
        layer_tops = get_avg_layer_depth()
        norm_factor = layer_tops["wm"] / 2000.0
        flat_coors[:, 2] *= norm_factor
    return flat_coors
