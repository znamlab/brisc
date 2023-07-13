"""Generate panel with coronal view of data cricksaw data matching the
confocal view.
"""

# %%
# Imports

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import cv2
import bg_atlasapi as bga

import flexiznam as flz
from flexiznam.schema import Dataset
from cricksaw_analysis import atlas_utils
from brisc.starter_cell_density import coronal_view_utils as cvu
import cricksaw_analysis.io
from iss_preprocess import vis
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# %%
# Define parameters
PROJECT = "rabies_barcoding"
MOUSE = "BRYC64.2h"
atlas_size = 10

OVERVIEW = np.array([200, 1900, 680, 2250])
Zs = [661, 662, 663]
INSET = np.array(
    [352, 2125, 415, None]
)  # bbox for inset image after rotation, the last value
# is calculated
aspect_ratio = (OVERVIEW[2] - OVERVIEW[0]) / (OVERVIEW[3] - OVERVIEW[1])
INSET[-1] = int(INSET[1] + (INSET[2] - INSET[0]) / aspect_ratio * 2)
print(INSET)

# %%
# Get atlas and flexilims info
atlas_name = "allen_mouse_%dum" % atlas_size
bg_atlas = bga.bg_atlas.BrainGlobeAtlas(atlas_name)

raw = Path(flz.PARAMETERS["data_root"]["raw"])
processed = Path(flz.PARAMETERS["data_root"]["processed"])
flm_sess = flz.get_flexilims_session(project_id=PROJECT)
root_folder = processed / PROJECT
output_folder = root_folder / "figures"
output_folder.mkdir(exist_ok=True)
cricksaw_ds = flz.Dataset.from_flexilims(
    name=f"{MOUSE}_cricksaw_0", flexilims_session=flm_sess
)

# %%
# define helper functions


def load_slice(image_z, vmax=[10000, 30000, 500]):
    n_planes = cricksaw_ds.extra_attributes["scannersettings"]["numOpticalSlices"]
    slice_n = int(image_z / n_planes) + 1
    plane_n = int(image_z - (slice_n - 1) * n_planes) + 1
    print(f"Image Z {image_z} is slice {slice_n} plane {plane_n}")
    data_folder = cricksaw_ds.path_full.parent / "stitchedImages_050"
    data = []
    for channel in [3, 2, 1]:
        img_file = (
            data_folder / str(channel) / f"section_{slice_n:03d}_{plane_n:02d}.tif"
        )
        assert img_file.is_file(), f"File {img_file} does not exist"
        image = np.asarray(PIL.Image.open(img_file))
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        # perform the rotation
        rot_scale = 1
        angle = -15
        M = cv2.getRotationMatrix2D(center, angle, rot_scale)
        image = cv2.warpAffine(image, M, (w, h))
        data.append(image)
    data = np.stack(data, axis=2)
    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rgb = vis.to_rgb(data[:, :, [0, 0, 2]], colors, vmax=vmax, vmin=None)
    return rgb


# %%
# Do the plot

scale = cricksaw_ds.extra_attributes["voxelsize"]

inset_line_props = dict(color="grey", linestyle="dashed", linewidth=1)
fig = plt.figure(figsize=[5, 5 * aspect_ratio / 2.05])
axes = [plt.subplot2grid((2, 2), (0, 0), rowspan=2)] + [
    plt.subplot2grid((2, 2), (i, 1)) for i in range(2)
]

zplanes = [3, 3, 5]
for iax, part in enumerate([OVERVIEW, INSET, INSET]):
    ax = axes[iax]
    vmax = [5000, 15000, 300] if iax == 0 else [5000, 15000, 300]
    rgb = load_slice(Zs[iax], vmax=vmax)
    if iax == 0:
        rec = np.array(INSET)
        rec[[0, 2]] -= OVERVIEW[0]
        rec[[1, 3]] -= OVERVIEW[1]
        rec = plt.Rectangle(
            rec[[1, 0]],
            rec[3] - rec[1],
            rec[2] - rec[0],
            fill=False,
            **inset_line_props,
        )
        xyA1 = [(INSET[1] - OVERVIEW[1]), (INSET[0] - OVERVIEW[0])]
        xyA2 = [(INSET[1] - OVERVIEW[1]), (INSET[2] - OVERVIEW[0])]
    else:
        rec = None

    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ax.imshow(rgb[part[0] : part[2], part[1] : part[3]])
    if rec is not None:
        ax.add_patch(rec)
    # ax.set_yticklabels([str(int(i * factor) + part[0]) for i in ax.get_yticks()])
    # ax.set_xticklabels([str(int(i * factor) + part[1]) for i in ax.get_xticks() ])

    bar_length = 100 if iax == 0 else 20
    scalebar = AnchoredSizeBar(
        ax.transData,
        bar_length / scale["X"],
        label=None,
        loc="lower right",
        label_top=True,
        color="white",
        frameon=False,
        size_vertical=8 if iax == 0 else 2,
    )

    ax.add_artist(scalebar)
    ax.set_xticks([])
    ax.set_yticks([])
starters = [np.array([25, 20]), np.array([52, 50])]
for iax, ax in enumerate(axes[1:]):
    for starter in starters:
        ax.annotate(
            "",
            xy=starter,
            xytext=starter + np.array([-5, -5]),
            arrowprops=dict(
                facecolor="white", shrink=0.05, edgecolor="none", width=2, headwidth=10
            ),
        )

fig.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0.01, hspace=0.01)

# %%
# Save the figure
fig.savefig(output_folder / f"{MOUSE}_local_cre_example_panels_brainsaw.png", dpi=300)


# %%
