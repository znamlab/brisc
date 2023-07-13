"""Script to generate panels for the local CRE confocal example."""

# %% imports
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from czifile import CziFile
import flexiznam as flz
from iss_preprocess import vis
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# %% Define parameters
PROJECT = "rabies_barcoding"
MOUSE = "BRYC64.2h"
IMAGE_FILE = "Slide_3_section_1.czi"
ROTATION = 103  # rotation in degrees to put pia at the top of the image
OVERVIEW = [1000, 2600, 5600, 6400]  # bbox for overview image after rotation
INSET = [2500, 4600, 3100, None]  # bbox for inset image after rotation, the last value
# is calculated
aspect_ratio = (OVERVIEW[2] - OVERVIEW[0]) / (OVERVIEW[3] - OVERVIEW[1])
INSET[-1] = int(INSET[1] + (INSET[2] - INSET[0]) / aspect_ratio * 2)
print(INSET)

# Note that this slice is flipped on the slide, so need to mirror it

# %% Get the data
root_folder = Path(flz.PARAMETERS["data_root"]["processed"]) / PROJECT
output_folder = root_folder / "figures"
output_folder.mkdir(exist_ok=True)
confocal_data = root_folder / MOUSE / "zeiss_confocal"

with CziFile(confocal_data / IMAGE_FILE) as czi:
    metadata = czi.metadata(raw=False)
    img = np.squeeze(czi.asarray())

scale = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
scale = {s["Id"]: s["Value"] * 1e6 for s in scale}

# %% Define helper functions
def rotate(image, angle, center=None, scale=1.0, flip=True):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    if flip:
        rotated = cv2.flip(rotated, 1)
    return rotated

# %% Do the plot

inset_line_props = dict(color="grey", linestyle="dashed", linewidth=1)
fig = plt.figure(figsize=[5, 5 * aspect_ratio / 2.05])
axes = [plt.subplot2grid((2, 2), (0, 0), rowspan=2)] + [
    plt.subplot2grid((2, 2), (i, 1)) for i in range(2)
]

zplanes = [3, 3, 5]
for iax, part in enumerate([OVERVIEW, INSET, INSET]):
    ax = axes[iax]
    if iax == 0:
        factor = 4
        rec = np.array(INSET)
        rec[[0, 2]] -= OVERVIEW[0]
        rec[[1, 3]] -= OVERVIEW[1]
        rec = rec // factor
        rec = plt.Rectangle(
            rec[[1, 0]],
            rec[3] - rec[1],
            rec[2] - rec[0],
            fill=False,
            **inset_line_props,
        )
        xyA1 = [(INSET[1] - OVERVIEW[1]) // factor, (INSET[0] - OVERVIEW[0]) // factor]
        xyA2 = [(INSET[1] - OVERVIEW[1]) // factor, (INSET[2] - OVERVIEW[0]) // factor]
        if False:
            for ax_corn, xyA in enumerate([xyA1, xyA2]):
                con = ConnectionPatch(
                    xyA=xyA,
                    xyB=[0, 1 - ax_corn],
                    coordsA=axes[0].transData,
                    coordsB=axes[1].transAxes,
                    **inset_line_props,
                )
                axes[1].add_artist(con)
    else:
        factor = 1
        rec = None
    lim = np.array(part) // factor
    z = zplanes[iax]
    stack = np.dstack(
        [
            rotate(np.nanmean(i, axis=0), ROTATION)[lim[0] : lim[2], lim[1] : lim[3]]
            for i in [
                img[1, z : z + 1, ::factor, ::factor],
                img[1, z : z + 1, ::factor, ::factor],
                img[0, z : z + 1, ::factor, ::factor],
            ]
        ]
    )
    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rgb = vis.to_rgb(stack, colors, vmax=[1000, 10000, 5000], vmin=None)
    ax.imshow(rgb, vmin=0, vmax=5000)
    if rec is not None:
        ax.add_patch(rec)
    # ax.set_yticklabels([str(int(i * factor) + part[0]) for i in ax.get_yticks()])
    # ax.set_xticklabels([str(int(i * factor) + part[1]) for i in ax.get_xticks() ])

    bar_length = 100 if iax == 0 else 20
    scalebar = AnchoredSizeBar(
        ax.transData,
        bar_length / scale["X"] / factor,
        label=None,
        loc="lower right",
        label_top=True,
        color="white",
        frameon=False,
        size_vertical=10,
    )

    ax.add_artist(scalebar)
    ax.set_xticks([])
    ax.set_yticks([])
starters = [np.array([390, 200]), np.array([650, 480])]
for iax, ax in enumerate(axes[1:]):
    for starter in starters:
        ax.annotate(
            "",
            xy=starter,
            xytext=starter + np.array([-75, -75]),
            arrowprops=dict(
                facecolor="white", shrink=0.05, edgecolor="none", width=2, headwidth=10
            ),
        )
fig.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0.01, hspace=0.01)

# %% Save


fig.savefig(output_folder / f"{MOUSE}_local_cre_example_panels.png", dpi=300)
# %%
