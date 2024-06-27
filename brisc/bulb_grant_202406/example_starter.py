# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
import pandas as pd
import iss_preprocess as issp
import iss_analysis as issa
import matplotlib.cm as cm
from skimage.transform import downscale_local_mean

# %%
project = "becalia_rabies_barseq"
mouse = "BRAC8498.3e"

error_correction_ds_name = "BRAC8498.3e_error_corrected_barcodes_16"
# %% reload rabies

(
    rab_spot_df,
    rab_cells_barcodes,
    rab_cells_properties,
) = issa.segment.get_barcode_in_cells(
    project,
    mouse,
    error_correction_ds_name,
    valid_chambers=None,
    save_folder=None,
    verbose=True,
)
# find starter
starters_positions = issa.io.get_starter_cells(project, mouse)
rabies_cell_properties = issa.segment.match_starter_to_barcodes(
    project,
    mouse,
    rab_cells_properties,
    rab_spot_df,
    starters=starters_positions,
    redo=False,
)
rabies_cell_properties.head()
# find barcodes with unique starter
starter_properties = rabies_cell_properties[rabies_cell_properties["starter"]]
starter_barcodes = rab_cells_barcodes.loc[starter_properties.index]
n_starter_per_barcode = (starter_barcodes.values > 0).sum(axis=0)
valid_barcodes = starter_barcodes.columns[n_starter_per_barcode == 1]
print(f"Number of starters: {len(starter_properties)}")
print(
    f"Number of barcodes with starter: {np.sum(n_starter_per_barcode>0)}/{len(n_starter_per_barcode)}"
)
print(f"Number of barcodes with unique starter: {len(valid_barcodes)}")

# %%
import numpy as np
import matplotlib.pyplot as plt

n_cell_per_barcode = (rab_cells_barcodes > 0).sum(axis=0)

# %%

# Select a candidate example
# it must be a unique starter with at least 50 cells
example_barcodes = valid_barcodes[n_cell_per_barcode.loc[valid_barcodes] > 50]
print(
    f"Number of barcodes with unique starter and at least 50 cells: {len(example_barcodes)}"
)

# %%
# Pick the chamber/roi with the most examples
example_starters_barcodes = starter_barcodes[
    starter_barcodes[example_barcodes].any(axis=1)
]
example_starters_properties = rabies_cell_properties.loc[
    example_starters_barcodes.index
]
best_rois = (
    example_starters_properties.groupby(["chamber", "roi"])
    .size()
    .sort_values(ascending=False)
)

# %%
best_roi = best_rois.index[1]
chamber, roi = best_roi
print(f"Plotting chamber {chamber} roi {roi}")
# %%
# read genes spots
data_path = f"{project}/{mouse}/{chamber}"
processed_path = issp.io.get_processed_path(data_path)
genes_per_cell = pd.read_pickle(processed_path / "cells" / f"genes_df_roi{roi}.pkl")
genes_spot_df = pd.read_pickle(processed_path / f"genes_round_spots_{roi}.pkl")
print(f"Loaded {len(genes_spot_df)} gene spots in {len(genes_per_cell)} cells")
# %%
# Plot the reference DAPI and mCherry
manual_click = (
    issp.io.get_processed_path(f"{project}/{mouse}/{chamber}") / "manual_starter_click"
)
dapi = manual_click / f"{mouse}_{chamber}_{roi}_reference.tif"
dapi_img = tifffile.imread(dapi)
mch = manual_click / f"{mouse}_{chamber}_{roi}_mCherry.tif"
mch_img = tifffile.imread(mch)
st = np.dstack([dapi_img, mch_img])

print("Downsampling")

ds_st = downscale_local_mean(st, factors=(10, 10, 1))
print("To rgb")
rgb = issp.vis.to_rgb(
    ds_st,
    colors=[(0, 0, 1), (1, 0, 0), (0, 0, 0)],
    vmin=[0, 0, 0],
    vmax=[200, 300, 100],
)
print("Plotting")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(np.swapaxes(rgb, 0, 1), origin="lower")
xlim = (500, 2400)
ylim = (800, 1700)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_axis_off()

# %%
# Plot the rabies spots raw data
rabies_img = manual_click / f"{mouse}_{chamber}_{roi}_rab.tif"
rabies_img = tifffile.imread(rabies_img)
print(rabies_img.shape)
ds_st = downscale_local_mean(rabies_img, factors=(10, 10, 1))
rgb = issp.vis.to_rgb(
    ds_st,
    colors=[(1, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1)],
    vmin=[0.02, 0.02, 0.02, 0.02],
    vmax=np.array([4, 2, 2.5, 2.5]) * 1.5,
)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(np.swapaxes(rgb, 0, 1), origin="lower")

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_axis_off()

# %%
# Plot the hyb spots raw data
hyb_img = manual_click / f"{mouse}_{chamber}_{roi}_hyb.tif"
hyb_img = tifffile.imread(hyb_img)
print(hyb_img.shape)
ds_st = downscale_local_mean(hyb_img, factors=(10, 10, 1))
# dodger or cornflower
rgb = issp.vis.to_rgb(
    ds_st,
    colors=[(1, 0, 0), (1, 0.5, 0), (0, 1, 1), (0, 0.3, 0.8)],
    vmin=[0, 0, 0, 20],
    vmax=[1500, 100, 2000, 40],
)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(np.swapaxes(rgb, 0, 1), origin="lower")

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_axis_off()
# %%
# Plot the genes spots raw data
genes_img = manual_click / f"{mouse}_{chamber}_{roi}_genes.tif"
genes_img = tifffile.imread(genes_img)
print(genes_img.shape)
ds_st = downscale_local_mean(genes_img, factors=(10, 10, 1))

ds_st = genes_img[::10, ::10, :]
rgb = issp.vis.to_rgb(
    ds_st,
    colors=[(1, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1)],
    vmin=[0, 0, 0, 0],
    vmax=[2] * 4,
)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(np.swapaxes(rgb, 0, 1), origin="lower")

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_axis_off()

# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_aspect("equal")
ax.set_title(f"Chamber {chamber} roi {roi}")
# for background, use all assigned rabies spot of that chamber
bg = rab_spot_df[
    (rab_spot_df["chamber"] == chamber)
    & (rab_spot_df["roi"] == roi)
    & (rab_spot_df["cell_mask"] > 0)
]
ax.scatter(bg.y / 10, bg.x / 10, c="k", s=1, label="All rabies rolonies", alpha=0.2)

# plot the example starters of that roi
colors = cm.get_cmap("Set1").colors
ex_st = example_starters_properties[
    (example_starters_properties["chamber"] == chamber)
    & (example_starters_properties["roi"] == roi)
]
for i, (cell_id, st) in enumerate(ex_st.iterrows()):
    barcode = starter_barcodes.loc[cell_id].idxmax()
    ax.scatter(
        st.y / 10, st.x / 10, color=colors[i], s=100, marker="*", label=barcode, ec="k"
    )
    sp = bg[bg.corrected_bases == barcode]
    ax.scatter(sp.y / 10, sp.x / 10, color=colors[i], s=5, alpha=0.5)

ax.legend(loc="upper left", ncol=2)
ax.set_xlim(xlim)
ax.set_ylim(ylim)


# ax.set_axis_off()

# %%
# plot gene calls

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

# %%
if False:
    for gene, df in genes_spot_df.groupby("gene"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.set_title(gene)
        spots = genes_spot_df[genes_spot_df.gene == gene]
        ax.scatter(
            spots.y / 10,
            spots.x / 10,
            s=1,
            color="white",
            marker="o",
            label=gene,
            alpha=0.2,
        )
# %%

LAYER_GENES = dict(
    Pde1a=("tomato", "o"),
    Cux2=("gold", "o"),
    Ptn=("orangered", "o"),
    Lypd1=("firebrick", "o"),
    Dgkb=("yellowgreen", "o"),
    Lamp5=("forestgreen", "o"),
    Luzp2=("lime", "o"),
    Necab1=("teal", "o"),
    Nov=("greenyellow", "o"),
    Rgs4=("cornflowerblue", "o"),
    Rorb=("dodgerblue", "o"),
    # Rbp4=('deepskyblue', 'o'),
    # Serpine2=('blue', 's'),
    Spock3=("aquamarine", "o"),
    Fezf2=("royalblue", "o"),
    Igfbp4=("crimson", "o"),
    Foxp2=("darkviolet", "o"),
    Pcp4=("darkmagenta", "o"),
    # Crym=('cornflowerblue', 'o'),
    Ctgf=("darkorchid", "o"),
)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_facecolor("black")
ax.set_aspect("equal")
zorder = 100
i = 0
for gene, (col, mark) in LAYER_GENES.items():
    if False:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.set_title(gene)
    spots = genes_spot_df[genes_spot_df.gene == gene]
    if len(spots) == 0:
        continue
    nspot = len(spots)
    alpha = min(0.5, 3e3 / nspot)
    ax.scatter(
        spots.y / 10,
        spots.x / 10,
        s=0.5,
        color=col,
        marker=mark,
        label=gene,
        alpha=alpha,
        zorder=zorder,
    )
    zorder = 100 if i % 2 else 50
    i += 1
leg = ax.legend(loc="upper left", ncol=2)
for lh in leg.legendHandles:
    lh.set_alpha(1)
    lh.set_sizes([10])

ax.set_title(f"Chamber {chamber} roi {roi}, gene {gene}")
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# %%

plot_colortable(mcolors.CSS4_COLORS)
plt.show()
# %%
