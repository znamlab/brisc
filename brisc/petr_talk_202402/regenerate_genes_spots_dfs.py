"""
Script to regenerate the genes spots df but removing duplicates differently

Here we keep only points who have most of the their neighbours in the same tile.
"""
# %%
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from flexiznam.config import PARAMETERS
from pathlib import Path
from itertools import cycle
from matplotlib.animation import FuncAnimation

import iss_preprocess as iss

mouse = "BRAC8501.6a"
chamber = "chamber_07"
data_path = f"becalia_rabies_barseq/{mouse}/{chamber}/"
# data_path = 'becalia_rabies_barseq/BRYC64.2j/chamber_12/'

processed_path = iss.io.load.get_processed_path(data_path)
metadata = iss.io.load_metadata(data_path)
ops = iss.io.load_ops(data_path=data_path)
print(data_path)

# %%
# Remake genes spot dataframe but keeping duplicates
from scipy.spatial import KDTree
from collections import Counter


def find_nearest_neighbours(points, ids, k):
    # slowish takes about 2minutes per ROI
    # Create a KDTree instance
    tree = KDTree(points)

    # For each point, find the k-1 nearest neighbours (-1 because point is included)
    # Limit the search in 50px radius around the point.
    nearest_neighbours_id = np.zeros_like(ids)
    for i in range(len(points)):
        dist, idx = tree.query(points[i], k, distance_upper_bound=50)
        # missing neighbours have infinite distance
        idx = idx[~np.isinf(dist)]
        # count the most common ids. Do not use np.median as it averages ties
        counts = Counter(ids[idx])
        max_count = max(counts.values())
        for id_, count in counts.items():
            if count == max_count:
                nearest_neighbours_id[i] = id_
                break
    return nearest_neighbours_id


for roi in ops["use_rois"]:
    print(f"Regenerating genes_round_spots_{roi}.pkl...")
    all_genes = iss.pipeline.stitch.merge_and_align_spots(
        data_path,
        roi,
        spots_prefix="genes_round",
        reg_prefix="genes_round_4_1",
        ref_prefix="genes_round_4_1",
        keep_all_spots=True,
    )
    points = all_genes[["x", "y"]].values
    tiles = {t: i for i, t in enumerate(all_genes["tile"].unique())}
    ids = all_genes["tile"].map(tiles).values
    nearest_neighbours = find_nearest_neighbours(points, ids, 10)
    points2keep = nearest_neighbours == ids
    all_genes = all_genes[points2keep].copy()
    all_genes.to_pickle(processed_path / f"genes_round_spots_{roi}.pkl")  

# %%
# plot one example for the last ROI to show what I did
print(points.shape)
skip = 10
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].scatter(points[::skip, 0], points[::skip, 1], c=ids[::skip], cmap="prism", s=0.5, alpha=0.5)
axes[0].set_title("ROI of original points")
axes[1].scatter(points[::skip, 0], points[::skip, 1], c=nearest_neighbours[::skip], cmap="prism", s=0.5, alpha=0.5)
axes[1].set_title("Filter by neighbours")
axes[2].scatter(points[::skip, 0], points[::skip, 1], c=points2keep[::skip], cmap="Pastel1", s=1, alpha=0.5)
axes[2].set_title("Points to keep")

for x in axes:
    x.set_aspect("equal")
    x.set_xticks([])
    x.set_yticks([])
    x.set_facecolor("black")
    x.invert_yaxis()
# %%
fig.savefig(
    processed_path / "figures" / "remove_duplicates_using_nearest_neighbours.png",
    dpi=1200,
)
# %%
