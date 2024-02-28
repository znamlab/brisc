"""Script to generate the cells df

Cannot use the current version of the pipeline if I want to error barcode
all rois together first
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
from brisc.petr_talk_202402 import helper

barcode_spots, gmm, all_barcode_spots = helper.get_barcodes(
    data_path,
    mean_intensity_threshold=0.01,
    dot_product_score_threshold=0.2,
    mean_score_threshold=0.75,
)
# Error correct
# that is a bit slow
barcode_spots = iss.call.correct_barcode_sequences(barcode_spots, 2)
barcode_spots.shape
# %%
from iss_preprocess.pipeline.segment import _get_big_masks, spot_mask_value, count_spots

mask_expansion = 2

for roi in ops["use_rois"]:
    print(f"Counting spots in cells for roi {roi}", flush=True)
    # get mask and add value to spots df
    genes_spots, roi_barcode_spots = helper.get_spots(data_path, roi, barcode_spots)
    spots_dict = {}
    print("   getting mask", flush=True)
    big_masks = _get_big_masks(
        data_path, roi, masks=None, mask_expansion=mask_expansion
    )
    print("   finding barcode id", flush=True)
    spots_dict["barcode_spots"] = spot_mask_value(big_masks, roi_barcode_spots.copy())
    print("   finding genes id", flush=True)
    spots_dict["genes_spots"] = spot_mask_value(big_masks, genes_spots.copy())

    spots_in_cells = dict()
    for prefix, spot_df in spots_dict.items():
        print(f"    counting {prefix} spots in cells", flush=True)
        grouping_column = "corrected_bases" if prefix.startswith("barcode") else "gene"
        cell_df = count_spots(spots=spot_df, grouping_column=grouping_column)
        spots_in_cells[prefix] = cell_df

    # Save barcodes
    barcode_df = spots_in_cells.pop("barcode_spots")
    save_dir = iss.io.get_processed_path(data_path) / "cells"
    save_dir.mkdir(exist_ok=True)
    barcode_df.to_pickle(save_dir / f"barcode_df_roi{roi}.pkl")
    genes_df = spots_in_cells.pop("genes_spots")
    genes_df.to_pickle(save_dir / f"genes_df_roi{roi}.pkl")


# %%
