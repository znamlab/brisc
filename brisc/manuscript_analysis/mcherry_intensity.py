import iss_analysis as issa

import numpy as np
import pandas as pd
import flexiznam as flz
import seaborn as sns


def load_mcherry_data(
    project="becalia_rabies_barseq",
    mouse="BRAC8498.3e",
    error_correction_ds_name="BRAC8498.3e_error_corrected_barcodes_26",
):
    df_file = flz.get_processed_path(
        f"becalia_rabies_barseq/BRAC8498.3e/analysis/{error_correction_ds_name}_cell_barcode_df.pkl"
    )

    full_df = pd.read_pickle(df_file)
    mcherry_cells = issa.io.get_mcherry_cells(
        project, mouse, verbose=True, which="curated", prefix="mCherry_1"
    )
    print(f"Loaded {len(full_df)} cells")
    barcoded_cells = full_df.query("all_barcodes.notna()")
    print(f"Found {len(barcoded_cells)} barcoded cells")
    starter_cells_df = barcoded_cells.query("is_starter == True")
    starter_barcode = {}
    for i, row in starter_cells_df.iterrows():
        for bc in row["all_barcodes"]:
            starter_barcode.setdefault(bc, []).append(i)
    print(
        f"Found {len(starter_barcode)} unique starter barcodes in {len(starter_cells_df)} starter cells"
    )
    # Add mcherry intensity to the starter_df
    mcherry_cells.set_index("mcherry_uid", inplace=True, drop=False)
    starter_cells_df = starter_cells_df.copy()
    starter_cells_df["mcherry_intensity"] = starter_cells_df["mcherry_uid"].map(
        lambda x: mcherry_cells.loc[x, "intensity_mean-1"]
    )
    # BC present in only 1 starter:
    single_starter_barcodes = {k: v for k, v in starter_barcode.items() if len(v) == 1}
    print(
        f"Found {len(single_starter_barcodes)} barcodes present in only 1 starter cell"
    )
    mcherry_cells.set_index("mcherry_uid", inplace=True, drop=False)
    mcherry_cells["is_starter"] = False
    mcherry_cells["n_presynaptic"] = np.nan

    exploded = barcoded_cells.all_barcodes.explode()
    for bc, starter in single_starter_barcodes.items():
        s_df = starter_cells_df.loc[starter[0]]
        mch = s_df.mcherry_uid
        cell_with_bc = exploded[exploded == bc].copy()
        mcherry_cells.loc[mch, "is_starter"] = True
        mcherry_cells.loc[mch, "n_presynaptic"] = len(cell_with_bc)
    valid = mcherry_cells.query("is_starter == True")

    return valid


def plot_mcherry_intensity_presyn(
    valid,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    marker_size=10,
):
    # ax.set(xscale="log", yscale="log")
    sns.regplot(
        x=np.log(valid["intensity_mean-0"]),
        y=np.log(valid["n_presynaptic"]),
        scatter_kws={
            "s": marker_size,
            "color": "darkslategray",
            "edgecolor": "black",
            "alpha": 0.5,
        },
        line_kws={"color": "darkslategray"},
        robust=True,
    )

    # ax.scatter(

    #     valid["n_presynaptic"],
    #     alpha=0.5,
    #     s=spot_size,
    # )
    ax.set_xlabel(
        "Starter mCherry\nfluorescence (AU)",
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        "Number of labelled cellss",
        fontsize=label_fontsize,
    )
    ax.set_yticks(np.log([1, 10, 100]), labels=[1, 10, 100])
    ax.set_xticks(np.log([100, 1000]), labels=[100, 1000])

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
    sns.despine(ax=ax)
