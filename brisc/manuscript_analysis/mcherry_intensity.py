import iss_analysis as issa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from scipy.stats import linregress
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
    xcol="intensity_mean-0",
    set_ticks=True,
):
    """Plot the number of presynaptic cells of each starter against its mCherry
    fluorescence, with a robust linear fit of the log-log relationship.

    Returns:
        dict: `plotted_element` with a `starter_cells` entry holding the plotted
            coordinates of every starter (the natural logarithm of its mCherry
            fluorescence `x` and of its presynaptic-cell count `y`) and a `robust_fit`
            entry holding the fitted line as drawn by seaborn, with the bounds of its
            bootstrap confidence band (`ci_lower`, `ci_upper`).
    """
    if ax is None:
        ax = plt.gca()
    # ax.set(xscale="log", yscale="log")
    n_lines_before = len(ax.lines)
    n_collections_before = len(ax.collections)
    sns.regplot(
        x=np.log(valid[xcol]),
        y=np.log(valid["n_presynaptic"]),
        scatter_kws={
            "s": marker_size,
            "color": "darkslategray",
            "edgecolor": "black",
            "alpha": 0.5,
        },
        line_kws={"color": "darkslategray"},
        robust=True,
        ax=ax,
    )
    plotted_element = _mcherry_plotted_element(
        ax, valid, xcol, n_lines_before, n_collections_before
    )
    slope, intercept, rvalue, pvalue, stderr = linregress(
        x=np.log(valid[xcol].values), y=np.log(valid["n_presynaptic"].values)
    )
    txt = f"n = {len(valid)}, # presynaptic = {slope:.2f} mCherry + {intercept:.2f}."
    txt += f" rvalue={rvalue:.2f}, pvalue={pvalue:.2e}"
    print(txt)
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
        "Number of presynaptic cells + 1",
        fontsize=label_fontsize,
    )
    if set_ticks:
        ax.set_xticks(np.log([100, 1000]), labels=[100, 1000])
    ax.set_yticks(np.log([1, 10, 100]), labels=[1, 10, 100])

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
    sns.despine(ax=ax)
    return plotted_element


def _mcherry_plotted_element(ax, valid, xcol, n_lines_before, n_collections_before):
    """Collect what `sns.regplot` just drew on `ax`, for the Source Data workbook.

    Seaborn fits and resamples internally, so the fitted line and its confidence band
    are read back from the artists it added rather than recomputed.

    Args:
        ax (matplotlib.axes.Axes): Axes `sns.regplot` drew on.
        valid (pd.DataFrame): Starter cells that were plotted.
        xcol (str): Column holding the mCherry fluorescence.
        n_lines_before (int): Number of lines on `ax` before the regplot call.
        n_collections_before (int): Number of collections on `ax` before the call.

    Returns:
        dict: `plotted_element`, see `plot_mcherry_intensity_presyn`.
    """
    labels = dict(
        xlabel="Starter mCherry fluorescence (AU)",
        ylabel="Number of presynaptic cells + 1",
    )
    plotted_element = dict(
        starter_cells=dict(
            x=np.log(valid[xcol].values.astype(float)),
            y=np.log(valid["n_presynaptic"].values.astype(float)),
            color="darkslategray",
            **labels,
        )
    )
    new_lines = ax.lines[n_lines_before:]
    if not new_lines:
        return plotted_element
    line = new_lines[-1]
    grid = np.asarray(line.get_xdata(), dtype=float)
    fit = dict(
        x=grid,
        y=np.asarray(line.get_ydata(), dtype=float),
        color="darkslategray",
    )
    bands = [
        collection
        for collection in ax.collections[n_collections_before:]
        if isinstance(collection, PolyCollection)
    ]
    if bands:
        # `fill_between` closes the band into one polygon, so the two bounds are the
        # lowest and highest vertex at each point of the fitted grid.
        vertices = bands[-1].get_paths()[0].vertices
        lower, upper = [], []
        for x in grid:
            at_x = vertices[vertices[:, 0] == x, 1]
            lower.append(at_x.min() if at_x.size else np.nan)
            upper.append(at_x.max() if at_x.size else np.nan)
        fit["ci_lower"] = np.array(lower)
        fit["ci_upper"] = np.array(upper)
    fit.update(labels)
    plotted_element["robust_fit"] = fit
    return plotted_element
