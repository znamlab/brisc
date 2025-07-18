import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from .utils import despine


def plot_hist(
    data_df,
    ax,
    col="n_unique_barcodes",
    tick_fontsize=12,
    y_offset=0.05,
    max_val=None,
    show_zero=False,
    show_counts=True,
    linewidth=0.5,
):
    """Plots a histogram of the number of unique barcodes.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        col (str, optional): Column to plot. Defaults to "n_unique_barcodes".
        tick_fontsize (int, optional): Fontsize for the tick labels. Defaults to 12.
        y_offset (float, optional): Offset for the text annotations. Defaults to 0.05.
        max_val (int, optional): Maximum value for the x-axis. Defaults to None.
        show_zero (bool, optional): Whether to show zero values. Defaults to False.
        show_counts (bool, optional): Whether to show counts on the bars. Defaults to True.
        linewidth (float, optional): Linewidth for the bars. Defaults to 0.5.
    """
    if max_val is None:
        max_val = data_df[col].max() + 1
    if show_zero:
        min_val = 0
    else:
        min_val = 1
    counts = np.bincount(data_df[col].values, minlength=max_val + 1)
    props = counts / np.sum(counts)
    counts = counts[min_val : max_val + 1]
    props = props[min_val : max_val + 1]
    plt.stairs(
        props,
        np.arange(min_val, max_val + 2) - 0.5,
        fill=True,
        edgecolor="black",
        facecolor="slategray",
        linewidth=linewidth,
    )
    if show_counts:
        for i, (count, prop) in enumerate(zip(counts, props)):
            # Annotate unfiltered data with padding if heights are similar
            plt.text(
                i + min_val - 0.2,
                y_offset + prop,
                str(count),
                ha="left",
                fontsize=tick_fontsize,
                color="black",
                alpha=0.8,
                rotation=35,
            )

    ax.set_ylim(0, 1)
    ax.set_xlim(min_val - 0.5, max_val + 0.5)

    ax.set_xticks(np.arange(min_val, max_val + 1, 1))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
    despine(ax)


def plot_presyn_per_barcode(
    barcodes_df,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    max_val=30,
    colors=("darkorange", "dodgerblue"),
    linewidth=0.5,
    alpha=0.8,
    log_scale=True,
):
    """Plots histograms of the number of presynaptic cells per barcode.

    The function distinguishes between barcodes associated with starter cells
    ('non-orphan') and those not associated with any starter cell ('orphan').
    It can display the distribution on a linear or logarithmic x-axis.

    Args:
        barcodes_df (pd.DataFrame): DataFrame containing barcode data.
            Must include 'n_starters' and 'n_presynaptic' columns.
        ax (plt.Axes, optional): Matplotlib axes object to plot on.
            If None, the current axes are used. Defaults to None.
        label_fontsize (int, optional): Font size for axis labels. Defaults to 12.
        tick_fontsize (int, optional): Font size for tick labels. Defaults to 12.
        max_val (int, optional): Maximum value for the x-axis when
            `log_scale` is False. Defaults to 30.
        colors (tuple, optional): Colors for the two histograms
            (orphan, non-orphan). Defaults to ("darkorange", "dodgerblue").
        linewidth (float, optional): Linewidth for the histogram steps.
            Defaults to 0.5.
        alpha (float, optional): Alpha transparency for the histogram fill. Defaults to
            0.8.
        log_scale (bool, optional): If True, plots with a logarithmic x-axis
            and filled steps. If False, plots with a linear x-axis and step lines.
            Defaults to True.
    """
    cells_with_starter = barcodes_df[barcodes_df["n_starters"] > 0][
        "n_presynaptic"
    ].values
    cells_without_starter = barcodes_df[barcodes_df["n_starters"] == 0][
        "n_presynaptic"
    ].values
    # Regular histogram (left subplot)

    # Non-starters *with* starter
    labels = ["Orphan barcodes", "Non-orphan barcodes"]
    if log_scale:
        ax = plt.gca()
        max_n = max(cells_without_starter.max(), cells_with_starter.max())
        print(max_n)
        bins = 10 ** (np.arange(0, np.log10(max_n), 0.16))
        i = 0
        for cells, color in zip((cells_without_starter, cells_with_starter), colors):
            # c = np.log10(c)
            h, b = np.histogram(cells, bins=bins)
            ax.stairs(
                h / len(cells),
                b,
                fill=True,
                color=color,
                lw=0,
                alpha=alpha,
                label=labels[i],
            )
            ax.stairs(
                h / len(cells),
                b,
                fill=False,
                color=color,
                lw=linewidth,
                alpha=1,
                label="__no_legend__",
            )

            zero_cells = (cells == 0).sum()
            ax.stairs(
                [zero_cells / len(cells)],
                [0.48, 0.69],
                fill=True,
                color=color,
                lw=linewidth,
                alpha=alpha,
                label="__no_legend__",
            )
            ax.stairs(
                [zero_cells / len(cells)],
                [0.48, 0.69],
                fill=False,
                color=color,
                lw=linewidth,
                alpha=1,
                label="__no_legend__",
            )

            i += 1

        ax.set_xscale("log")
        ax.set_xlim(0.48, b[-1])
        ax.set_xticks(
            [np.sqrt(0.48 * 0.69), 1, 10, 100],
            labels=["0", "1", "10", "100"],
            fontsize=tick_fontsize,
        )
        # Manually set minor ticks to appear only for values >= 1
        # Generate minor ticks for decades starting from 10**0 (which is 1)
        minor_ticks = [
            j * (10**i)
            for i in range(0, 100)
            for j in range(2, 10)
            if j * (10**i) < 100
        ]
        ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    else:
        bin_edges = np.arange(0, max_val + 1, 1)
        bin_edges[-1] = 1e4
        i = 0
        for cells, color in zip((cells_without_starter, cells_with_starter), colors):
            counts, _ = np.histogram(cells, bins=bin_edges)
            props = counts / np.sum(counts)
            # add 0 to finish the last bar
            edge2plot = np.hstack([bin_edges[:-1], max_val])
            props2plot = np.hstack([props, 0])
            plt.plot(
                edge2plot,
                props2plot,
                linewidth=linewidth,
                color=color,
                drawstyle="steps-post",
                alpha=alpha,
                label=labels[i],
            )
            i += 1
        ax.set_xlim(0, max_val + 1)
        labels = [str(l) for l in np.arange(0, max_val + 10, 10).astype(int)]
        labels[-1] = f">{labels[-1]}"
        ax.set_xticks(np.arange(0, max_val + 10, 10), labels=labels)

    ax.set_xlabel(
        "Presynaptic cells per barcode",
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        "Proportion of barcodes",
        fontsize=label_fontsize,
    )
    ax.legend(
        loc="upper right",
        fontsize=tick_fontsize,
        frameon=False,
        bbox_to_anchor=[1.3, 1],
        handlelength=1,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
    despine(ax)


def analyze_multibarcoded_starters(
    cells_df,
):
    """
    Load data for double barcodes analysis

    Args:
        data_path (str): Path to the data folder

    Returns:
        result_df (pd.DataFrame): DataFrame with columns:
        - starter_cell_id (str): ID of the starter cell
        - barcode1 (str): First barcode
        - barcode2 (str): Second barcode
        - n_presyn_with_barcode1 (int): Number of presynaptic cells with barcode1
        - n_presyn_with_barcode2 (int): Number of presynaptic cells with barcode2
        - n_presyn_with_both (int): Number of presynaptic cells with both barcodes
    """
    # For each cell, define 'unique_barcodes' = intersection of its barcodes with singletons
    # Among starter cells, pick out those whose unique_barcodes has length == 2
    double_barcoded_starters = cells_df[
        (cells_df["is_starter"] == True) & (cells_df["unique_barcodes"].apply(len) > 1)
    ].copy()
    presyn_cells = cells_df[cells_df["is_starter"] == False]
    # For each double-barcoded starter, count how many presyn cells have bc1, bc2, or both
    results = []
    for i, row in double_barcoded_starters.iterrows():
        starter_id = row["cell_id"]
        barcodes = row["unique_barcodes"]
        n_presyn_per_barcode = [
            presyn_cells["unique_barcodes"]
            .apply(lambda x: len(x.intersection(barcodes)) == 1 and barcode in x)
            .sum()
            for barcode in barcodes
        ]
        n_presyn = (
            presyn_cells["unique_barcodes"]
            .apply(lambda x: len(x.intersection(barcodes)) > 0)
            .sum()
        )
        barcode_counts = (
            presyn_cells["unique_barcodes"]
            .apply(lambda x: len(x.intersection(barcodes)))
            .value_counts()
            .values
        )
        results.append(
            {
                "starter_cell_id": starter_id,
                "barcodes": barcodes,
                "n_presyn_per_barcode": n_presyn_per_barcode,
                "barcode_counts": barcode_counts,
                "n_presyn": n_presyn,
            }
        )
    result_df = pd.DataFrame(results)

    return result_df


def plot_multibarcoded_starters(
    multibarcoded_starters,
    ax=None,
    label_fontsize=12,
    tick_fontsize=12,
    barcode_proportion=True,
    legend_fontsize=8,
):
    """
    Plot a barstack and violin plot of the number of presynaptic cells with each barcode

    Args:
        ax_stack (matplotlib.axes.Axes): Axes for the barstack plot
        ax_violin (matplotlib.axes.Axes): Axes for the violin plot
        result_df (pd.DataFrame): DataFrame with columns:
            - starter_cell_id (str): ID of the starter cell
            - barcode1 (str): First barcode
            - barcode2 (str): Second barcode
            - n_presyn_with_barcode1 (int): Number of presynaptic cells with barcode1
            - n_presyn_with_barcode2 (int): Number of presynaptic cells with barcode2
            - n_presyn_with_both (int): Number of presynaptic cells with both barcodes
    """
    max_bc = (
        multibarcoded_starters["n_presyn_per_barcode"].apply(len).max()
    )  # .astype(int)
    multibarcoded_starters["n_barcodes"] = multibarcoded_starters["barcodes"].apply(len)
    multibarcoded_starters["n_presyn_per_barcode"] = multibarcoded_starters[
        "n_presyn_per_barcode"
    ].apply(lambda x: sorted(x, reverse=True))
    multibarcoded_starters["top_barcode_prop"] = multibarcoded_starters.apply(
        lambda x: x["n_presyn_per_barcode"][0] / x["n_presyn"],
        axis=1,
    )
    if barcode_proportion:
        multibarcoded_starters = multibarcoded_starters.sort_values(
            [
                "top_barcode_prop",
            ],
            ascending=False,
        )
    else:
        multibarcoded_starters = multibarcoded_starters.sort_values(
            [
                "n_presyn",
            ],
            ascending=False,
        )
    # pad barcode_counts to max_bc+1
    barcode_counts = np.stack(
        multibarcoded_starters["barcode_counts"]
        .apply(
            lambda x: np.append(
                x,
                [
                    0,
                ]
                * (max_bc - len(x) + 1),
            )
        )
        .values
    )
    n_presyn_per_barcode = np.stack(
        multibarcoded_starters["n_presyn_per_barcode"]
        .apply(
            lambda x: np.append(
                x,
                [
                    0,
                ]
                * (max_bc - len(x)),
            )
        )
        .values
    )

    bar_data = np.hstack([n_presyn_per_barcode, barcode_counts[:, 2:]])
    if barcode_proportion:
        bar_data /= np.sum(bar_data, axis=1)[:, None]
    bottom = 0
    colors = [
        "lightskyblue",
        "dodgerblue",
        "royalblue",
        "darkblue",
        "red",
        "orange",
        "orangered",
    ]
    labels = [
        "Barcode 1",
        "Barcode 2",
        "Barcode 3",
        "Barcode 4",
        "Any 2",
        "Any 3",
        "Any 4",
    ]
    for i, color, label in zip(range(bar_data.shape[1]), colors, labels):
        ax.bar(
            np.arange(bar_data.shape[0]) + 1,
            bar_data[:, i],
            bottom=bottom,
            color=color,
            label=label,
            width=1,
        )
        bottom += bar_data[:, i]
    if barcode_proportion:
        ax.set_ylabel("Proportion of presynaptic cells", fontsize=label_fontsize)
    else:
        ax.set_ylabel("# presynaptic cells", fontsize=label_fontsize)
    ax.set_xlabel(
        "Starter cell",
        fontsize=label_fontsize,
    )
    ax.legend(
        fontsize=legend_fontsize,
        loc="upper left",
        handlelength=1,
        frameon=False,
        bbox_to_anchor=[0.6, 1.1],
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
    ax.set_xticks([1, bar_data.shape[0]])
    despine(ax)
