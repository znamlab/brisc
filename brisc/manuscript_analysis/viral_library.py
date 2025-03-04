import numpy as np
import matplotlib.ticker as mticker
from pathlib import Path


def plasmid_sequencing_data(seq_path):
    """Create array holding the frequency of each barcode in the sequenced library.
    Input data should be stored in tabbed rows, with the first column being
    the barcode index, and the second column being the barcode abundance.

    Args:
        seq_path (str): Path to the sequencing data file.

    Returns:
        np.ndarray: Array containing barcode index and abundance.
    """
    with open(seq_path, "r", encoding="utf-8-sig") as encoded_path:
        sequencing_counts = np.genfromtxt(
            encoded_path, delimiter="\t", dtype=int, usecols=(0)
        )

    array = np.zeros((len(sequencing_counts), 2))
    array[:, 0] = np.arange(1, len(sequencing_counts) + 1)
    array[:, 1] = sequencing_counts
    return array


def virus_sequencing_data(seq_path):
    """Create array holding the frequency of each barcode in the sequenced library.
    Input data should be stored in tabbed rows, with the first column being
    the barcode abundance, and the second column being the barcode sequence.

    Args:
        seq_path (str): Path to the sequencing data file.

    Returns:
        np.ndarray: Array containing barcode index and abundance.
    """
    with open(seq_path, "r", encoding="utf-8-sig") as encoded_path:
        sequencing_counts = np.genfromtxt(
            encoded_path, delimiter="\t", dtype=int, usecols=(0)
        )

    array = np.zeros((len(sequencing_counts), 2))
    array[:, 0] = np.arange(1, len(sequencing_counts) + 1)
    array[:, 1] = sequencing_counts
    return array


def probability_distribution(barcode_distribution):
    """Calculate the probability of picking each barcode.

    Args:
        barcode_distribution (np.ndarray): Array containing barcode index and abundance.

    Returns:
        np.ndarray: Array containing the probability of picking each barcode.
    """
    total_barcodes = sum(barcode_distribution[:, 1])
    barcode_probability = barcode_distribution[:, 1] / total_barcodes
    return barcode_probability


def fraction_unique(barcode_probability, cell_num):
    """Calculate the fraction of uniquely labelled cells.

    Args:
        barcode_probability (np.ndarray): Array containing the probability of
        picking each barcode.
        cell_num (int): Number of infected cells.

    Returns:
        float: Fraction of uniquely labelled cells.
    """
    barcode_contribution = barcode_probability * (
        1 - (1 - barcode_probability) ** (cell_num - 1)
    )
    unique = 1 - sum(barcode_contribution)
    return unique


def cum_probability_array(probability_distribution, i):
    """Creates array of cumulative probability.

    Args:
        probability_distribution (np.ndarray): Array containing the probability
        of picking each barcode.
        i (int): Number of infected cells.

    Returns:
        np.ndarray: Array containing barcode index and cumulative probability.
    """
    cum_array = np.zeros((len(probability_distribution), 2))
    sequences = np.flip(virus_sequencing_data(i)[:, 1])
    prob = np.flip(probability_distribution)
    cum_array[:, 0] = sequences
    cum_array[:, 1] = np.cumsum(prob)

    return cum_array


def despine(ax):
    """
    Remove right and top spines from a matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): Axes to despine.
    """
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_barcode_counts_and_percentage(
    data_path,
    virus_to_plot,
    virus_ed,
    virus_collapse,
    plasmid_to_plot,
    plasmid_ed,
    plasmid_collapse,
    ax=None,
    verbose=False,
    label_fontsize=20,
    label_pad=20,
    tick_fontsize=20,
    tick_pad=10,
    line_alpha=0.6,
    line_width=2,
    colors=("dodgerblue", "turquoise", "gold", "darkorange", "green"),
):
    """
    Plot barcode abundance (log scale, left y-axis) and
    percentage of total UMI counts (linear scale, right y-axis)
    vs. barcode index (log scale on x-axis).

    Args:
        data_path (str): Path to the data directory.
        virus_to_plot (list): List of virus names to plot.
        virus_ed (int): Edit distance for virus data.
        virus_collapse (str): Collapse method for virus data.
        plasmid_to_plot (list): List of plasmid names to plot.
        plasmid_ed (int): Edit distance for plasmid data.
        plasmid_collapse (str): Collapse method for plasmid data.
        ax (matplotlib.axes.Axes): Axes to plot on.
        verbose (bool): Print verbose output.
        label_fontsize (int): Font size for axis labels.
        label_pad (int): Padding for axis labels.
        tick_fontsize (int): Font size for axis ticks.
        tick_pad (int): Padding for axis ticks.
        line_alpha (float): Alpha value for plot lines.
        line_width (int): Width of plot lines.
        colors (list): List of colors for plotting.

    Returns:
        matplotlib.axes.Axes: Axes containing the plot.
    """

    data_path = Path(data_path)
    # Prepare file paths
    virus_seq_data = [
        data_path / virus / f"{virus}_{virus_collapse}_ed{virus_ed}.txt"
        for virus in virus_to_plot
    ]
    plasmid_seq_data = [
        data_path / plasmid / f"{plasmid}_{plasmid_collapse}_ed{plasmid_ed}.txt"
        for plasmid in plasmid_to_plot
    ]

    # Color mapping
    all_samples = virus_to_plot + plasmid_to_plot
    color_dict = {sample: color for sample, color in zip(all_samples, colors)}

    ax_left = ax

    # Plot: Barcode abundance on ax_left (log y-axis)
    for i, plasmid_file in enumerate(plasmid_seq_data):
        if verbose:
            print(f"Preparing log barcode counts for plasmid: {plasmid_file}")
        data = plasmid_sequencing_data(plasmid_file)
        label = plasmid_to_plot[i]
        color = color_dict[label]
        ax_left.plot(
            data[:, 0],
            data[:, 1],
            drawstyle="steps-pre",
            alpha=line_alpha,
            linewidth=line_width,
            color=color,
            label=label,
        )

    for i, virus_file in enumerate(virus_seq_data):
        if verbose:
            print(f"Preparing log barcode counts for virus: {virus_file}")
        data = virus_sequencing_data(virus_file)
        label = virus_to_plot[i]
        color = color_dict[label]
        ax_left.plot(
            data[:, 0],
            data[:, 1],
            drawstyle="steps-pre",
            alpha=line_alpha,
            linewidth=line_width,
            color=color,
            label=label,
        )

    # Format ax_left
    ax_left.set_xscale("log")
    ax_left.set_xlim(1, 1e8)
    ax_left.xaxis.set_major_locator(mticker.FixedLocator(locs=np.logspace(0, 8, 9)))
    ax_left.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax_left.set_xlabel("Barcode index", fontsize=label_fontsize, labelpad=label_pad)

    # Optionally hide every other tick label
    for lbl in ax_left.xaxis.get_ticklabels()[1::2]:
        lbl.set_visible(False)

    ax_left.set_yscale("log")
    ax_left.set_ylim(1, 1e6)
    ax_left.yaxis.set_major_locator(mticker.FixedLocator(locs=np.logspace(0, 6, 7)))
    ax_left.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    for lbl in ax_left.yaxis.get_ticklabels()[1::2]:
        lbl.set_visible(False)

    ax_left.set_ylabel("Barcode abundance", fontsize=label_fontsize, labelpad=label_pad)
    ax_left.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        pad=tick_pad,
    )

    # Create secondary y-axis on the same x-axis
    ax_right = ax_left.twinx()
    ax_right.set_xscale("log")
    ax_right.set_xlim(1, 1e8)
    ax_right.xaxis.set_major_locator(mticker.FixedLocator(locs=np.logspace(0, 8, 9)))
    ax_right.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    # Make the right y-axis also log
    ax_right.set_yscale("log")
    ax_right.set_ylabel(
        "Percentage of total UMI counts (%)",
        fontsize=label_fontsize,
        labelpad=label_pad,
    )
    ax_right.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        pad=tick_pad,
    )
    total_umis = 0
    for plasmid_file in plasmid_seq_data:
        d = plasmid_sequencing_data(plasmid_file)
        total_umis += d[:, 1].sum()
    for virus_file in virus_seq_data:
        d = virus_sequencing_data(virus_file)
        total_umis += d[:, 1].sum()

    ymin, ymax = ax_left.get_ylim()
    ax_right.set_ylim(ymin / total_umis * 100, ymax / total_umis * 100)

    ax_right.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
    ax_right.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax_right.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Legend (based on ax_left handles/labels)
    handles, labels = ax_left.get_legend_handles_labels()
    ax_left.legend(handles, labels, fontsize=tick_fontsize, loc="best")

    return ax_left, ax_right


def plot_unique_label_fraction(
    data_path,
    virus_to_plot,
    virus_ed,
    virus_collapse,
    plasmid_to_plot,
    plasmid_ed,
    plasmid_collapse,
    ax=None,
    stride=50,
    max_cells=2000,
    min_max_percent_unique_range=(0.5, 1.0),
    verbose=False,
    label_fontsize=20,
    label_pad=20,
    tick_fontsize=20,
    tick_pad=10,
    line_alpha=0.6,
    line_width=2,
    colors=("dodgerblue", "turquoise", "gold", "darkorange", "green"),
):
    """
    Plot fraction of uniquely labeled cells vs. number of infections,
    for both plasmids and viruses. If `ax` is None, a new figure is created.

    Args:
        data_path (str): Path to the data directory.
        virus_to_plot (list): List of virus names to plot.
        virus_ed (int): Edit distance for virus data.
        virus_collapse (str): Collapse method for virus data.
        plasmid_to_plot (list): List of plasmid names to plot.
        plasmid_ed (int): Edit distance for plasmid data.
        plasmid_collapse (str): Collapse method for plasmid data.
        ax (matplotlib.axes.Axes): Axes to plot on.
        stride (int): Number of infection conditions to evaluate.
        max_cells (int): Maximum number of infected cells to plot.
        min_max_percent_unique_range (tuple): Y-axis range for unique fraction.
        verbose (bool): Print verbose output.
        label_fontsize (int): Font size for axis labels.
        label_pad (int): Padding for axis labels.
        tick_fontsize (int): Font size for axis ticks.
        tick_pad (int): Padding for axis ticks.
        line_alpha (float): Alpha value for plot lines.
        line_width (int): Width of plot lines.
        colors (list): List of colors for plotting.

    Returns:
        matplotlib.axes.Axes: Axes containing the plot.
    """

    data_path = Path(data_path)
    virus_seq_data = [
        data_path / virus / f"{virus}_{virus_collapse}_ed{virus_ed}.txt"
        for virus in virus_to_plot
    ]
    plasmid_seq_data = [
        data_path / plasmid / f"{plasmid}_{plasmid_collapse}_ed{plasmid_ed}.txt"
        for plasmid in plasmid_to_plot
    ]

    # Color mapping
    all_samples = virus_to_plot + plasmid_to_plot
    color_dict = {sample: color for sample, color in zip(all_samples, colors)}

    evaluation_points = np.linspace(1, max_cells, stride, dtype=int)
    ax_ = ax
    # Plot plasmid data
    for i, plasmid_file in enumerate(plasmid_seq_data):
        if verbose:
            print(f"Preparing uniquely labeled plot for plasmid: {plasmid_file}")
        barcode_distribution = plasmid_sequencing_data(plasmid_file)
        barcode_probability = probability_distribution(barcode_distribution)

        fractions = []
        for num in evaluation_points:
            frac = fraction_unique(barcode_probability, num)
            fractions.append(frac)

        ax_.plot(
            evaluation_points,
            fractions,
            alpha=line_alpha,
            linewidth=line_width,
            color=color_dict[plasmid_to_plot[i]],
            label=plasmid_to_plot[i],
        )

    # Plot virus data
    for i, virus_file in enumerate(virus_seq_data):
        if verbose:
            print(f"Preparing uniquely labeled plot for virus: {virus_file}")
        barcode_distribution = virus_sequencing_data(virus_file)
        barcode_probability = probability_distribution(barcode_distribution)

        fractions = []
        for num in evaluation_points:
            frac = fraction_unique(barcode_probability, num)
            fractions.append(frac)

        ax_.plot(
            evaluation_points,
            fractions,
            alpha=line_alpha,
            linewidth=line_width,
            color=color_dict[virus_to_plot[i]],
            label=virus_to_plot[i],
        )

    # Formatting
    ax_.set_xlabel(
        "No. of independent infections", fontsize=label_fontsize, labelpad=label_pad
    )
    ax_.set_ylabel(
        "Uniquely labeled cells (%)", fontsize=label_fontsize, labelpad=label_pad
    )
    ax_.set_xlim(0, max_cells)
    xticks = np.linspace(0, max_cells, 5, dtype=int)
    ax_.set_xticks(xticks)
    yticks = np.linspace(
        min_max_percent_unique_range[0], min_max_percent_unique_range[1], 5
    )
    ax_.set_yticks(yticks)
    # Convert fractions to integer percents
    ax_.set_yticklabels((yticks * 100).astype(str))
    ax_.set_ylim(min_max_percent_unique_range[0], min_max_percent_unique_range[1])
    ax_.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        pad=tick_pad,
    )

    handles, labels = ax_.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax_.legend(
        unique_legend.values(), unique_legend.keys(), loc="best", fontsize=tick_fontsize
    )

    return ax_
