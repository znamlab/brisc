import numpy as np
import matplotlib.ticker as mticker
from scipy.optimize import minimize_scalar
from .utils import despine


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


def load_library_data(data_path, library, edit_distance, collapse):
    # Prepare file paths
    fname = data_path / library / f"{library}_{collapse}_ed{edit_distance}.txt"
    with open(fname, "r", encoding="utf-8-sig") as encoded_path:
        data = np.genfromtxt(encoded_path, delimiter="\t")
        # Check if data has one or two columns
        if data.ndim == 1:  # Single column
            sequencing_counts = data.astype(int)
        else:  # Two columns
            sequencing_counts = data[:, 0].astype(int)

    array = np.zeros((len(sequencing_counts), 2))
    array[:, 0] = np.arange(1, len(sequencing_counts) + 1)
    array[:, 1] = sequencing_counts
    return array


def plot_barcode_counts_and_percentage(
    libraries,
    ax,
    label_fontsize=8,
    tick_fontsize=6,
    line_alpha=0.6,
    line_width=2,
    colors=("dodgerblue", "turquoise", "gold", "darkorange", "green"),
    show_legend=True,
):
    """
    Plot barcode abundance (log scale, left y-axis) and
    percentage of total UMI counts (linear scale, right y-axis)
    vs. barcode index (log scale on x-axis).

    Args:
        libraries (list): List of libraries to plot.
        ax (matplotlib.axes.Axes): Axes to plot on.
        verbose (bool): Print verbose output.
        label_fontsize (int): Font size for axis labels.
        label_pad (int): Padding for axis labels.
        tick_fontsize (int): Font size for axis ticks.
        tick_pad (int): Padding for axis ticks.
        line_alpha (float): Alpha value for plot lines.
        line_width (int): Width of plot lines.
        colors (list): List of colors for plotting.
        show_legend (bool): display legend

    Returns:
        matplotlib.axes.Axes: Axes containing the plot.
    """
    for library_label, color in zip(libraries.keys(), colors):
        ax.plot(
            libraries[library_label][:, 0],
            libraries[library_label][:, 1],
            drawstyle="steps-pre",
            alpha=line_alpha,
            linewidth=line_width,
            color=color,
            label=library_label,
        )

    # Format ax_left
    ax.set_xscale("log")
    ax.set_xlim(1, 1e8)
    ax.xaxis.set_major_locator(mticker.FixedLocator(locs=np.logspace(0, 8, 9)))
    ax.set_xlabel("Barcode index", fontsize=label_fontsize)

    # Optionally hide every other tick label
    for lbl in ax.xaxis.get_ticklabels()[1::2]:
        lbl.set_visible(False)

    ax.set_yscale("log")
    ax.set_ylim(0.8, 1e6)
    ax.yaxis.set_major_locator(mticker.FixedLocator(locs=np.logspace(0, 6, 7)))
    for lbl in ax.yaxis.get_ticklabels()[1::2]:
        lbl.set_visible(False)

    ax.set_ylabel("Barcode abundance", fontsize=label_fontsize)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    if show_legend:
        # Legend (based on ax_left handles/labels)
        ax.legend(
            fontsize=tick_fontsize,
            loc="upper right",
            frameon=False,
            handlelength=1,
            bbox_to_anchor=(1.2, 1.05),
            borderpad=0.0,
        )
    despine(ax)


def find_max_cell_below_percprop(
    barcode_probability, fractions, evaluation_points, thres
):
    """Find the maximum number of cells that can be infected to achieve a target
    fraction of uniquely labeled cells.

    This function uses numerical optimization to find the number of infected
    cells (`cell_num`) that minimizes the difference between the calculated
    `fraction_unique(barcode_probability, cell_num)` and the desired
    threshold `thres`.

    Args:
        barcode_probability (np.ndarray): An array containing the probability of
            picking each barcode from the library.
        fractions (list or np.ndarray): Pre-computed fractions of uniquely
            labeled cells corresponding to the `evaluation_points`.
        evaluation_points (list or np.ndarray): The number of infected cells
            for which `fractions` were calculated.
        thres (float): The target fraction of uniquely labeled cells (e.g., 0.95
            for 95%).

    Returns:
        int: The estimated maximum number of infected cells that results in a
             unique labeling fraction approximately equal to `thres`.
    """
    closest_id = len(fractions) - np.searchsorted(fractions[::-1], thres) - 1

    def cost(x):
        return ((fraction_unique(barcode_probability, x) - thres)) ** 2

    max_thres_cells = evaluation_points[closest_id]
    res = minimize_scalar(cost)
    max_thres_cells = int(np.round(res.x))
    return max_thres_cells


def plot_unique_label_fraction(
    libraries,
    log_scale=False,
    ax=None,
    stride=50,
    max_cells=2000,
    min_max_percent_unique_range=(0.5, 1.0),
    label_fontsize=20,
    tick_fontsize=20,
    line_alpha=0.6,
    line_width=2,
    colors=("dodgerblue", "turquoise", "gold", "darkorange", "green"),
    show_legend=True,
    verbose=True,
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
        show_legend (bool): Show legend.
        verbose (bool): Print verbose output.


    Returns:
        matplotlib.axes.Axes: Axes containing the plot.
    """
    if not log_scale:
        evaluation_points = np.linspace(1, max_cells, stride, dtype=int)
    else:
        evaluation_points = np.logspace(0, np.log10(max_cells), dtype=int)
    # Plot plasmid data
    for library_label, color in zip(libraries.keys(), colors):
        barcode_probability = probability_distribution(libraries[library_label])

        fractions = [
            fraction_unique(barcode_probability, num) for num in evaluation_points
        ]

        if verbose:
            # We want to print the number of cells that can be picked to have 95% unique
            # barcodes
            max_95cells = find_max_cell_below_percprop(
                barcode_probability, fractions, evaluation_points, 0.95
            )
            max_99cells = find_max_cell_below_percprop(
                barcode_probability, fractions, evaluation_points, 0.99
            )
            txt = f"For {library_label}, 95% unique at {max_95cells:.0f} cells"
            txt += f"-- 99% unique at {max_99cells:.0f} cells"
            print(txt)

        if not log_scale:
            ax.plot(
                evaluation_points,
                fractions,
                alpha=line_alpha,
                linewidth=line_width,
                color=color,
                label=library_label,
            )
            ax.set_xlim(0, max_cells)
        else:
            ax.semilogx(
                evaluation_points,
                fractions,
                alpha=line_alpha,
                linewidth=line_width,
                color=color,
                label=library_label,
            )
            ax.set_xlim(1, max_cells)

    # Formatting
    ax.set_xlabel("Number of infections", fontsize=label_fontsize)
    ax.set_ylabel(
        "Proportion of uniquely\n labeled cells",
        fontsize=label_fontsize,
    )
    pad = 0.02
    ax.set_ylim(min_max_percent_unique_range[0], min_max_percent_unique_range[1] + pad)

    # explicitly reset yticks to the "nice" ones
    orig_ticks = np.linspace(
        min_max_percent_unique_range[0], min_max_percent_unique_range[1], 3
    )
    ax.set_yticks(orig_ticks)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
    if show_legend:
        ax.legend(loc="best", fontsize=tick_fontsize, frameon=False)

    despine(ax)
