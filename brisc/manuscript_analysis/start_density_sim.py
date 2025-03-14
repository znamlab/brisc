import numpy as np
from brisc.manuscript_analysis.utils import despine


def plot_starter_spread_sim(
    density=np.logspace(-4, 0, num=50),
    ns=[10, 20, 40],
    starter_spread_probability=0.05,
    v1_cell_density=150000,
    x_range=(1e-4, 1),
    ax=None,
    label_fontsize=12,
    tick_fontsize=10,
    line_width=0.5,
    colors=("lightsalmon", "tomato", "red")
):
    """Plot the probability of starter-to-starter spread on a single figure,
    using fraction on the bottom x-axis and absolute density (mm^{-3}) on a
    second x-axis at the top.

    Args:
        density (array-like): Fraction of all cells that are starter neurons.
        ns (array-like): Number of presynaptic cells per starter neuron.
        starter_spread_probability (float): Probability of spread between starter neurons.
        v1_cell_density (int): Number of cells per mm^3 in V1.
        x_range (tuple): Range of x-axis values.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        label_fontsize (int): Font size for labels.
        tick_fontsize (int): Font size for ticks.

    Returns:
        matplotlib.axes.Axes: Axes object with the plot.
        matplotlib.axes.Axes: Axes object with the secondary x-axis.
    """

    for n, color in zip(ns, colors):
        p = 1 - (1 - density) ** n
        ax.loglog(density, p, label=str(n), lw=line_width, c=color)

    # Horizontal dashed line at starter_spread_probability
    ax.hlines(
        starter_spread_probability,
        x_range[0],
        x_range[1],
        linestyles="dashed",
        colors="black",
        lw=line_width
    )

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_xscale("log")
    ax.set_xlabel(
        "Proportion of starter neurons",
        fontsize=label_fontsize,
    )

    ax.set_ylabel(
        "Probability of spread\n between starter neurons",
        fontsize=label_fontsize,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    ax.legend(
        title="Presynaptic cells\nper starter",
        fontsize=tick_fontsize,
        title_fontsize=tick_fontsize,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        handlelength=1,
        alignment="left"
    )
    despine(ax)
    # Define forward and inverse transforms:
    # fraction -> absolute density, and absolute density -> fraction
    # def fraction_to_density(x):
    #     return x * v1_cell_density

    # def density_to_fraction(x):
    #     return x / v1_cell_density

    # Create a secondary x-axis at the top that shows absolute density
    # ax2 = ax.secondary_xaxis(
    #     "top", functions=(fraction_to_density, density_to_fraction)
    # )
    # ax2.set_xscale("log")
    # ax2.set_xlabel(
    #     "Density of starter neurons (mm$^{-3}$)",
    #     fontsize=label_fontsize,
    # )

    # # Optionally set the secondary axis' x-limits to match the transformed primary limits
    # ax2.set_xlim(fraction_to_density(x_range[0]), fraction_to_density(x_range[1]))
    # ax2.tick_params(
    #     axis="both",
    #     which="major",
    #     labelsize=tick_fontsize,
    # )