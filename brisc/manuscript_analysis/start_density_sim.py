import numpy as np
from scipy.optimize import minimize_scalar
from .utils import despine


def plot_starter_spread_sim(
    density=np.logspace(-4, 0, num=50),
    ns=[10, 20, 40],
    starter_spread_probability=0.05,
    v1_cell_density=150e3,
    x_range=(1e-4, 1),
    ax=None,
    label_fontsize=12,
    tick_fontsize=10,
    line_width=0.5,
    colors=("lightsalmon", "tomato", "red"),
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

    # Horizontal dashed line at starter_spread_probability
    ax.axhline(
        starter_spread_probability,
        linestyle="dashed",
        color="black",
        lw=line_width * 0.7,
    )

    # Vertical dashed line at the cross of the n=20 line and the proba
    def cost(density):
        return ((1 - (1 - density) ** 20) - starter_spread_probability) ** 2

    density_threshold = minimize_scalar(cost)
    assert density_threshold.success
    density_threshold = density_threshold.x
    ax.axvline(
        density_threshold,
        linestyle="dashed",
        color="black",
        lw=line_width * 0.7,
    )
    # Lines for different cell/starter number
    for n, color in zip(ns, colors):
        p = 1 - (1 - density) ** n
        ax.loglog(density, p, label=str(n), lw=line_width, c=color)

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
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])

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
        alignment="left",
    )
    despine(ax)

    # Define forward and inverse transforms:
    # fraction -> absolute density, and absolute density -> fraction
    def fraction_to_density(x):
        return x * v1_cell_density

    def density_to_fraction(x):
        return x / v1_cell_density

    print(
        f"For 20 presynaptics per starter, the {starter_spread_probability*100:.0f}% "
        f"threshold is crossed at p={density_threshold*100:.2f}%, or "
        f"{fraction_to_density(density_threshold):.0f} cells/mm$^3$"
    )

    # Create a secondary x-axis at the top that shows absolute density
    ax2 = ax.secondary_xaxis(
        "top", functions=(fraction_to_density, density_to_fraction)
    )
    ax2.set_xscale("log")
    ax2.set_xlabel(
        "Density of starter\nneurons (mm$^{-3}$)",
        fontsize=label_fontsize,
    )

    # Optionally set the secondary axis' x-limits to match the transformed primary limits
    ax2.set_xlim(fraction_to_density(x_range[0]), fraction_to_density(x_range[1]))
    ax2.set_xticks([1e2, 1e3, 1e4, 1e5])
    ax2.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )
