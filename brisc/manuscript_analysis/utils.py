

def despine(ax):
    """
    Remove right and top spines from a matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): Axes to despine.
    """
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)