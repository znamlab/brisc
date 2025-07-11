from pathlib import Path


def despine(ax):
    """
    Remove right and top spines from a matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): Axes to despine.
    """
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def get_output_folder(data_root=None):
    """Get the path to the manuscript's output folder.

    This function constructs a path to a specific output directory,
    `becalick_2025`, which is used for saving figures and other outputs
    for the manuscript.

    The base path can be specified directly via the `data_root` argument.
    If `data_root` is not provided, the function attempts to determine the
    base path using `flexiznam`.

    Args:
        data_root (str or pathlib.Path, optional): The root directory for the
            data. If None, `flexiznam` is used to determine the path.
            Defaults to None.

    Returns:
        pathlib.Path: The full path to the output folder.
    """
    if data_root is None:
        import flexiznam as flz

        processed = flz.get_processed_path("becalia_rabies_barseq").parent
        processed = processed.parent / "presentations"
    else:
        processed = Path(data_root)
    save_path = processed / "becalick_2025"
    save_path.mkdir(exist_ok=True)
    return save_path


def get_path(pathname, data_root=None):
    """
    Get the path to the processed data.

    Args:
        pathname (str): The name of the data.
        data_root (str): The root directory of the data.
        use_flexiznam (bool): Whether to use flexiznam to get the processed path.

    Returns:
        pathlib.Path: The path to the processed data.
    """
    if data_root is None:
        import flexiznam as flz

        return flz.get_processed_path(pathname)
    else:
        return Path(data_root) / pathname
