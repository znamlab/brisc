import os
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

    The function follows this priority to find the path:
    1. Environment variable `BRISCO_OUTPUT_DIR`.
    2. Shared lab `presentations` directory (via `flexiznam`).
    3. The provided `data_root`.
    4. A local `figures` directory in the current working directory.

    Args:
        data_root (str or pathlib.Path, optional): The root directory for the
            data. Used as fallback if no shared path is found.
            Defaults to None.

    Returns:
        pathlib.Path: The full path to the output folder.
    """
    # 1. Check for environment variable override
    env_path = os.environ.get("BRISCO_OUTPUT_DIR")
    if env_path:
        save_path = Path(env_path)
    else:
        try:
            # 2. Try to find the shared lab location (presentations folder)
            import flexiznam as flz

            # Go up two levels from the rabies_barcoding processed path to find the project root
            # /nemo/project/proj-znamenp-barseq/processed/rabies_barcoding -> /nemo/project/proj-znamenp-barseq/
            base = flz.get_processed_path("rabies_barcoding").parent.parent
            save_path = base / "presentations" / "becalick_2025"
        except Exception:
            # 3. Fallback to data_root or local directory
            if data_root is not None:
                save_path = Path(data_root) / "becalick_2025"
            else:
                save_path = Path.cwd() / "figures" / "becalick_2025"

    save_path.mkdir(exist_ok=True, parents=True)
    return save_path


def get_path(pathname, data_root=None):
    """
    Get the path to the processed data.

    If data_root is provided, returns data_root / pathname.
    Otherwise, attempts to find the path via flexiznam. Returns None if
    flexiznam is unavailable or the path cannot be found.

    Args:
        pathname (str): The name of the data or project.
        data_root (str or pathlib.Path, optional): The root directory.
            Defaults to None.

    Returns:
        pathlib.Path: The path to the processed data, or None if not found.
    """
    if data_root is not None:
        return Path(data_root) / pathname

    try:
        import flexiznam as flz

        return flz.get_processed_path(pathname)
    except Exception:
        return None
