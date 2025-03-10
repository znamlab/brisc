import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_brainrender_3d(ax, image_path):
    """
    Load a .png file from image_path and display it on the given Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to display the image.
    image_path : str or pathlib.Path
        The path to the .png image file.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axis with the image displayed.
    """
    # Load the image data
    img = mpimg.imread(image_path)

    # Display the image
    ax.imshow(img)

    # Remove axis ticks if desired
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
