import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np


def plot_flat_ml_rv_cells(
    cells,
    ax=None,
    label_fontsize=7,
    tick_fontsize=5,
    presyn_size=1,
    starter_size=2,
    presyn_alpha=0.7,
    legend_height=1.8,
):
    """
    Plot the medio-lateral and dorso-ventral coordinates of cells on the flatmap.

    Args:
        cells (pd.DataFrame): DataFrame containing the cell data.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        label_fontsize (int): Font size for the axis labels.
        tick_fontsize (int): Font size for the axis ticks.
        presyn_size (int): Size of the presynaptic cell markers.
        starter_size (int): Size of the starter cell markers.
        presyn_alpha (float): Transparency of the presynaptic cell markers.
    """
    # Prep data
    cells = cells[cells["main_barcode"].notna()]
    cells["cortical_area"] = cells["cortical_area"].astype("category")
    cells["cortical_layer"] = cells["cortical_layer"].astype("category")
    starter_cells = cells[cells["is_starter"]]
    categories = cells["cortical_area"].cat.categories
    n_categories = len(categories)

    # Map categories to their respective colormap values
    cmap = cm.get_cmap("nipy_spectral_r", n_categories)
    color_mapping = {
        category: cmap(i / (n_categories - 1)) for i, category in enumerate(categories)
    }
    legend_patches = [
        mpatches.Patch(color=color_mapping[category], label=category)
        for category in categories
    ]

    x_coords = cells["flatmap_dorsal_x"]
    y_coords = cells["normalised_depth"] / 2
    x_starters = starter_cells["flatmap_dorsal_x"]
    y_starters = starter_cells["normalised_depth"] / 2

    # Plot
    ax.scatter(
        x_coords,
        y_coords,
        alpha=presyn_alpha,
        s=presyn_size,
        linewidths=0,
        c=cells["cortical_area"].cat.codes.map(lambda x: color_mapping[categories[x]]),
        rasterized=True,
    )

    ax.scatter(
        x_starters,
        y_starters,
        s=starter_size,
        edgecolors="none",
        c="black",
        rasterized=True,
    )
    ax.set_xlabel("Medio-lateral coordinates (µm)", fontsize=label_fontsize)
    ax.set_ylabel("Cortical Depth (µm)", fontsize=label_fontsize)
    ax.set_xlim(16000, 26000)
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    # Filter out unwanted categories
    legend_patches = [
        mpatches.Patch(color=color_mapping[category], label=category)
        for category in categories
        if category not in ["non_cortical", "hippocampal", "fiber_tract"]
    ]

    # Modify the legend placement and format
    ax.legend(
        handles=legend_patches,
        title="Cortical Area",
        loc="upper center",
        # prop={'size': 4},
        bbox_to_anchor=(0.5, legend_height),
        fontsize=tick_fontsize,
        title_fontsize=tick_fontsize,
        ncol=4,
    )
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)


def plot_rabies_cells(
    ax_interest,
    ax_density,
    all_cell_properties,
    label_fontsize=10,
    tick_fontsize=8,
):
    layer_tops = {
        "2/3": 116.8406715462,
        "4": 349.9050202564,
        "5": 477.8605504893,
        "6a": 717.1835081307,
        "6b": 909.8772394508,
        "wm": 957.0592130899,
    }

    current_max = 2000.0
    target_max = layer_tops["wm"]
    norm_factor = target_max / current_max

    x_all = all_cell_properties["flatmap_dorsal_x"]
    y_all = all_cell_properties["normalised_layers"]

    cells = all_cell_properties[all_cell_properties["main_barcode"].notna()]
    cells["cortical_area"] = cells["cortical_area"].astype("category")
    cells["cortical_layer"] = cells["cortical_layer"].astype("category")
    starter_cells = cells[cells["is_starter"]]

    x_coords = cells["flatmap_dorsal_x"]
    y_coords = cells["normalised_layers"]
    x_starters = starter_cells["flatmap_dorsal_x"]
    y_starters = starter_cells["normalised_layers"]

    y_all = y_all * norm_factor
    y_coords = y_coords * norm_factor
    y_starters = y_starters * norm_factor

    x_min, x_max = 19800, 21800
    y_min, y_max = 957.0592130899, y_coords.min()

    mask_region_all = (
        (x_all >= x_min) & (x_all <= x_max) & (y_all <= y_min) & (y_all >= y_max)
    )
    mask_region = (
        (x_coords >= x_min)
        & (x_coords <= x_max)
        & (y_coords <= y_min)
        & (y_coords >= y_max)
    )
    mask_region_starters = (
        (x_starters >= x_min)
        & (x_starters <= x_max)
        & (y_starters <= y_min)
        & (y_starters >= y_max)
    )
    x_all_sub = x_all[mask_region_all]
    y_all_sub = y_all[mask_region_all]
    x_sub = x_coords[mask_region]
    y_sub = y_coords[mask_region]
    c_sub = cells["cortical_layer"][mask_region]
    x_starters = x_starters[mask_region_starters]
    y_starters = y_starters[mask_region_starters]

    scatter = ax_interest.scatter(
        x_sub,
        y_sub,
        s=3,
        edgecolors="none",
        c=c_sub.cat.codes,
        cmap="tab20",
        rasterized=True,
    )
    ax_interest.scatter(
        x_starters, y_starters, s=8, edgecolors="none", c="black", rasterized=True
    )

    ax_interest.set_xlim(x_min, x_max)
    ax_interest.set_ylim(y_min, y_max)
    # ax_interest.set_title("Rabies barcoded cells", fontsize=label_fontsize)
    ax_interest.set_xlabel("Medio-lateral coordinates (µm)", fontsize=label_fontsize)
    ax_interest.set_ylabel("Cortical Depth (µm)", fontsize=label_fontsize)
    for layer, z in layer_tops.items():
        ax_interest.axhline(z, c="black", lw=0.5)

    unique_codes = np.unique(cells["cortical_layer"].cat.codes)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=scatter.cmap(scatter.norm(code)),
            markersize=5,
            linestyle="None",
        )
        for code in unique_codes
    ]
    labels = cells["cortical_layer"].cat.categories[unique_codes]

    # Define the items to exclude from legend
    exclude = ["TH", "fiber_tract", "non_cortical", "hippocampal"]
    filtered_handles_labels = [
        (h, l) for h, l in zip(handles, labels) if l not in exclude
    ]
    handles, labels = (
        zip(*filtered_handles_labels) if filtered_handles_labels else ([], [])
    )

    ax_interest.legend(
        handles,
        labels,
        title="Cell types",
        loc="lower center",
        fontsize=tick_fontsize,
        bbox_to_anchor=(0.5, -0.4),
        ncol=3,
    )

    def plot_density(ax, y_filtered, color, label):
        if len(y_filtered) > 2:
            counts, bin_edges = np.histogram(
                y_filtered,
                bins=50,
                density=True,
            )
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.plot(
                counts,
                bin_centers,
                color=color,
                label=label,
                alpha=0.8,
                drawstyle="steps-post",
                linewidth=2,
            )
            ax.fill_betweenx(bin_centers, 0, counts, color=color, alpha=0.1, step="pre")

    ax_secondary = ax_density.twiny()

    def plot_normalized_density(ax_secondary, y_all, y_rab, label="Normalized Density"):
        if len(y_all) > 2 and len(y_rab) > 2:
            bin_edges = np.linspace(
                min(y_all.min(), y_rab.min()), max(y_all.max(), y_rab.max()), 51
            )
            counts_all, _ = np.histogram(y_all, bins=bin_edges, density=True)
            counts_rab, _ = np.histogram(y_rab, bins=bin_edges, density=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized_density = np.divide(
                    counts_rab,
                    counts_all,
                    out=np.zeros_like(counts_rab),
                    where=counts_all > 0,
                )
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax_secondary.plot(
                normalized_density,
                bin_centers,
                color="red",
                label=label,
                alpha=0.8,
                drawstyle="steps-post",
                linewidth=2,
            )
            ax_secondary.fill_betweenx(
                bin_centers, 0, normalized_density, color="red", alpha=0.2, step="pre"
            )

    plot_density(ax_density, y_starters, "black", label="Density -\nStarter Cells")
    ax_density.set_xlabel("Fraction of starter cells", fontsize=label_fontsize)
    ax_density.set_yticks([])

    ax_density.set_ylim(y_min, y_max)
    ax_density.set_xlim(0, 0.0035)
    ax_secondary.set_xlim(0, 2.1)
    ax_secondary.set_xlabel("Ratio of density", fontsize=label_fontsize, color="red")
    ax_secondary.tick_params(axis="x", colors="red", labelsize=tick_fontsize)
    for layer, z in layer_tops.items():
        ax_density.axhline(z, c="black", lw=0.5)
    plot_normalized_density(
        ax_secondary, y_all_sub, y_sub, label="Density Ratio -\n(Rabies / All Cells)"
    )
    ax_secondary.set_ylabel(
        "Density Ratio (Rabies / All Cells)", fontsize=label_fontsize, labelpad=10
    )
    ax_interest.tick_params(axis="both", labelsize=tick_fontsize)
    ax_density.tick_params(axis="both", labelsize=tick_fontsize)

    # Fetch handles and labels from both axes
    handles_density, labels_density = ax_density.get_legend_handles_labels()
    handles_secondary, labels_secondary = ax_secondary.get_legend_handles_labels()
    # Combine handles and labels from both plots
    handles = handles_density + handles_secondary
    labels = labels_density + labels_secondary
    ax_density.legend(
        handles,
        labels,
        loc="lower center",
        fontsize=tick_fontsize,
        bbox_to_anchor=(0.5, -0.4),
    )
