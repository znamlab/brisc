import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import brainglobe_atlasapi as bga
import pandas as pd
import seaborn as sns
from cricksaw_analysis import atlas_utils


def prepare_area_labels(
    xpos=860,
    structures=[ "root", "CTX", "MB", "DG", "DG-mo", "DG-sg", "SCdg", "SCdw", "SCig", "SCiw", "SCop", "SCsg", "SCzo", "PAG", "MRN", "TH", "RN",],
    atlas_size=10
):
    atlas = bga.bg_atlas.BrainGlobeAtlas(f"allen_mouse_{atlas_size}um")    
    bin_image = atlas.get_structure_mask(atlas.structures["root"]["id"])[xpos, :, :]
    for i, structure in enumerate(structures):
        mask = atlas.get_structure_mask(atlas.structures[structure]["id"])[xpos, :, :]
        bin_image[mask > 0] = i + 1
    return bin_image


def plot_all_rv_cells(
    cells_df,
    ax_coronal,
    ax_flatmap,
    bin_image,
    legend_fontsize=6,
    atlas_size=10,
    area_colors={
        'AUDp': "limegreen", 
        'AUDpo': "mediumseagreen", 
        'AUDv': "springgreen", 
        'RSP': "darkorchid", 
        'TEa': "forestgreen", 
        'TH': "orangered", 
        'VISal': "aquamarine", 
        'VISl': "darkturquoise", 
        'VISli': "mediumaquamarine",
        'VISp': "deepskyblue", 
        'VISpm': "royalblue", 
        'fiber_tract': "gray",
    },
    presynaptic_marker_size=1,
    starter_marker_size=2,
):
    ax_coronal.contour(
        bin_image, 
        levels=np.arange(0.5, np.max(bin_image) + 1, 0.5), 
        colors="slategray", 
        linewidths=0.5,
        zorder=0,
    )
    cells_df["inside"] = cells_df["cortical_area"].apply(lambda area: area != "hippocampal" and not pd.isnull(area))
    cells_inside = cells_df[(cells_df["inside"] == True) & (cells_df["area"] != "outside")]
    cells_inside["cortical_area"] = cells_inside["cortical_area"].astype("category")
    cells_inside["cortical_layer"] = cells_inside["cortical_layer"].astype("category")
    starters = cells_inside[cells_inside["is_starter"] == True]

    areas = cells_inside["cortical_area"].cat.categories
    ax_coronal.scatter(
        cells_inside["ara_z"]  * 1000 / atlas_size, 
        cells_inside["ara_y"] * 1000 / atlas_size, 
        s=presynaptic_marker_size,
        linewidths=0,
        c=cells_inside["cortical_area"].cat.codes.map(lambda x: area_colors[areas[x]]),
        zorder=1,
        alpha=0.3,
    )
    ax_coronal.scatter(
        starters["ara_z"]  * 1000 / atlas_size, 
        starters["ara_y"] * 1000 / atlas_size, 
        s=starter_marker_size,
        edgecolors="none",
        c="black",
        zorder=2,
        alpha=0.6,        
    )
    ax_coronal.set_xlim(550, 1150)
    ax_coronal.set_ylim(450, 0)
    ax_coronal.set_axis_off()
    ax_coronal.set_aspect("equal")

    atlas_utils.plot_flatmap(
        ax_flatmap,
        hemisphere="right",
        ccf_streamlines_folder=None,
    )
    ax_flatmap.scatter(
        cells_inside["flatmap_x"],
        cells_inside["flatmap_y"],
        s=presynaptic_marker_size,
        linewidths=0,
        c=cells_inside["cortical_area"].cat.codes.map(lambda x: area_colors[areas[x]]),
        zorder=1,
        alpha=0.3,
    )
    ax_flatmap.scatter(
        starters["flatmap_x"],
        starters["flatmap_y"],
        s=starter_marker_size,
        edgecolors="none",
        c="black",
        zorder=2,
        alpha=0.6,
    )
    ax_flatmap.set_ylim(800, 1350)
    ax_flatmap.set_xlim(100, 1200)
    ax_flatmap.invert_yaxis()
    ax_flatmap.set_axis_off()
    ax_flatmap.set_aspect("equal")


    # Filter out unwanted categories
    legend_patches = [
        mpatches.Patch(color=area_colors[area], label=area)
        for area in areas
    ]

    # Modify the legend placement and format
    ax_flatmap.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=[0, 0],
        frameon=False,
        handlelength=1,
        ncols=3,
        fontsize=legend_fontsize
    )


def plot_example_barcodes(
    cells_df,
    ax_coronal,
    ax_flatmap,
    bin_image,
    barcodes=(),
    barcode_colors=(),
    legend_fontsize=6,
    atlas_size=10,
    starter_marker_size=15,
    presynaptic_marker_size=2,
    all_cells_marker_size=1,
    starter_marker="o",
):
    cells_df = cells_df[
        cells_df["cortical_area"].apply(lambda area: not pd.isnull(area))
    ]

    ax_coronal.contour(
        bin_image, 
        levels=np.arange(0.5, np.max(bin_image) + 1, 0.5), 
        colors="slategray", 
        linewidths=0.5,
        zorder=0,
    )
    starters = cells_df[cells_df["is_starter"] == True]
    ax_coronal.scatter(
        cells_df["ara_z"]  * 1000 / atlas_size, 
        cells_df["ara_y"] * 1000 / atlas_size, 
        s=all_cells_marker_size,
        linewidths=0,
        c="gray",
        alpha=0.15,
        zorder=1,
        label="All barcoded cells"
    )

    atlas_utils.plot_flatmap(
        ax_flatmap,
        hemisphere="right",
        ccf_streamlines_folder=None,
    )
    ax_flatmap.scatter(
        cells_df["flatmap_x"],
        cells_df["flatmap_y"],
        s=all_cells_marker_size,
        linewidths=0,
        c="gray",
        zorder=1,
        alpha=0.15,
    )

    for barcode, color in zip(barcodes, barcode_colors):
        this_barcode = cells_df[cells_df["all_barcodes"].apply(lambda bcs: barcode in bcs)]
        ax_coronal.scatter(
            this_barcode["ara_z"]  * 1000 / atlas_size, 
            this_barcode["ara_y"] * 1000 / atlas_size, 
            alpha=1,
            linewidths=0,
            s=presynaptic_marker_size,
            c=color,
            zorder=2,
        )
        ax_flatmap.scatter(
            this_barcode["flatmap_x"],
            this_barcode["flatmap_y"],
            alpha=1,
            linewidths=0,
            s=presynaptic_marker_size,
            c=color,
            zorder=2,
        )
        starters = this_barcode[this_barcode["is_starter"] == True]
        print(f"barcode {barcode} in starters {starters.index.values}")
        ax_coronal.scatter(
            starters["ara_z"]  * 1000 / atlas_size, 
            starters["ara_y"] * 1000 / atlas_size, 
            s=starter_marker_size,
            alpha=1,            
            edgecolors="white",
            marker=starter_marker,
            c=color,
            linewidths=1,
            label=barcode,
            zorder=3,
        )
        ax_flatmap.scatter(
            starters["flatmap_x"],
            starters["flatmap_y"],
            s=starter_marker_size * 0.7,
            edgecolors="white",
            marker=starter_marker,
            c=color,
            linewidths=1,
            label=barcode,
            zorder=3,
        )

    ax_coronal.set_xlim(550, 1150)
    ax_coronal.set_ylim(450, 0)
    ax_coronal.set_axis_off()
    ax_coronal.set_aspect("equal")

    ax_flatmap.set_ylim(790, 1350)
    ax_flatmap.set_xlim(100, 1200)
    ax_flatmap.invert_yaxis()
    ax_flatmap.set_axis_off()
    ax_flatmap.set_aspect("equal")

    # Modify the legend placement and format
    ax_coronal.legend(
        loc="lower left",
        bbox_to_anchor=[0.9, 0],
        frameon=False,
        fontsize=legend_fontsize,
        ncols=3,
    )


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
    cells = cells[cells["all_barcodes"].notna()]
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

    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)


def plot_rabies_cells(
    ax_interest,
    ax_density,
    cells_df,
    label_fontsize=10,
    tick_fontsize=8,
):
    layer_tops = {
        "1": 0.0,
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
    x_min, x_max = 19800, 21800
    y_min, y_max = 957.0592130899, cells_df["normalised_layers"].min()    
    # cells_df = cells_df[
    #     (cells_df["flatmap_dorsal_x"] >= x_min)
    #     & (cells_df["flatmap_dorsal_x"] <= x_max)
    #     & (cells_df["normalised_layers"] * norm_factor <= y_min)
    # ]
    cells_df = cells_df[cells_df["cortical_area"] == "VISp"]
    ax_interest.scatter(
        cells_df["flatmap_dorsal_x"],
        cells_df["normalised_layers"] * norm_factor,
        s=2,
        edgecolors="none",
        c="gray",
        alpha=0.5,
        label="Presynaptic cells",
    )
    ax_interest.scatter(
        cells_df[cells_df["is_starter"]]["flatmap_dorsal_x"],
        cells_df[cells_df["is_starter"]]["normalised_layers"] * norm_factor,
        s=3, 
        edgecolors="none", 
        c="black", 
        label="Starter cells",
    )
    ax_interest.set_xticks([0, 1000])
    ax_interest.set_xlim(x_min, x_max)
    ax_interest.set_ylim(y_min, y_max)
    ax_interest.set_xlabel("M-L\ncoordinates (µm)", fontsize=label_fontsize)
    ax_interest.set_ylabel("Cortical depth (µm)", fontsize=label_fontsize)
    for layer, z in layer_tops.items():
        ax_interest.axhline(z, c="black", lw=0.5, linestyle="--")

    sns.violinplot(
        y=cells_df["normalised_layers"] * norm_factor, 
        hue=cells_df["is_starter"], 
        split=True, 
        fill=True,
        alpha=1,
        ax=ax_density,
        bw_adjust=0.5,        
        linewidth=1,
        inner=None,
        palette={True: "black", False: "gray"},
        hue_order=[True, False],
    )
    ax_density.legend(
        labels=["Starter cells", "Presynaptic cells"],
        loc="lower center",
        fontsize=tick_fontsize,
        bbox_to_anchor=(-0.2, 1.0),
        frameon=False,
        handlelength=1,
        ncol=2,
    )
    ax_density.set_xlabel("Cell density", fontsize=label_fontsize)
    ax_density.set_yticks([])
    ax_density.set_xticks([])

    ax_density.set_ylim(y_min, y_max)
    ax_density.set_ylabel("")
    ax_interest.tick_params(axis="both", labelsize=tick_fontsize)
    ax_density.tick_params(axis="both", labelsize=tick_fontsize)
    for layer, z in layer_tops.items():
        ax_density.axhline(z, c="black", lw=0.5, linestyle="--")
    # Combine handles and labels from both plots
    # add yticks on the right
    ax_density.yaxis.tick_right()
    ax_density.yaxis.set_label_position("right")
    # prepend a 0 to layer tops and find layer centres
    layer_edges = np.array(list(layer_tops.values()))
    layer_centres = (layer_edges[1:] + layer_edges[:-1]) / 2
    ax_density.set_yticks(layer_centres, labels=list(layer_tops.keys())[:-1])
    
