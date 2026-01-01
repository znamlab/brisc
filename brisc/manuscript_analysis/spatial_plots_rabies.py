import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import brainglobe_atlasapi as bga
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from cricksaw_analysis import atlas_utils
from .flatmap_projection import get_avg_layer_depth


def prepare_area_labels(
    xpos=860,
    structures=[
        "root",
        "CTX",
        "MB",
        "DG",
        "DG-mo",
        "DG-sg",
        "SCdg",
        "SCdw",
        "SCig",
        "SCiw",
        "SCop",
        "SCsg",
        "SCzo",
        "PAG",
        "MRN",
        "TH",
        "RN",
    ],
    atlas_size=10,
):
    atlas = bga.bg_atlas.BrainGlobeAtlas(f"allen_mouse_{atlas_size}um")
    bin_image = atlas.get_structure_mask(atlas.structures["root"]["id"])[xpos, :, :]
    for i, structure in enumerate(tqdm(structures, desc="Preparing area labels")):
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
        "AUDp": "limegreen",
        "AUDpo": "mediumseagreen",
        "AUDv": "springgreen",
        "RSP": "darkorchid",
        "TEa": "forestgreen",
        "TH": "orangered",
        "VISal": "aquamarine",
        "VISl": "darkturquoise",
        "VISli": "mediumaquamarine",
        "VISp": "deepskyblue",
        "VISpm": "royalblue",
        "fiber_tract": "gray",
    },
    presynaptic_marker_size=1,
    starter_marker_size=2,
    invert_xaxis=True,
    drop_areas=("hippocampal", "fiber_tract"),
    rasterized=True,
):
    ax_coronal.contour(
        bin_image,
        levels=np.arange(0.5, np.max(bin_image) + 1, 0.5),
        colors="black",
        linewidths=0.5,
        zorder=0,
    )
    cells_df["inside"] = cells_df["cortical_area"].apply(
        lambda area: area not in drop_areas and not pd.isnull(area)
    )
    cells_inside = cells_df[
        (cells_df["inside"] == True) & (cells_df["area"] != "outside")
    ]
    cells_inside["cortical_area"] = cells_inside["cortical_area"].astype("category")
    cells_inside["cortical_layer"] = cells_inside["cortical_layer"].astype("category")
    starters = cells_inside[cells_inside["is_starter"] == True]

    areas = cells_inside["cortical_area"].cat.categories
    ax_coronal.scatter(
        cells_inside["ara_z"] * 1000 / atlas_size,
        cells_inside["ara_y"] * 1000 / atlas_size,
        s=presynaptic_marker_size,
        linewidths=0,
        c=cells_inside["cortical_area"].cat.codes.map(lambda x: area_colors[areas[x]]),
        zorder=1,
        alpha=0.3,
        rasterized=rasterized,
    )
    ax_coronal.scatter(
        starters["ara_z"] * 1000 / atlas_size,
        starters["ara_y"] * 1000 / atlas_size,
        s=starter_marker_size,
        edgecolors="none",
        c="black",
        zorder=2,
        alpha=0.6,
        rasterized=rasterized,
    )
    ax_coronal.plot([980, 1080], [50, 50], color="black", lw=3)

    ax_coronal.set_xlim(570, 1100)
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
        rasterized=rasterized,
    )
    ax_flatmap.scatter(
        starters["flatmap_x"],
        starters["flatmap_y"],
        s=starter_marker_size,
        edgecolors="none",
        c="black",
        zorder=2,
        alpha=0.6,
        rasterized=rasterized,
    )
    ax_flatmap.plot([100, 200], [1330, 1330], color="black", lw=3)
    ax_flatmap.set_ylim(740, 1350)
    ax_flatmap.set_xlim(100, 1200)
    ax_flatmap.invert_yaxis()
    ax_flatmap.set_axis_off()
    ax_flatmap.set_aspect("equal")
    if invert_xaxis:
        ax_flatmap.invert_xaxis()
        ax_coronal.invert_xaxis()

    legend_patches = [
        mpatches.Patch(color=area_color, label=area_name)
        for area_name, area_color in area_colors.items()
    ]

    # Modify the legend placement and format
    ax_flatmap.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=[0, 0],
        frameon=False,
        handlelength=1,
        ncols=3,
        fontsize=legend_fontsize,
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
    invert_xaxis=True,
    rasterized=True,
    starter2presynaptics_kwargs=None,
):
    # cells_df = cells_df[
    #     cells_df["cortical_area"].apply(lambda area: not pd.isnull(area))
    # ]

    ax_coronal.contour(
        bin_image,
        levels=np.arange(0.5, np.max(bin_image) + 1, 0.5),
        colors="black",
        linewidths=0.5,
        zorder=0,
    )
    starters = cells_df[cells_df["is_starter"] == True]
    ax_coronal.scatter(
        cells_df["ara_z"] * 1000 / atlas_size,
        cells_df["ara_y"] * 1000 / atlas_size,
        s=all_cells_marker_size,
        linewidths=0,
        c="gray",
        alpha=0.15,
        zorder=1,
        label="All barcoded cells",
        rasterized=rasterized,
    )

    atlas_utils.plot_flatmap(
        ax_flatmap,
        hemisphere="right",
        ccf_streamlines_folder=None,
    )
    ax_flatmap.scatter(
        cells_df["flatmap_x"],
        cells_df["flatmap_y"],
        s=all_cells_marker_size * 2,
        linewidths=0,
        c="gray",
        zorder=1,
        alpha=0.2,
        rasterized=rasterized,
    )
    if starter2presynaptics_kwargs is not None:
        linekw = dict(
            color="black",
            lw=0.5,
            zorder=-3,
            rasterized=rasterized,
        )
        linekw.update(starter2presynaptics_kwargs)
    for barcode, color in zip(barcodes, barcode_colors):
        this_barcode = cells_df[
            cells_df["all_barcodes"].apply(lambda bcs: barcode in bcs)
        ]
        ax_coronal.scatter(
            this_barcode["ara_z"] * 1000 / atlas_size,
            this_barcode["ara_y"] * 1000 / atlas_size,
            alpha=1,
            linewidths=0,
            s=presynaptic_marker_size,
            c=color,
            zorder=2,
            rasterized=rasterized,
        )
        ax_flatmap.scatter(
            this_barcode["flatmap_x"],
            this_barcode["flatmap_y"],
            alpha=1,
            linewidths=0,
            s=presynaptic_marker_size * 2,
            c=color,
            zorder=2,
            rasterized=rasterized,
        )
        starters = this_barcode[this_barcode["is_starter"] == True]
        print(f"barcode {barcode} in starters {starters.index.values}")
        ax_coronal.scatter(
            starters["ara_z"] * 1000 / atlas_size,
            starters["ara_y"] * 1000 / atlas_size,
            s=starter_marker_size,
            alpha=1,
            edgecolors="black",
            marker=starter_marker,
            c=color,
            linewidths=1,
            label=barcode,
            zorder=3,
            rasterized=rasterized,
        )
        ax_flatmap.scatter(
            starters["flatmap_x"],
            starters["flatmap_y"],
            s=starter_marker_size,
            edgecolors="black",
            marker=starter_marker,
            c=color,
            linewidths=1,
            label=barcode,
            zorder=3,
            rasterized=rasterized,
        )
        if starter2presynaptics_kwargs is not None:
            assert len(starters) == 1, f"Multiple starters for barcode {barcode}"
            starter = starters.iloc[0]
            linekw["color"] = color
            for i, presyn in this_barcode.iterrows():
                ax_coronal.plot(
                    [
                        starter["ara_z"] * 1000 / atlas_size,
                        presyn["ara_z"] * 1000 / atlas_size,
                    ],
                    [
                        starter["ara_y"] * 1000 / atlas_size,
                        presyn["ara_y"] * 1000 / atlas_size,
                    ],
                    **linekw,
                )
                ax_flatmap.plot(
                    [starter["flatmap_x"], presyn["flatmap_x"]],
                    [starter["flatmap_y"], presyn["flatmap_y"]],
                    **linekw,
                )
    ax_coronal.plot([980, 1080], [70, 70], color="black", lw=3)
    ax_coronal.set_xlim(570, 1100)
    ax_coronal.set_ylim(450, 0)
    ax_coronal.set_axis_off()
    ax_coronal.set_aspect("equal")

    ax_flatmap.plot([640, 740], [1130, 1130], color="black", lw=3)
    ax_flatmap.set_ylim(850, 1250)
    ax_flatmap.set_xlim(250, 980)
    ax_flatmap.invert_yaxis()
    ax_flatmap.set_axis_off()
    ax_flatmap.set_aspect("equal")

    if invert_xaxis:
        ax_flatmap.invert_xaxis()
        ax_coronal.invert_xaxis()
    # Modify the legend placement and format
    ax_coronal.legend(
        loc="lower left",
        bbox_to_anchor=[1.0, 0.05],
        frameon=False,
        fontsize=legend_fontsize,
        ncols=2,
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
    rasterized=True,
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
        rasterized=rasterized,
    )

    ax.scatter(
        x_starters,
        y_starters,
        s=starter_size,
        edgecolors="none",
        c="black",
        rasterized=rasterized,
    )
    ax.set_xlabel("Medio-lateral coordinates (µm)", fontsize=label_fontsize)
    ax.set_ylabel("Cortical Depth (µm)", fontsize=label_fontsize)
    ax.set_xlim(16000, 26000)
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()

    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)


def plot_layer_distribution(
    ax_interest,
    ax_density,
    cells_df,
    label_fontsize=10,
    tick_fontsize=8,
    show_cells=True,
    rasterized=True,
):
    layer_tops = get_avg_layer_depth()
    layer_tops["1"] = 0.0

    y_min, y_max = 1000, 0
    cells_df = cells_df[cells_df["cortical_area"] == "VISp"]
    scale = 10  # micron per px
    if show_cells:
        ax_interest.scatter(
            cells_df["flatmap_x"] * scale,
            cells_df["normalised_layers"] * scale,
            s=2,
            edgecolors="none",
            c="gray",
            alpha=0.5,
            label="Presynaptic cells",
            rasterized=rasterized,
        )
        ax_interest.scatter(
            cells_df[cells_df["is_starter"]]["flatmap_x"] * scale,
            cells_df[cells_df["is_starter"]]["normalised_layers"] * scale,
            s=3,
            edgecolors="none",
            c="black",
            label="Starter cells",
            rasterized=rasterized,
        )
        ax_interest.set_xlim(18750, 22750)
        ax_interest.set_ylim(y_min, y_max)
        ax_interest.set_xlabel("Medio-lateral\nlocation (mm)", fontsize=label_fontsize)
        ax_interest.set_ylabel("Cortical depth (µm)", fontsize=label_fontsize)
        for layer, z in layer_tops.items():
            ax_interest.axhline(z, c="black", lw=0.5, linestyle="--")
        ax_interest.set_xticks(
            (18750, 19750, 20750, 21750, 22750), labels=[-2, -1, 0, 1, 2]
        )
        ax_interest.tick_params(axis="both", labelsize=tick_fontsize)

    sns.violinplot(
        y=cells_df["normalised_layers"] * scale,
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
        loc="lower left",
        fontsize=tick_fontsize,
        bbox_to_anchor=(-0.3, 1.0),
        frameon=False,
        handlelength=1,
        ncol=2 if show_cells else 1,
    )
    ax_density.set_xlabel("Cell\ndensity", fontsize=label_fontsize)
    ax_density.set_ylim(y_min, y_max)
    ax_density.tick_params(axis="both", labelsize=tick_fontsize)
    for layer, z in layer_tops.items():
        ax_density.axhline(z, c="black", lw=0.5, linestyle="--")
    # Combine handles and labels from both plots
    # add yticks on the right
    if show_cells:
        ax_density.set_ylabel("")
    else:
        ax_density.set_ylabel("Cortical depth (µm)", fontsize=label_fontsize)

    # prepend a 0 to layer tops and find layer centres
    ax_right = ax_density.twinx()
    ax_right.set_ylim(y_min, y_max)
    layer_edges = np.array(list(layer_tops.values()))
    layer_centres = (layer_edges[1:] + layer_edges[:-1]) / 2
    ax_right.set_yticks(layer_centres, labels=list(layer_tops.keys())[:-1])
    ax_right.tick_params(axis="both", labelsize=tick_fontsize, length=0)
