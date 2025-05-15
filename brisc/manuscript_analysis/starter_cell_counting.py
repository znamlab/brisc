import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import brainglobe_atlasapi as bga
from cricksaw_analysis import atlas_utils
from cricksaw_analysis.io import load_cellfinder_results
from cricksaw_analysis.atlas_utils import cell_density_by_areas
from iss_preprocess import vis
from scipy.stats import gaussian_kde
from brisc.manuscript_analysis.utils import despine
import tifffile as tf
import flexiznam as flz
import cv2
from czifile import CziFile
from xml.etree import ElementTree


def plot_starter_dilution_densities(
    ax,
    label_fontsize=12,
    tick_fontsize=12,
    processed=Path("/nemo/lab/znamenskiyp/home/shared/projects/"),
):
    """
    Plot the densities of the starter cells for the different dilutions of PHP.eb (only for V1)
    """
    atlas_size = 25
    cortical_areas = "ALL"
    bg_atlas = bga.bg_atlas.BrainGlobeAtlas("allen_mouse_%dum" % atlas_size)
    cdf = atlas_utils.create_ctx_table(bg_atlas)
    project = "rabies_barcoding"
    mouse_csv = processed / project / "mice_list.csv"
    mice_df = pd.read_csv(mouse_csv, skipinitialspace=True)
    mice_df.set_index("Mouse", inplace=True, drop=False)

    if cortical_areas == "ALL":
        cortical_areas = list(cdf.area_acronym.unique())

    summary_density = dict()
    area = "VISp"

    for mouse, m_df in mice_df[mice_df["Virus Batch"] == "A87"].iterrows():
        if mouse in []:
            continue
        mouse_cellfinder_folder = processed / project / mouse / "cellfinder_results"
        if not mouse_cellfinder_folder.is_dir():
            print("No cellfinder folder for %s" % mouse)
            print(mouse_cellfinder_folder)
            continue
        # print("Doing %s" % mouse)
        density_fig_path = (
            processed / project / mouse / ("%s_neighbour_by_distance.png" % mouse)
        )
        summary_density_mouse_path = (
            processed / project / mouse / ("%s_summary_density.csv" % mouse)
        )

        REDO = False
        PLOT_DENSITY = False
        PLOT_SUMMARY = True

        need_data = False
        if REDO:
            need_data = True
        if PLOT_DENSITY and (not density_fig_path.is_file()):
            need_data = True
        if PLOT_SUMMARY and (not summary_density_mouse_path.is_file()):
            need_data = True

        if need_data:
            try:
                cells, _, atlas = load_cellfinder_results(mouse_cellfinder_folder)
            except IOError or FileNotFoundError as err:
                print("Failed to load data: %s" % err)
                continue
            rd = np.array(np.round(cells.values), dtype=int)
            atlas_id = atlas[rd[:, 0], rd[:, 1], rd[:, 2]]
        if PLOT_SUMMARY:
            if REDO or (not summary_density_mouse_path.is_file()):
                assert need_data
                summary_density[mouse] = pd.DataFrame(
                    cell_density_by_areas(atlas_id, atlas)
                )
                summary_density[mouse].to_csv(summary_density_mouse_path)
            else:
                summary_density[mouse] = pd.read_csv(
                    summary_density_mouse_path, index_col=0
                )

    long_format = []
    for mouse, mdata in summary_density.items():
        for area, adata in mdata.items():
            d = dict(
                area=area,
                mouse=mouse,
                dilution=mice_df.loc[mouse, "Dilution"],
                zres=mice_df.loc[mouse, "Zresolution"],
            )
            for w, value in adata.items():
                d["what"] = w
                d["value"] = value
                long_format.append(dict(d))
    long_format = pd.DataFrame(long_format)

    # Extract V1 data only
    v1 = long_format[long_format.area == "VISp"]
    vdf = v1[v1.what == "volume"].set_index("mouse", inplace=False)
    v1 = v1[v1.what == "count"].set_index("mouse", inplace=False)
    v1.columns = ["count" if c == "value" else c for c in v1.columns]
    v1["density"] = v1["count"] / vdf["value"]

    # Remove zres 25
    v1 = v1.loc[v1["zres"] == 8]

    # Plot the data points with dodge
    sns.stripplot(
        data=v1,
        x="dilution",
        y="density",
        order=["1/100", "1/330", "1/1000", "1/3300"],
        dodge=True,
        jitter=True,
        ax=ax,
        color="lightgray",
        edgecolor="black",
        linewidth=0.5,
        size=3,
    )

    # Calculate positions for the boxplots
    positions_v1 = np.arange(len(["1/100", "1/330", "1/1000", "1/3300"]))  # - 0.2

    # Plot the boxplot for v1
    sns.boxplot(
        data=v1,
        x="dilution",
        y="density",
        order=["1/100", "1/330", "1/1000", "1/3300"],
        dodge=False,
        showmeans=True,
        meanline=True,
        meanprops={"color": "grey", "ls": "-", "lw": 2},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        showfliers=False,
        showbox=False,
        showcaps=False,
        width=0.4,
        ax=ax,
        positions=positions_v1,
    )

    plt.setp(ax, ylim=(0))
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=tick_fontsize,
    )
    ax.set_xticklabels(
        ["1/100", "1/330", "1/1000", "1/3300"],
        fontsize=tick_fontsize,
        rotation=45,
    )
    plt.xlabel(
        "AAV hSyn-Cre dilution",
        fontsize=label_fontsize,
    )
    plt.ylabel(
        "tdTomato+ cell\ndensity (mm$^{-3}$)",
        fontsize=label_fontsize,
    )
    despine(ax)


def load_confocal_image(image_fname):
    with CziFile(image_fname) as czi:
        metadata = czi.metadata(raw=False)
        img = np.squeeze(czi.asarray())
    return metadata, img


def plot_starter_confocal(ax, img, metadata):
    """
    Plot the two inset images inside the given axis, one above the other.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        label_fontsize (int, optional): Font size for labels. Defaults to 12.
        tick_fontsize (int, optional): Font size for ticks. Defaults to 10.
    """
    # Define parameters
    ROTATION = 103  # rotation in degrees to put pia at the top of the image
    INSET = [
        2580,
        4800,
        3180,
        None,
    ]
    aspect_ratio = 1.5  # Approximate aspect ratio adjustment
    INSET[-1] = int(INSET[1] + (INSET[2] - INSET[0]) / aspect_ratio * 2)

    scale = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
    scale = {s["Id"]: s["Value"] * 1e6 for s in scale}

    def rotate(image, angle, center=None, scale=1.0, flip=True):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return cv2.flip(rotated, 1) if flip else rotated

    # Create two sub-axes within the given ax
    # inset_axes = [ax.inset_axes([0, 0.5, 1, 0.5]), ax.inset_axes([0, 0, 1, 0.5])]
    zplanes = [3, 4, 5]

    img = np.mean(img[:, zplanes, :, :], axis=1)

    # for sub_ax, z in zip(inset_axes, zplanes):
    lim = np.array(INSET)
    stack = np.dstack(
        [
            rotate(i, ROTATION)[lim[0] : lim[2], lim[1] : lim[3]]
            for i in [
                img[1, :, :],
                img[1, :, :],
                img[0, :, :],
            ]
        ]
    )
    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rgb = vis.to_rgb(stack, colors, vmax=[1000, 10000, 5000], vmin=None)
    ax.imshow(rgb, vmin=0, vmax=5000)

    # Add scalebar
    bar_length = 20  # Scale bar in micrometers
    scalebar2 = plt.Line2D(
        [rgb.shape[1] - 140, rgb.shape[1] + bar_length / scale["X"] - 140],
        [rgb.shape[0] - 50, rgb.shape[0] - 50],
        color="white",
        linewidth=4,
    )
    # fontprops = fm.FontProperties(size=label_fontsize)
    # scalebar = AnchoredSizeBar(
    #     ax.transData,
    #     bar_length / scale["X"],
    #     loc="lower right",
    #     label_top=True,
    #     color="white",
    #     frameon=False,
    #     size_vertical=5,
    #     prop=fontprops,
    # )
    # ax.add_artist(scalebar)

    # Add starter cell arrows
    # starters = [np.array([390, 200]), np.array([650, 480])]
    starters = [np.array([220, 130]), np.array([460, 430])]
    for starter in starters:
        ax.annotate(
            "",
            xy=starter,
            xytext=starter + np.array([-75, -75]),
            arrowprops=dict(
                facecolor="white",
                shrink=0.05,
                edgecolor="none",
                width=1,
                headwidth=6,
            ),
        )

    # sub_ax.set_xticks([])
    # sub_ax.set_yticks([])

    # Adjust layout
    # ax.figure.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
    ax.axis("off")
    return ax


def plot_tail_vs_local_images(
    ax_local, ax_tail, vmin, vmax, xl=[400, 1600], yl=[100, 1300], scale_size=250
):
    """Plots max projection images of tail vein vs local injection.

    Args:
        ax_local (matplotlib.axes._axes.Axes): Axis for the local injection image.
        ax_tail (matplotlib.axes._axes.Axes): Axis for the tail vein injection image.
        vmin (int, int): Minimum value for the red and cyan channels.
        vmax (int, int): Maximum value for the red and cyan channels.
        xl (list, optional): x limits for the image crop. Defaults to [400,1600].
        yl (list, optional): y limits for the image crop. Defaults to [100, 1300].
        scale_size (int, optional): Size of the scale bar in micrometers. Defaults to
            250.
    """
    taillocal_projections = flz.get_processed_path(
        "becalia_rabies_barseq/tail_vs_local"
    )
    tail_img = tf.imread(taillocal_projections / "MAX_BRAC10946.1f_injection_site.tif")
    local_img = tf.imread(taillocal_projections / "MAX_BRAC10946.1c_injection_site.tif")
    tail_img = np.moveaxis(tail_img, 0, 2)
    local_img = np.moveaxis(local_img, 0, 2)

    # the two injections are not exactly centered. Keep same image dimension but shift one
    shift_tail = [0, 80]

    rgb = vis.to_rgb(
        local_img[yl[0] : yl[1], xl[0] : xl[1]],
        colors=[(1, 0, 0), (0, 1, 1)],
        vmax=vmax,
        vmin=vmin,
    )
    ax_local.imshow(rgb)

    xl = np.array(xl) + shift_tail[0]
    yl = np.array(yl) + shift_tail[1]
    rgb = vis.to_rgb(
        tail_img[yl[0] : yl[1], xl[0] : xl[1]],
        colors=[(1, 0, 0), (0, 1, 1)],
        vmax=vmax,
        vmin=vmin,
    )
    ax_tail.imshow(rgb)
    for ax in [ax_tail, ax_local]:
        ax.set_axis_off()

    # scale bar
    um_per_px = 0.977
    length = scale_size / um_per_px
    scale = plt.Rectangle(
        (np.diff(xl) - length - 70, np.diff(yl) - 70), length, 40, color="w"
    )
    ax_tail.add_artist(scale)


def plot_tailvein_vs_local_cells(
    ax_scatter, ax_dist, fontsize_dict, linewidth=2, kde_bw=1
):
    taillocal_projections = flz.get_processed_path(
        "becalia_rabies_barseq/tail_vs_local"
    )

    def load_xml_data(path2xml):
        tree = ElementTree.parse(path2xml)
        root = tree.getroot()
        marker_data = root[1]
        marker_1 = marker_data[1]
        cells = []
        for marker in marker_1:
            if marker.tag != "Marker":
                continue
            assert marker[0].tag == "MarkerX"
            assert marker[1].tag == "MarkerY"
            assert marker[2].tag == "MarkerZ"
            cells.append([int(marker[i].text) for i in range(3)])
        cells = np.array(cells)
        return cells

    # cell counting on full resolution dataset
    scale = np.array([1, 1, 5]) / 1000.0
    mouse_names = dict(tail="BRAC10946.1f", local="BRAC10946.1c")
    clicked_cells = {}
    for where, mouse in mouse_names.items():
        clicked_cells[where] = (
            load_xml_data(taillocal_projections / f"{mouse}_manual_click_full_res.xml")
            * scale
        )

    color = dict(
        local="forestgreen",
        tail="slateblue",
    )
    max_val = 0
    label = dict(local="Intracortical", tail="Tail vein")
    for where in ["local", "tail"]:
        cells = clicked_cells[where]
        center = np.nanmean(cells, axis=0)
        rel_cells = cells - center
        ax_scatter.scatter(
            rel_cells[:, 0],
            rel_cells[:, 2],
            color=color[where],
            marker="o",
            edgecolor="w",
            alpha=0.7,
            s=10,
            linewidths=0.5,
            label=label[where],
        )
        max_val = max(max_val, np.nanmax(np.abs(rel_cells[:, 2])))
        kde = gaussian_kde(rel_cells[:, 2], bw_method=kde_bw)
        bins = np.arange(-0.4, 0.4, 0.01)
        ax_dist.plot(
            bins,
            kde(bins),
            color=color[where],
            linewidth=linewidth,
        )
    ax_dist.set_xticks(
        [-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"]
    )
    ax_dist.set_xlabel(
        "ML distance to\ninjection site (mm)", fontsize=fontsize_dict["label"]
    )
    ax_dist.set_ylabel("Density (mm$^{-1}$)", fontsize=fontsize_dict["label"])
    ax_dist.set_yticks([0, 2, 4], labels=[0, 2, 4], fontsize=fontsize_dict["tick"])
    ax_scatter.legend(
        title="Injection site",
        fontsize=fontsize_dict["legend"],
        title_fontsize=fontsize_dict["legend"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.1),
        handlelength=1,
        alignment="left",
    )

    ax_scatter.set_xticks(
        [-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"]
    )
    ax_scatter.set_yticks(
        [-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"]
    )
    ax_scatter.set_xlabel(
        "ML distance to\ninjection site (mm)", fontsize=fontsize_dict["label"]
    )
    ax_scatter.set_ylabel(
        "AP distance to\ninjection site (mm)", fontsize=fontsize_dict["label"]
    )
