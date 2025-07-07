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
    titre=4.5e13,
    volume=50,
    label_fontsize=12,
    tick_fontsize=12,
    processed=Path("/nemo/lab/znamenskiyp/home/shared/projects/"),
):
    """
    Plot the densities of the starter cells for the different dilutions of PHP.eb (only for V1)

    Args:
        ax:
        titre: titre in vg/ml
        volume: injected volume in ul
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
    order = ["1/100", "1/330", "1/1000", "1/3300"][::-1]
    if titre is not None:
        dilution = np.array([1 / 100, 1 / 330, 1 / 1000, 1 / 3300])[::-1]
        num_particle = dilution * volume * 1e-3 * titre
        lab = [f"{p:.1e}" for p in num_particle]
        xtlabel = []
        for l in lab:
            parts = l.split("e+")
            xtlabel.append(f"{parts[0]}x$10^{{{int(parts[1])}}}$")
        xlabel = "Number of viral particles injected"
    else:
        xtlabel = order
        xlabel = ("AAV hSyn-Cre dilution",)
    # Plot the data points with dodge
    sns.stripplot(
        data=v1,
        x="dilution",
        y="density",
        order=order,
        dodge=True,
        jitter=True,
        ax=ax,
        color="lightgray",
        edgecolor="black",
        linewidth=0.5,
        size=3,
    )

    # Calculate positions for the boxplots
    positions_v1 = np.arange(len(order))  # - 0.2

    # Plot the boxplot for v1
    sns.boxplot(
        data=v1,
        x="dilution",
        y="density",
        order=order,
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

    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=tick_fontsize,
    )
    ax.set_xticklabels(
        xtlabel,
        fontsize=tick_fontsize,
        rotation=45 if titre is None else 0,
    )
    plt.xlabel(
        xlabel,
        fontsize=label_fontsize,
    )
    plt.ylabel(
        "Cell density (mm$^{-3}$)",
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
    ax.axis("off")
    return ax


def plot_tail_vs_local_images(
    local_img,
    tail_img,
    ax_local,
    ax_tail,
    vmin,
    vmax,
    xl=None,
    yl=None,
    scale_size=250,
):
    """Plots max projection images of tail vein vs local injection.

    Args:
        local_img (np.ndarray): Max projection image of local injection.
        tail_img (np.ndarray): Max projection image of tail vein injection.
        ax_local (matplotlib.axes._axes.Axes): Axis for the local injection image.
        ax_tail (matplotlib.axes._axes.Axes): Axis for the tail vein injection image.
        vmin (int, int): Minimum value for the red and cyan channels.
        vmax (int, int): Maximum value for the red and cyan channels.
        xl (list, optional): x limits for the image crop. Defaults to [400,1600].
        yl (list, optional): y limits for the image crop. Defaults to [100, 1300].
        scale_size (int, optional): Size of the scale bar in micrometers. Defaults to
            250.
    """
    if xl is not None:
        local_img = local_img[:, xl[0] : xl[1]]
        tail_img = tail_img[:, xl[0] : xl[1]]
    if yl is not None:
        local_img = local_img[yl[0] : yl[1], :]
        tail_img = tail_img[yl[0] : yl[1], :]

    rgb = vis.to_rgb(
        local_img,
        colors=[(1, 0, 0), (0, 1, 1)],
        vmax=vmax,
        vmin=vmin,
    )
    ax_local.imshow(rgb)
    rgb = vis.to_rgb(
        tail_img,
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
        (tail_img.shape[1] - length - 70, tail_img.shape[0] - 70), length, 20, color="w"
    )
    ax_tail.add_artist(scale)


def load_tail_vs_local_images():
    taillocal_projections = flz.get_processed_path(
        "becalia_rabies_barseq/tail_vs_local"
    )
    tail_img = tf.imread(taillocal_projections / "MAX_BRAC10946.1f_injection_site.tif")
    local_img = tf.imread(taillocal_projections / "MAX_BRAC10946.1c_injection_site.tif")
    tail_img = np.moveaxis(tail_img, 0, 2)
    local_img = np.moveaxis(local_img, 0, 2)
    return local_img, tail_img


def load_cell_click_data(relative=False, return_px=False):
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
        cells = np.array(cells, dtype=float)
        return cells

    # cell counting on full resolution dataset
    if return_px:
        scale = 1
    else:
        scale = np.array([1, 1, 5]) / 1000.0
    mouse_names = dict(tail="BRAC10946.1f", local="BRAC10946.1c")
    clicked_cells = {}
    for where, mouse in mouse_names.items():
        cells = load_xml_data(
            taillocal_projections / f"{mouse}_manual_click_full_res.xml"
        )
        if relative:
            center = np.nanmedian(cells, axis=0)
            cells -= center

        clicked_cells[where] = cells * scale

    return clicked_cells


def plot_3d_scatters(ax_local, ax_tail, fontsize_dict, colors, **kwargs):
    # NOT USED
    clicked_cells = load_cell_click_data(relative=True)
    axes = dict(
        local=ax_local,
        tail=ax_tail,
    )
    colors = dict(
        local=colors[0],
        tail=colors[1],
    )
    for where, ax in axes.items():
        coords = [clicked_cells[where][:, i] for i in [0, 2, 1]]
        ax.scatter(*coords, color=colors[where], **kwargs)
        ax.set_xlabel("M/L (mm)", fontsize=fontsize_dict["label"], labelpad=-8)
        ax.set_zlabel("D/V (mm)", fontsize=fontsize_dict["label"], labelpad=-8)
        ax.set_ylabel("A/P (mm)", fontsize=fontsize_dict["label"], labelpad=-8)
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.4, 0.4])
        ax.set_zlim([-0.4, 0.4])
        ax.set_xticks(
            [-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"]
        )
        ax.set_yticks(
            [-0.4, 0, 0.4], labels=["", 0, ""], fontsize=fontsize_dict["tick"]
        )
        ax.set_zticks(
            [-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"]
        )
        ax.view_init(elev=20, azim=35, roll=0)
        [t.set_va("center") for t in ax.get_yticklabels()]
        [t.set_ha("right") for t in ax.get_yticklabels()]
        ax.tick_params(pad=-4)


def plot_tailvein_vs_local_cells(
    ax_transverse, ax_distri, fontsize_dict, linewidth=2, kde_bw=1
):
    color = dict(
        local="forestgreen",
        tail="slateblue",
    )

    clicked_cells = load_cell_click_data(relative=True)

    label = dict(local="Intracortical", tail="Tail vein")
    for where in ["local", "tail"]:
        cells = clicked_cells[where]
        ax_transverse.scatter(
            cells[:, 0],
            cells[:, 2],
            color=color[where],
            marker="o",
            edgecolor="w",
            alpha=0.7,
            s=10,
            linewidths=0.5,
            label=label[where],
        )
        kde = gaussian_kde(cells[:, 2], bw_method=kde_bw)
        bins = np.arange(-0.4, 0.4, 0.01)
        ax_distri.plot(
            bins,
            kde(bins),
            color=color[where],
            linewidth=linewidth,
        )
    ax_distri.set_xticks(
        [-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"]
    )
    ax_distri.set_xlabel("ML position (mm)", fontsize=fontsize_dict["label"])
    ax_distri.set_ylabel("Density (mm$^{-1}$)", fontsize=fontsize_dict["label"])
    ax_distri.set_yticks([0, 2, 4], labels=[0, 2, 4], fontsize=fontsize_dict["tick"])
    ax_transverse.legend(
        title="Injection site",
        fontsize=fontsize_dict["legend"],
        title_fontsize=fontsize_dict["legend"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.1),
        handlelength=1,
        alignment="left",
    )

    ax_transverse.set_xticks(
        [-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"]
    )
    ax_transverse.set_yticks(
        [-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"]
    )
    ax_transverse.set_xlabel("ML position (mm)", fontsize=fontsize_dict["label"])
    ax_transverse.set_ylabel("AP position (mm)", fontsize=fontsize_dict["label"])
    for ax in [ax_transverse, ax_distri]:
        despine(ax)


def plot_taillocal_ml_distribution(ax, colors, fontsize_dict, **kwargs):
    """"""
    clicked_cells = load_cell_click_data(relative=True)
    colors = dict(
        local=colors[0],
        tail=colors[1],
    )
    bins = np.arange(-0.5, 0.5, 0.01)
    lines = []
    for where in ["local", "tail"]:
        cells = clicked_cells[where]
        kde = gaussian_kde(cells[:, 0], bw_method=0.3)(bins)
        line = ax.plot(bins, kde / kde.max(), color=colors[where], **kwargs)
        lines.append(line[0])
    ax.set_xticks([-0.5, 0, 0.5], labels=[-0.5, 0, 0.5], fontsize=fontsize_dict["tick"])
    ax.set_yticks([0, 1], labels=[0, 1], fontsize=fontsize_dict["tick"])
    ax.set_xlabel("ML position (mm)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("Normalised cell density", fontsize=fontsize_dict["label"])
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    despine(ax)
    return lines


def plot_taillocal_scatter(ax, colors, fontsize_dict, clicked_cells=None, **kwargs):
    """"""
    if clicked_cells is None:
        clicked_cells = load_cell_click_data(relative=True)
    colors = dict(
        local=colors[0],
        tail=colors[1],
    )
    zorder = dict(tail=5, local=1)
    scatters = []
    for where in ["local", "tail"]:
        cells = clicked_cells[where]
        sc = ax.scatter(
            cells[:, 0],
            cells[:, 2],
            color=colors[where],
            zorder=zorder[where],
            **kwargs,
        )
        scatters.append(sc)

    ax.set_aspect("equal")
    ax.set_xticks([-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"])
    ax.set_yticks([-0.4, 0, 0.4], labels=[-0.4, 0, 0.4], fontsize=fontsize_dict["tick"])
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    despine(ax)
    ax.set_xlabel("ML position (mm)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("AP position (mm)", fontsize=fontsize_dict["label"])
    return scatters


def plot_pairwise_dist_distri(ax, colors, fontsize_dict, clicked_cells=None, **kwargs):
    """Plots the distribution of pairwise distances between cells for two conditions.

    This function visualizes the spatial spread of cells from 'local' and 'tail'
    injections by plotting the kernel density estimate (KDE) of all pairwise
    Euclidean distances within each group. The median distance for each group is
    indicated with a triangular marker.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        colors (tuple or list): A sequence of two colors for the 'local' and
            'tail' distributions, respectively.
        fontsize_dict (dict): A dictionary containing font sizes for plot elements,
            e.g., {'label': 12, 'tick': 10}.
        clicked_cells (dict, optional): A dictionary with keys 'local' and 'tail',
            where each value is a numpy array of cell coordinates (N, 3). If not
            provided, data is loaded using `load_cell_click_data`.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to `ax.plot` for the KDE lines.
    """
    if clicked_cells is None:
        clicked_cells = load_cell_click_data(relative=True)

    pairwise = {}
    bins = np.arange(0, 1, 0.01)
    colors = dict(
        local=colors[0],
        tail=colors[1],
    )
    for where in ["local", "tail"]:
        cells = clicked_cells[where]
        dst = np.linalg.norm(cells[:, None] - cells[None, :], axis=2)
        pairwise[where] = dst[~np.eye(dst.shape[0], dtype=bool)]
        med = np.median(pairwise[where])
        sc = ax.scatter(med, 1.05, color=colors[where], marker="v", s=10)
        sc.set_clip_on(False)
        print(f"{where} median: {med:.2f} mm, n: {len(cells)} cells")
        kde = gaussian_kde(pairwise[where], bw_method=0.1)(bins)

        (line,) = ax.plot(bins, kde / kde.max(), color=colors[where], **kwargs)
        line.set_clip_on(False)

    ax.set_xticks([0, 0.5, 1], labels=[0, 0.5, 1], fontsize=fontsize_dict["tick"])
    ax.set_yticks([0, 1], labels=[0, 1], fontsize=fontsize_dict["tick"])
    ax.set_xlabel("Pairwise distance (mm)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("Normalised cell density", fontsize=fontsize_dict["label"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    despine(ax)
