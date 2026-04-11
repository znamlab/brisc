"""
Plotting functions for gene transcript spatial distributions
on coronal brain section outlines.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scanpy as sc


def _label_boundaries(label_image: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Compute a 1‑pixel boolean boundary mask between labels.

    This prevents stacked contour lines (and darker appearance) by
    plotting only a single outline per interface.

    Parameters
    ----------
    label_image : np.ndarray
        2‑D integer label image.
    connectivity : int
        4 or 8. Use 8 to include diagonal differences.

    Returns
    -------
    np.ndarray
        Boolean mask (True at boundaries).
    """
    labels = np.asarray(label_image)
    boundary = np.zeros(labels.shape, dtype=bool)
    # 4-neighbour differences
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    if connectivity == 8:
        boundary[1:, 1:] |= labels[1:, 1:] != labels[:-1, :-1]
        boundary[1:, :-1] |= labels[1:, :-1] != labels[:-1, 1:]
    return boundary

def _adjacent_label_midlevels(label_image: np.ndarray) -> np.ndarray:
    """Compute unique mid-levels for contours at label adjacencies.

    For an integer label image, plotting contours at many levels can
    duplicate the same physical boundary (when neighbouring labels
    differ by >1). This function finds all unique unordered pairs of
    4-neighbouring labels and returns the (a+b)/2 mid-values to contour
    exactly once per shared boundary.

    Returns
    -------
    np.ndarray
        1-D float array of contour levels. May be empty.
    """
    labels = np.asarray(label_image)
    # vertical neighbours
    a = labels[1:, :].ravel()
    b = labels[:-1, :].ravel()
    m1 = a != b
    pairs = np.stack((a[m1], b[m1]), axis=1) if np.any(m1) else np.empty((0, 2), dtype=labels.dtype)
    # horizontal neighbours
    a = labels[:, 1:].ravel()
    b = labels[:, :-1].ravel()
    m2 = a != b
    pairs2 = np.stack((a[m2], b[m2]), axis=1) if np.any(m2) else np.empty((0, 2), dtype=labels.dtype)
    if pairs.size == 0 and pairs2.size == 0:
        return np.asarray([], dtype=float)
    all_pairs = np.vstack((pairs, pairs2)) if pairs.size and pairs2.size else (pairs if pairs2.size == 0 else pairs2)
    # sort each pair so (a,b) and (b,a) collapse
    all_pairs = np.sort(all_pairs.astype(float), axis=1)
    # unique rows → mid-levels, ensure unique and strictly increasing
    uniq = np.unique(all_pairs, axis=0)
    levels = uniq.mean(axis=1).astype(float)
    levels = np.unique(levels)  # also sorts ascending
    return levels


def plot_gene_transcripts_mosaic(
    fig,
    gs_slot,
    gene_data,
    bin_image,
    genes,
    fontsize_dict,
    atlas_size=10,
    spot_score_threshold=0.2,
    s=0.3,
    alpha=0.4,
    ncols=None,
    colors=None,
    border_color="black",
    border_linewidth=0.35,
    xlim=(1100, 500),
    ylim=(420, 0),
):
    """
    Plot gene transcript coordinates on coronal brain outlines.

    Each subplot shows the spatial distribution of a single gene's
    transcripts, overlaid on brain area contours (identical to those
    used in ``plot_cluster_mosaic``).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to draw into.
    gs_slot : matplotlib.gridspec.SubplotSpec
        A GridSpec slot to subdivide for the mosaic.
    gene_data : pd.DataFrame
        Gene-spot table with columns ``gene``, ``ara_y``, ``ara_z``,
        and ``spot_score``.
    bin_image : np.ndarray
        2-D label image returned by
        ``spatial_plots_rabies.prepare_area_labels``.
    genes : list[str]
        Genes to plot (one subplot each).
    fontsize_dict : dict
        Font-size dictionary with keys ``"title"``, ``"label"``,
        ``"tick"``, ``"legend"``.
    atlas_size : int
        Resolution of the Allen atlas (µm per voxel).
    spot_score_threshold : float
        Minimum ``spot_score`` for a transcript to be plotted.
    s : float
        Marker size for transcript scatter points.
    alpha : float
        Marker opacity.
    ncols : int or None
        Number of columns. Defaults to ``len(genes)``.
    colors : list[str] or None
        One colour per gene.  Defaults to a tab10 cycle.
    border_color : str or tuple
        Colour of the brain-area contour lines.  Default ``"black"``.
    border_linewidth : float
        Line width of the brain-area contour lines.  Default ``0.35``.
    xlim : tuple
        X-axis limits for the coronal view (inverted).
    ylim : tuple
        Y-axis limits for the coronal view (inverted).

    Returns
    -------
    genes : list[str]
        The genes that were plotted.
    axes : list[matplotlib.axes.Axes]
        One axis per gene.
    """
    if ncols is None:
        ncols = len(genes)

    nrows = int(np.ceil(len(genes) / ncols))

    # default colour cycle
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(genes))]

    # filter by score and remove spots outside the brain
    gene_data = gene_data[
        (gene_data["spot_score"] > spot_score_threshold)
        & (gene_data["area_acronym"] != "outside")
    ].copy()

    subgs = gs_slot.subgridspec(
        nrows=nrows,
        ncols=ncols,
        wspace=0.02,
        hspace=0.08,
    )

    axes = []
    for i, gene in enumerate(genes):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(subgs[r, c])
        axes.append(ax)

        # brain outline contour (single boundary mask; avoids overplot darkening)
        _boundary = _label_boundaries(bin_image, connectivity=4)
        ax.contour(
            _boundary.astype(float),
            levels=[0.5],
            colors=border_color,
            linewidths=border_linewidth,
            zorder=0,
        )

        # gene transcript scatter
        spots = gene_data[gene_data["gene"] == gene]
        if len(spots) > 0:
            ax.scatter(
                spots["ara_z"] * 1000 / atlas_size,
                spots["ara_y"] * 1000 / atlas_size,
                s=s,
                alpha=alpha,
                c=colors[i],
                rasterized=True,
                linewidths=0,
                zorder=2,
            )

        ax.set_title(
            gene,
            fontsize=fontsize_dict["title"],
            pad=1.5,
        )

        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")

    # hide any leftover subplots
    for j in range(len(genes), nrows * ncols):
        r = j // ncols
        c = j % ncols
        ax_empty = fig.add_subplot(subgs[r, c])
        ax_empty.axis("off")

    return genes, axes


def plot_gene_expression_mosaic(
    fig,
    gs_slot,
    adata,
    bin_image,
    genes,
    fontsize_dict,
    atlas_size=10,
    use_raw=True,
    percentile=99,
    s=0.5,
    ncols=None,
    cmaps=None,
    uniform_cmap=None,
    chambers=None,
    exclude_areas=("outside",),
    border_color="black",
    border_linewidth=0.35,
    xlim=(1100, 567),
    ylim=(420, 0),
):
    """
    Plot per-cell gene expression on coronal brain outlines.

    Each subplot shows cell positions coloured by expression of a single
    gene.  Opacity is scaled so that zero-expression cells are fully
    transparent and cells at the chosen percentile of expression are
    fully opaque.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to draw into.
    gs_slot : matplotlib.gridspec.SubplotSpec
        A GridSpec slot to subdivide for the mosaic.
    adata : AnnData
        Annotated data with ``obs["ara_y"]`` / ``obs["ara_z"]``
        coordinates and gene counts in ``.X`` (or ``.raw.X``).
    bin_image : np.ndarray
        2-D label image from ``prepare_area_labels``.
    genes : list[str]
        Genes to plot (one subplot each).
    fontsize_dict : dict
        Font-size dictionary.
    atlas_size : int
        Allen atlas resolution (µm per voxel).
    use_raw : bool
        If True use ``adata.raw.X`` (integer counts); otherwise
        ``adata.X``.
    percentile : float
        Expression percentile (among cells with count > 0) that maps
        to full opacity.
    s : float
        Marker size.
    ncols : int or None
        Columns per row.  Defaults to ``len(genes)``.
    cmaps : list[str | Colormap] or None
        One colormap name (or object) per gene.  Defaults to a
        rotating set of sequential colourmaps.
    uniform_cmap : str or None
        If set, use this single colormap for every gene
        (overrides ``cmaps``).  E.g. ``"Greens"``.
    chambers : list[str] or None
        If provided, only include cells from these chambers
        (matched against ``obs["chamber"]``).
    exclude_areas : tuple[str] or list[str]
        Area acronyms to exclude (matched against
        ``obs["area_acronym"]`` or ``obs["cortical_area"]``).
        Defaults to ``("outside",)``.
    border_color : str or tuple
        Colour of the brain-area contour lines.  Default ``"black"``.
    border_linewidth : float
        Line width of the brain-area contour lines.  Default ``0.35``.
    xlim, ylim : tuple
        Axis limits for the coronal view (inverted).

    Returns
    -------
    genes : list[str]
        The genes that were plotted.
    axes : list[matplotlib.axes.Axes]
        One axis per gene.
    """
    # ---- filter cells ----
    mask = np.ones(adata.n_obs, dtype=bool)

    if chambers is not None and "chamber" in adata.obs.columns:
        mask &= adata.obs["chamber"].isin(list(chambers)).values

    if exclude_areas:
        exclude_set = set(exclude_areas)
        for col in ("area_acronym", "cortical_area"):
            if col in adata.obs.columns:
                mask &= ~adata.obs[col].isin(exclude_set).values
                break

    adata = adata[mask].copy()

    if ncols is None:
        ncols = len(genes)

    nrows = int(np.ceil(len(genes) / ncols))

    # uniform_cmap overrides everything
    if uniform_cmap is not None:
        cmaps = [uniform_cmap] * len(genes)
    elif cmaps is None:
        # default colormaps – monochrome sequential, arranged so
        # adjacent panels (horiz & vert in a 4-col grid) differ
        _default_cmaps = [
            "GnBu", "Oranges", "Greys", "Blues", "Reds",
            "pink_r", "Purples", "Greens", "Wistia", "RdPu",
        ]
        cmaps = [
            _default_cmaps[i % len(_default_cmaps)]
            for i in range(len(genes))
        ]

    # cell coordinates (atlas-scaled)
    z_coords = adata.obs["ara_z"].values * 1000 / atlas_size
    y_coords = adata.obs["ara_y"].values * 1000 / atlas_size

    subgs = gs_slot.subgridspec(
        nrows=nrows,
        ncols=ncols,
        wspace=0.02,
        hspace=0.08,
    )

    axes = []
    for i, gene in enumerate(genes):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(subgs[r, c])
        axes.append(ax)

        # brain outline contour (single boundary mask; avoids overplot darkening)
        _boundary = _label_boundaries(bin_image, connectivity=4)
        ax.contour(
            _boundary.astype(float),
            levels=[0.5],
            colors=border_color,
            linewidths=border_linewidth,
            zorder=0,
            alpha=0.5,
        )

        # extract expression vector for this gene
        if use_raw and adata.raw is not None:
            gene_idx = list(adata.raw.var_names).index(gene)
            expr = np.asarray(adata.raw.X[:, gene_idx].todense()).ravel().astype(float)
        else:
            gene_idx = list(adata.var_names).index(gene)
            col = adata.X[:, gene_idx]
            expr = np.asarray(col.todense()).ravel().astype(float) if hasattr(col, "todense") else np.asarray(col).ravel().astype(float)

        # compute alpha: 0 for zero-count, linearly scaled to 1 at
        # the chosen percentile of non-zero values
        nonzero = expr[expr > 0]
        if len(nonzero) > 0:
            vmax = np.percentile(nonzero, percentile)
        else:
            vmax = 1.0
        vmax = max(vmax, 1e-9)

        alpha = np.clip(expr / vmax, 0, 1)

        # build RGBA array from colormap
        cmap_obj = plt.get_cmap(cmaps[i])
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        rgba = cmap_obj(norm(expr))
        rgba[:, 3] = alpha  # overwrite alpha channel

        # plot zero-expression cells first (background), then non-zero on top
        order = np.argsort(expr)
        ax.scatter(
            z_coords[order],
            y_coords[order],
            s=s,
            c=rgba[order],
            rasterized=True,
            linewidths=0,
            zorder=2,
        )

        ax.set_title(
            gene,
            fontsize=fontsize_dict["title"],
            pad=1.5,
            fontstyle="italic",
        )
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")

    # hide leftover subplots
    for j in range(len(genes), nrows * ncols):
        r = j // ncols
        c = j % ncols
        ax_empty = fig.add_subplot(subgs[r, c])
        ax_empty.axis("off")

    return genes, axes
