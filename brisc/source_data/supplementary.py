"""Source Data exporters for the supplementary figures.

As in :mod:`brisc.source_data.figures`, each exporter takes the variables its notebook
plots and writes one worksheet per data-bearing panel. Micrographs, atlas outlines and
the exploratory cells of those notebooks have no worksheet.
"""

from functools import partial

import numpy as np
import pandas as pd

from brisc.source_data.figures import (
    CELL_SUBSAMPLE_NOTE,
    NOTE_ATTR,
    _abundance_sheet,
    _longrange_flatmap_sheet,
    _longrange_running_average_sheet,
    _longrange_scatter_sheet,
    _longrange_smoothed_map_sheet,
    _note,
    _presynaptic_density,
    _subsample_cells,
    _unique_fraction_sheet,
)
from brisc.source_data.io import save_excel_sheets

# ---------------------------------------------------------------------------
# Supplementary Figure 1 / 2 — presynaptic density and starter dilution
# ---------------------------------------------------------------------------

SUPP1_PANELS = [
    "Supp 1a Library abundance",
    "Supp 1b Unique fraction",
    "Supp 2c Presynaptic density",
    "Supp 2d Starter dilution",
]

#: Keys of ``suppfig2_plotted_data`` belonging to an image panel: the coronal rabies
#: slice is a micrograph and the confocal panel returns only the arrows drawn over one.
SUPP1_IMAGE_KEYS = ("coronal_rabies_slice", "starter_confocal")


def build_suppfig2_source_data(suppfig2_plotted_data=None):
    """Build the panel dictionary for Supplementary Figure 1/2 from what it drew.

    Args:
        suppfig2_plotted_data (dict): The notebook's ``suppfig2_plotted_data``, with
            ``"presynaptic_density"`` from `rabies_cell_counting.plot_rabies_density`
            and ``"starter_dilution"`` from
            `starter_cell_counting.plot_starter_dilution_densities`.

    Returns:
        dict: Sheet name to DataFrame, with worksheet notes in ``DataFrame.attrs``.
    """
    plotted = suppfig2_plotted_data or {}
    builders = (
        ("library_abundance", "Supp 1a Library abundance", _abundance_sheet),
        ("library_unique_fraction", "Supp 1b Unique fraction", _unique_fraction_sheet),
        (
            "presynaptic_density",
            "Supp 2c Presynaptic density",
            _supp1_presynaptic_density_sheet,
        ),
        ("starter_dilution", "Supp 2d Starter dilution", _supp1_starter_dilution_sheet),
    )
    panels = {
        sheet: build(plotted[key]) for key, sheet, build in builders if plotted.get(key)
    }

    known = {key for key, _, _ in builders} | set(SUPP1_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Supp 1: no sheet for plotted panels {unknown}")

    return panels


def _supp1_presynaptic_density_sheet(drawn):
    """Panel c — cumulative isocortex cell density around the injection site."""
    curve = drawn["cumulative_density"] if "cumulative_density" in drawn else drawn
    return pd.DataFrame(
        {
            "Distance_To_Injection_mm": np.asarray(curve["x"], dtype=float),
            "Cell_Density_Per_mm3": np.asarray(curve["y"], dtype=float),
        }
    )


def _supp1_starter_dilution_sheet(drawn):
    """Panel d — the per-mouse starter densities and the mean drawn over each dilution.

    The panel is a strip plot of one point per mouse with a mean line per dilution; the
    mouse identifiers and the cell counts the density is computed from are not drawn, so
    they are not written here.
    """
    frames = [
        pd.DataFrame(
            {
                "Series_Type": label,
                "Dilution": np.asarray(drawn[key]["x"]),
                "Cell_Density_Per_mm3": np.asarray(drawn[key]["y"], dtype=float),
            }
        )
        for key, label in (("individual", "Individual mouse"), ("mean", "Mean"))
        if drawn.get(key) is not None
    ]
    table = pd.concat(frames, ignore_index=True)
    order = drawn.get("dilution_order")
    note = (
        "Note: one row per mouse for the individual points, plus one row per dilution "
        "for the mean line the panel draws over them. The density is on a logarithmic "
        "axis."
    )
    if order:
        note += f" Dilutions are drawn in the order {', '.join(map(str, order))}."
    return _note(table, note)


def export_suppfig2_source_data(output_path, **kwargs):
    """Supplementary Figure 1 / 2."""
    panels = build_suppfig2_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=SUPP1_PANELS
    )


# ---------------------------------------------------------------------------
# Supplementary Figure 4 — barcode length
# ---------------------------------------------------------------------------

SUPP4_PANELS = [
    "Supp 4a Library abundance",
    "Supp 4b Unique fraction",
    "Supp 4c Unique vs length",
]


def build_suppfig4_source_data(suppfig4_plotted_data=None):
    """Build the panel dictionary for Supplementary Figure 4 from what it drew.

    The notebook collects the return value of every plotting call in
    ``suppfig4_plotted_data``, so each sheet holds the arrays that were handed to
    matplotlib rather than a second, re-derived copy of them: panel b used to be
    recomputed by :func:`figures._unique_fraction` on its own evaluation grid instead of
    the notebook's. Nothing that is not drawn enters the workbook, so the printed
    95%/99% unique-labelling estimates that
    `viral_library.plot_unique_label_fraction` also returns are dropped, as are the
    barcode sequences the curves are counted from.

    Every curve of this figure is our own RV35 library truncated to a given barcode
    length, so, unlike Figure 1, there is no library published by another laboratory to
    leave out with :func:`figures._own_libraries_only`.

    Args:
        suppfig4_plotted_data (dict): The notebook's ``suppfig4_plotted_data``.
            ``"library_abundance"`` comes from
            `viral_library.plot_barcode_counts_and_percentage` and
            ``"unique_fraction"`` from `viral_library.plot_unique_label_fraction`, each
            with one ``x``/``y`` entry per library; ``"unique_vs_length"`` is assembled
            in the figure cell, which draws panel c inline.

    Returns:
        dict: Sheet name to DataFrame. Sheets needing a worksheet note carry it in
        ``DataFrame.attrs``.
    """
    plotted = suppfig4_plotted_data or {}
    # Plotted key, sheet, and what turns one into the other; in panel order. Panels a
    # and b reuse Figure 1's curve builders, so every library curve of the manuscript is
    # tabulated the same way.
    builders = (
        ("library_abundance", "Supp 4a Library abundance", _abundance_sheet),
        ("unique_fraction", "Supp 4b Unique fraction", _unique_fraction_sheet),
        ("unique_vs_length", "Supp 4c Unique vs length", _unique_vs_length_sheet),
    )
    panels = {}
    for key, sheet, build in builders:
        drawn = plotted.get(key)
        if not drawn:
            continue
        panels[sheet] = build(drawn)

    known = {key for key, _, _ in builders}
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Supp 4: no sheet for plotted panels {unknown}")

    return panels


def _unique_vs_length_sheet(drawn):
    """Panel c — infections for 95% unique labelling against barcode length, as drawn.

    The panel also fills the markers of the 10 and 20 nucleotide libraries that panels a
    and b draw in full; those are points of this same curve, so the sheet holds it once.
    """
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "Barcode_Length_Nucleotides": np.asarray(curve["x"]).astype(int),
                    "Infections_For_95pc_Unique": np.asarray(curve["y"]).astype(int),
                }
            )
            for curve in drawn.values()
        ],
        ignore_index=True,
    )


def export_suppfig4_source_data(output_path, suppfig4_plotted_data=None):
    """Supplementary Figure 4.

    Args:
        output_path (str or Path): Workbook to write.
        suppfig4_plotted_data (dict): The notebook's ``suppfig4_plotted_data``.
    """
    panels = build_suppfig4_source_data(suppfig4_plotted_data)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=SUPP4_PANELS
    )


# ---------------------------------------------------------------------------
# Supplementary Figure 5 — mCherry cell positions
# ---------------------------------------------------------------------------

SUPP5_PANELS = [
    "Supp 5a Cells per section",
    "Supp 5b Coronal positions",
    "Supp 5c Dorsal positions",
    "Supp 5c Marker gene expression",
]

#: Keys of ``suppfig5_plotted_data`` with no worksheet: the atlas outlines the two
#: position panels are drawn over are reference geometry, not measurements.
SUPP5_IMAGE_KEYS = ("atlas_contours",)

SUPP5_CORONAL_NOTE = (
    "Note: coordinates are Allen CCF positions in 10 um voxels, as plotted, with the y "
    "axis inverted in the panel. Only the cells within the drawn section window are "
    "here; the window is +/-400 um for the cells outside the sequenced volume and "
    "+/-100 um for those inside it, as the figure draws them."
)

SUPP5_DORSAL_NOTE = (
    "Note: coordinates are Allen CCF positions in 10 um voxels, as plotted, with the x "
    "axis inverted in the panel. This panel draws every cell of both groups."
)


def build_suppfig5_source_data(suppfig5_plotted_data=None):
    """Build the panel dictionary for Supplementary Figure 5 from what it drew.

    Args:
        suppfig5_plotted_data (dict): The notebook's ``suppfig5_plotted_data``, with
            ``"cells_per_section"`` (one entry per line of panel a),
            ``"coronal_positions"`` and ``"dorsal_positions"`` (the two cell groups each
            scatter draws) and ``"dotplot"`` (scanpy's own ``dot_color_df`` and
            ``dot_size_df``, the values the dot plot maps to colour and size).

    Returns:
        dict: Sheet name to DataFrame, with worksheet notes in ``DataFrame.attrs``.
    """
    plotted = suppfig5_plotted_data or {}
    builders = (
        ("cells_per_section", "Supp 5a Cells per section", _supp5_sections_sheet),
        (
            "coronal_positions",
            "Supp 5b Coronal positions",
            partial(
                _supp5_positions_sheet,
                xcol="ARA_Z_px",
                ycol="ARA_Y_px",
                note=SUPP5_CORONAL_NOTE,
            ),
        ),
        (
            "dorsal_positions",
            "Supp 5c Dorsal positions",
            partial(
                _supp5_positions_sheet,
                xcol="ARA_X_px",
                ycol="ARA_Z_px",
                note=SUPP5_DORSAL_NOTE,
            ),
        ),
        ("dotplot", "Supp 5c Marker gene expression", _supp5_dotplot_sheet),
    )
    panels = {
        sheet: build(plotted[key]) for key, sheet, build in builders if plotted.get(key)
    }

    known = {key for key, _, _ in builders} | set(SUPP5_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Supp 5: no sheet for plotted panels {unknown}")

    return panels


def _supp5_sections_sheet(drawn):
    """Panel a — one row per drawn point of every mCherry and starter count line.

    The series name carries the chamber, which is what the panel's running section
    offset encodes; the sheet keeps the section position as drawn.
    """
    frames = []
    for series, line in drawn.items():
        cell_type, _, chamber = str(series).partition(", ")
        frames.append(
            pd.DataFrame(
                {
                    "Cell_Type": cell_type,
                    "Chamber": chamber,
                    "Section_Position_um": np.asarray(line["x"], dtype=float),
                    "Cell_Count": np.asarray(line["y"], dtype=float),
                }
            )
        )
    return _note(
        pd.concat(frames, ignore_index=True),
        "Note: mCherry counts are on the left axis and starter counts on the right one. "
        "Section_Position_um runs continuously across chambers, as the panel draws it.",
    )


def _supp5_positions_sheet(drawn, xcol, ycol, note):
    """Panels b and c — the cells each scatter drew, in the coordinates it drew them."""
    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "Cell_Group": group,
                    xcol: np.asarray(scatter["x"], dtype=float),
                    ycol: np.asarray(scatter["y"], dtype=float),
                }
            )
            for group, scatter in drawn.items()
        ],
        ignore_index=True,
    )
    table, subsampled = _subsample_cells(table)
    return _note(table, f"{note} {CELL_SUBSAMPLE_NOTE}" if subsampled else note)


def _supp5_dotplot_sheet(drawn):
    """Panel d — the dot plot's own colour and size values, as scanpy computed them."""
    colors = pd.DataFrame(drawn["dot_color_df"])
    sizes = pd.DataFrame(drawn["dot_size_df"])
    long_colors = (
        colors.rename_axis("Cluster")
        .reset_index()
        .melt(id_vars="Cluster", var_name="Gene", value_name="Scaled_Mean_Expression")
    )
    long_sizes = (
        sizes.rename_axis("Cluster")
        .reset_index()
        .melt(id_vars="Cluster", var_name="Gene", value_name="Fraction_Expressing")
    )
    table = long_colors.merge(long_sizes, on=["Cluster", "Gene"], how="left")
    return _note(
        table,
        "Note: Scaled_Mean_Expression is the dot colour, the mean expression rescaled "
        "to 0-1 within each gene (scanpy's standard_scale='var'), not the raw mean. "
        "Fraction_Expressing is the dot size.",
    )


def export_suppfig5_source_data(output_path, **kwargs):
    panels = build_suppfig5_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=SUPP5_PANELS
    )


# ---------------------------------------------------------------------------
# Supplementary Figure 6 — transcriptomic validation mosaic
# ---------------------------------------------------------------------------

SUPP6_PANELS = [
    "Supp 6 Cluster positions",
    "Supp 6 Cluster depth KDE",
]


#: Sheet note of the depth panel, whose curve is the only thing the panel draws.
SUPP6_KDE_NOTE = (
    "Note: the panel draws one Gaussian kernel density estimate of cortical depth per "
    "cluster (scipy.stats.gaussian_kde, bw_method 0.1), evaluated on 200 depths "
    "spanning the range observed in that cluster and scaled to its own maximum. The "
    "curve is what the panel shows and is what is tabulated here; the per-cell depths "
    "behind it are not drawn, and are deposited on Figshare instead (see "
    "figshare_dataset.md). Only cells of the flatmap window of the panel contribute."
)


def build_suppfig6_source_data(suppfig6_plotted_data=None):
    """Build the panel dictionary for Supplementary Figure 6 from what it drew."""
    plotted = dict(suppfig6_plotted_data or {})
    if "cluster_mosaic" in plotted:
        cm = plotted.pop("cluster_mosaic")
        if isinstance(cm, dict):
            for k, v in cm.items():
                plotted[k] = v

    if "depth_kde" in plotted and "cluster_depth_kde" not in plotted:
        plotted["cluster_depth_kde"] = plotted.pop("depth_kde")

    builders = (
        ("cluster_positions", "Supp 6 Cluster positions", _supp6_positions_sheet),
        ("cluster_depth_kde", "Supp 6 Cluster depth KDE", _supp6_depth_kde_sheet),
    )
    panels = {}
    for key, sheet, build in builders:
        drawn = plotted.get(key)
        if drawn is not None:
            df = build(drawn)
            if df is not None:
                panels[sheet] = df

    return panels


def _supp6_positions_sheet(drawn):
    """The coronal scatter of every cluster panel, one row per plotted cell."""
    if isinstance(drawn, pd.DataFrame):
        return drawn
    frames = [
        pd.DataFrame(
            {
                "Cluster": cluster,
                "ARA_Z_px": np.asarray(cluster_drawn["x"], dtype=float),
                "ARA_Y_px": np.asarray(cluster_drawn["y"], dtype=float),
            }
        )
        for cluster, cluster_drawn in drawn.items()
        if len(cluster_drawn["x"])
    ]
    if not frames:
        return pd.DataFrame(columns=["Cluster", "ARA_Z_px", "ARA_Y_px"])
    table = pd.concat(frames, ignore_index=True)
    table, subsampled = _subsample_cells(table)
    note = (
        "Note: coordinates are Allen CCF positions in 10 um voxels, as plotted. Each "
        "cluster is one panel of the mosaic."
    )
    if subsampled:
        note = f"{note} {CELL_SUBSAMPLE_NOTE}"
    return _note(table, note)


def _supp6_depth_kde_sheet(drawn):
    """The depth density curve drawn beside each cluster panel, as evaluated.

    `plot_cluster_mosaic` draws the curve with the density on the x axis and the depth
    on the y axis, which is the order the arrays come back in.
    """
    if isinstance(drawn, pd.DataFrame):
        return drawn
    frames = [
        pd.DataFrame(
            {
                "Cluster": cluster,
                "Cortical_Depth_um": np.asarray(cluster_drawn["y"], dtype=float),
                "Normalised_Density": np.asarray(cluster_drawn["x"], dtype=float),
            }
        )
        for cluster, cluster_drawn in drawn.items()
    ]
    table = pd.concat(frames, ignore_index=True)
    return _note(table, SUPP6_KDE_NOTE)


def export_suppfig6_source_data(output_path, **kwargs):
    panels = build_suppfig6_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=SUPP6_PANELS
    )


# ---------------------------------------------------------------------------
# Supplementary Figure 8 — barcodes in multiple starter cells
# ---------------------------------------------------------------------------

SUPP8_PANELS = [
    "Supp 8a Library abundance KDE",
    "Supp 8b Density difference",
    "Supp 8c Pairwise distances",
]

#: Keys of ``suppfig8_plotted_data`` that carry no worksheet. The bootstrap medians and
#: their confidence interval reach the panel only as marker positions and significance
#: brackets -- the interval's own plotting call is commented out in the notebook -- so
#: they are printed statistics rather than drawn data.
SUPP8_IMAGE_KEYS = ()


def build_suppfig8_source_data(suppfig8_plotted_data=None):
    """Build the panel dictionary for Supplementary Figure 8 from what it drew.

    Args:
        suppfig8_plotted_data (dict): The notebook's ``suppfig8_plotted_data``, with one
            entry per panel: ``"library_abundance_kde"`` and ``"density_difference"``
            (one curve per series, the multi-starter one carrying the drawn confidence
            band), and ``"pairwise_distances"`` (one kernel density estimate per
            comparison, with the median marker drawn above it).

    Returns:
        dict: Sheet name to DataFrame, with worksheet notes in ``DataFrame.attrs``.
    """
    plotted = suppfig8_plotted_data or {}
    builders = (
        ("library_abundance_kde", "Supp 8a Library abundance KDE", _supp8_kde_sheet),
        ("density_difference", "Supp 8b Density difference", _supp8_difference_sheet),
        ("pairwise_distances", "Supp 8c Pairwise distances", _supp8_pairwise_sheet),
    )
    panels = {
        sheet: build(plotted[key]) for key, sheet, build in builders if plotted.get(key)
    }

    known = {key for key, _, _ in builders} | set(SUPP8_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Supp 8: no sheet for plotted panels {unknown}")

    return panels


def _supp8_curve_sheet(drawn, xcol, ycol, note):
    """One row per point of every curve of a panel, plus its drawn confidence band.

    The band is stored against the series it belongs to, so a reader can tell which
    curve it shades; series without a band leave those cells empty.
    """
    frames = []
    for series, curve in drawn.items():
        if not isinstance(curve, dict) or "x" not in curve:
            continue  # scalars such as total_read_in_library, handled by the note
        table = pd.DataFrame(
            {
                "Series": series,
                xcol: np.asarray(curve["x"], dtype=float),
                ycol: np.asarray(curve["y"], dtype=float),
            }
        )
        if curve.get("ci_lower") is not None:
            table["CI_Lower"] = np.asarray(curve["ci_lower"], dtype=float)
            table["CI_Upper"] = np.asarray(curve["ci_upper"], dtype=float)
        frames.append(table)
    return _note(pd.concat(frames, ignore_index=True), note)


def _supp8_kde_sheet(drawn):
    """Panel a — the three read-abundance densities, as drawn."""
    note = (
        "Note: the x axis is the log10 read count of a barcode in the viral library; "
        "the panel labels it as a proportion of the library's unique reads, dividing "
        "by a total of {total:,.0f} reads. CI_Lower/CI_Upper are the bootstrap band "
        "drawn around the multiple-starter curve only."
    )
    total = drawn.get("total_read_in_library")
    note = note.format(total=total) if total else note.split(". CI_")[0] + "."
    return _supp8_curve_sheet(drawn, "Log10_Library_Reads", "Density", note)


def _supp8_difference_sheet(drawn):
    """Panel b — each density minus the library density, as drawn."""
    return _supp8_curve_sheet(
        drawn,
        "Log10_Library_Reads",
        "Density_Difference",
        "Note: CI_Lower/CI_Upper are the bootstrap band drawn around the "
        "multiple-starter curve only. The dashed line at zero is a reference, not data.",
    )


def _supp8_pairwise_sheet(drawn):
    """Panel c — the distance densities, each with the median marker drawn above it."""
    frames = []
    for comparison, curve in drawn.items():
        table = pd.DataFrame(
            {
                "Comparison": comparison,
                "Distance_Between_Starters_mm": np.asarray(curve["x"], dtype=float),
                "Density": np.asarray(curve["y"], dtype=float),
            }
        )
        median = curve.get("median")
        if median is not None:
            table["Median_Distance_mm"] = median["x"]
        frames.append(table)
    return _note(
        pd.concat(frames, ignore_index=True),
        "Note: Median_Distance_mm is constant within a comparison; it is the median of "
        "that distribution, drawn as a marker above the curves. The significance "
        "brackets of the panel are drawn from tests whose values the figure does not "
        "show, so they are not tabulated.",
    )


def export_suppfig8_source_data(output_path, **kwargs):
    panels = build_suppfig8_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=SUPP8_PANELS
    )


# ---------------------------------------------------------------------------
# Supplementary Figure 9 — double labelling estimation
# ---------------------------------------------------------------------------

SUPP9_PANELS = [
    "Supp 9a Injection site cells",
    "Supp 9b Observed vs expected",
]


def build_suppfig9_source_data(suppfig9_plotted_data=None):
    """Build the panel dictionary for Supplementary Figure 9 from what it drew.

    Args:
        suppfig9_plotted_data (dict): Notebook's ``suppfig9_plotted_data`` dict
            or results.

    Returns:
        dict: Sheet name to DataFrame.
    """
    plotted = suppfig9_plotted_data or {}
    panels = {}
    if "results" in plotted:
        results = plotted["results"]
    else:
        results = plotted

    adata_region = results.get("adata_region")
    if adata_region is not None:
        obs = adata_region.obs
        columns = {"ARA_X": "ara_x", "ARA_Y": "ara_y", "ARA_Z": "ara_z"}
        table = pd.DataFrame(
            {name: obs[col].values for name, col in columns.items() if col in obs}
        )
        for extra in ("n_barcodes", "is_barcoded", "inside_region"):
            if extra in obs:
                table[extra] = obs[extra].values
        panels["Supp 9a Injection site cells"] = table
    elif "injection_site" in results:
        panels["Supp 9a Injection site cells"] = pd.DataFrame(results["injection_site"])

    observed = results.get("observed")
    expected = results.get("expected")
    if observed is not None and expected is not None:
        table = pd.DataFrame(
            {
                "Barcodes_Per_Cell": np.asarray(pd.Series(observed).index),
                "Observed_Cells": np.asarray(pd.Series(observed).values),
                "Expected_Cells_Poisson": np.asarray(pd.Series(expected).values),
            }
        )
        table["Residual"] = table["Observed_Cells"] - table["Expected_Cells_Poisson"]
        if results.get("lambda_hat") is not None:
            table["Lambda_Hat"] = results["lambda_hat"]
        panels["Supp 9b Observed vs expected"] = table
    elif "observed_vs_expected" in results:
        panels["Supp 9b Observed vs expected"] = pd.DataFrame(
            results["observed_vs_expected"]
        )

    return panels


def export_suppfig9_source_data(output_path, suppfig9_plotted_data=None, results=None):
    """Supplementary Figure 9."""
    if suppfig9_plotted_data is not None:
        data = suppfig9_plotted_data
    elif results is not None:
        data = results
    else:
        data = {}
    panels = build_suppfig9_source_data(data)
    return save_excel_sheets(panels, output_path, expected=SUPP9_PANELS)


# ---------------------------------------------------------------------------
# Supplementary reviewer figure — elevation / antero-posterior
# ---------------------------------------------------------------------------

SUPP_REVIEWER_PANELS = [
    "Rev b Starter positions",
    "Rev c Presynaptic positions",
    "Rev d Smoothed starter map",
    "Rev e Starter vs presyn ML",
    "Rev e Running average and CI",
    "Rev f Elevation running avg",
]

#: Keys of ``reviewer_plotted_data`` that belong to an image panel. The flatmap outlines
#: of panels b, c and d, the elevation retinotopy inset of panel d and the dashed
#: contour of the data-coverage hull drawn on that inset are atlas images and drawing
#: geometry,
#: not measurements, so they have no worksheet. Listed so that a key added later is
#: noticed rather than silently dropped.
SUPP_REVIEWER_IMAGE_KEYS = ("retinotopy_map",)

#: Panel b draws the presynaptic cells of panel c again, in grey behind the starters.
REV_B_NOTE = (
    "Note: the starter cells are drawn coloured by their antero-posterior position "
    "(Starter_AP_Position_mm, turbo colour map, 8.5 to 8.9 mm). The grey cloud drawn "
    "behind them is the presynaptic population of panel c; it is not repeated here, "
    "see the 'Rev c Presynaptic positions' worksheet."
)

#: The smoothed map is an image whose opacity carries the local data support.
REV_D_NOTE = (
    "Note: Gaussian-smoothed mean starter antero-posterior position (mm) of the "
    "presynaptic cells at each flatmap location, drawn as the panel image. Column "
    "headers are flatmap X, the first column is flatmap Y; rows run from the bottom of "
    "the panel upwards. Blank cells lie outside the area covered by presynaptic cells "
    "and are not drawn. Inside it, the panel additionally fades pixels supported by "
    "few cells (opacity proportional to the summed Gaussian weight, saturating at 50), "
    "which is a display property of the image rather than a measurement."
)

#: What the shaded band and the dashed line of panel e are.
REV_E_NOTE = (
    "Note: Shuffle_Lower and Shuffle_Upper are the 2.5th and 97.5th percentiles of "
    "1,000 bootstrap resamplings of the starter cells, drawn as the shaded band; the "
    "individual resamplings are not drawn and are not given. "
    "Mean_Starter_AP_Position_mm is constant: it is the mean starter position over all "
    "presynaptic cells, drawn as the dashed horizontal line."
)

#: Panel f draws the running average only, not the per-cell elevations behind it.
REV_F_NOTE = (
    "Note: the panel draws the Gaussian-weighted running average of the receptive "
    "field elevation of the presynaptic cells against their medio-lateral position. "
    "The per-cell elevation values that go into the average are not drawn in this "
    "panel and so are not given here; the elevation map they are read from is the "
    "Allen atlas retinotopy shown as the inset of panel d."
)


def build_suppfig_reviewer_source_data(reviewer_plotted_data=None):
    """Build the reviewer figure's panels from what the figure actually drew.

    The figure repeats Figure 6 along the antero-posterior axis and elevation: the
    starter value is the antero-posterior atlas coordinate (``ara_x``, in mm) rather
    than the relative medio-lateral flatmap position, and the running average of panel f
    is over receptive-field elevation. As for Figure 6, the notebook collects every
    drawn series in ``reviewer_plotted_data``, so each sheet holds the arrays handed
    to matplotlib rather than a second, re-derived copy of them. Nothing that is not
    drawn enters the workbook: the cell identifiers, the individual bootstrap
    resamplings behind the shuffle band of panel e, the per-cell receptive-field
    elevations that only enter panel f through their running average and the atlas
    images are all dropped.

    Args:
        reviewer_plotted_data (dict): The notebook's ``reviewer_plotted_data``, with
            keys ``"starter_positions"`` (panel b), ``"presynaptic_positions"``
            (panel c), ``"smoothed_starter_map"`` (panel d),
            ``"starter_vs_presyn_ap"`` (panel e, both its scatter and its running
            average) and ``"elevation_running_average"`` (panel f). Image-only keys are
            listed in :data:`SUPP_REVIEWER_IMAGE_KEYS`.

    Returns:
        dict: Sheet name to DataFrame. Sheets needing a worksheet note carry it in
        ``DataFrame.attrs``.
    """
    plotted = reviewer_plotted_data or {}
    # Plotted key, sheet, and what turns one into the other; in panel order. Panel e
    # gives two sheets, its scatter and its running average, from the same key.
    builders = (
        (
            "starter_positions",
            "Rev b Starter positions",
            partial(
                _longrange_flatmap_sheet,
                value="Starter_AP_Position_mm",
                note=REV_B_NOTE,
            ),
        ),
        (
            "presynaptic_positions",
            "Rev c Presynaptic positions",
            partial(_longrange_flatmap_sheet, value="Starter_AP_Position_mm"),
        ),
        (
            "smoothed_starter_map",
            "Rev d Smoothed starter map",
            partial(_longrange_smoothed_map_sheet, note=REV_D_NOTE),
        ),
        (
            "starter_vs_presyn_ap",
            "Rev e Starter vs presyn ML",
            partial(_longrange_scatter_sheet, value="Starter_AP_Position_mm"),
        ),
        (
            "starter_vs_presyn_ap",
            "Rev e Running average and CI",
            partial(
                _longrange_running_average_sheet,
                value="Running_Average_Starter_AP_mm",
                mean_column="Mean_Starter_AP_Position_mm",
                note=REV_E_NOTE,
            ),
        ),
        (
            "elevation_running_average",
            "Rev f Elevation running avg",
            partial(
                _longrange_running_average_sheet,
                value="Running_Average_Elevation_deg",
                note=REV_F_NOTE,
            ),
        ),
    )
    panels = {
        sheet: build(plotted[key]) for key, sheet, build in builders if plotted.get(key)
    }

    known = {key for key, _, _ in builders} | set(SUPP_REVIEWER_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(
            "[Source Data] !! Reviewer figure: no sheet for plotted "
            f"panels {unknown}"
        )

    return panels


def export_suppfig_reviewer_source_data(output_path, **kwargs):
    panels = build_suppfig_reviewer_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=SUPP_REVIEWER_PANELS
    )
