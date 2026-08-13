"""Source Data builders and exporters for the main manuscript figures.

Each ``build_figN_source_data`` takes the variables the figure notebook actually plots
and returns one DataFrame per data-bearing panel, holding the numbers as drawn (same
evaluation grid, same transform). Pure-image panels — micrographs, atlas outlines,
schematics — have no worksheet.

The ``FIGn_PANELS`` lists are passed to :func:`~brisc.source_data.io.save_excel_sheets`
so that a panel that fails to build is reported instead of silently skipped.
"""

from functools import partial

import numpy as np
import pandas as pd

from brisc.source_data.io import save_excel_sheets

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


#: Libraries above this many barcodes are subsampled for the Source Data workbook.
ABUNDANCE_SUBSAMPLE_THRESHOLD = 500000

#: Leading ranks kept in full when subsampling a rank-abundance curve.
ABUNDANCE_HEAD_RANKS = 1000

#: Log-spaced samples taken beyond the head.
ABUNDANCE_LOG_SAMPLES = 100000

#: Sheet note for every worksheet holding a subsampled rank-abundance curve.
ABUNDANCE_SUBSAMPLE_NOTE = (
    "Note: curves for libraries with more than 500,000 barcodes are subsampled due to "
    "Microsoft Excel's limit of 1,048,576 rows per worksheet. The first 1,000 barcode "
    "ranks are given in full; beyond that the ranks are sampled on a log scale "
    "(100,000 points), matching the logarithmic axes of the plotted panel. The "
    "complete, un-subsampled dataset is deposited on Figshare (see "
    "figshare_dataset.md)."
)


#: Key under which a note is carried on a panel DataFrame, read back by the exporters.
NOTE_ATTR = "source_data_note"


def _note(table, note):
    """Attach a worksheet note to a panel, read back by ``export_figN_source_data``."""
    table.attrs[NOTE_ATTR] = note
    return table


def _log_subsample_indices(
    n,
    head=ABUNDANCE_HEAD_RANKS,
    num=ABUNDANCE_LOG_SAMPLES,
):
    """Indices keeping the first `head` ranks in full, then log-spaced up to `n - 1`.

    The abundance curves are drawn on log-log axes, so a linear subsample would put
    almost every kept point in the flat tail and smooth away the steep head. Sampling
    geometrically keeps the head exactly and thins the tail, which is where the curve
    carries no shape. The first and last rank are always kept.

    Args:
        n (int): Number of barcodes in the library.
        head (int): Number of leading ranks kept without thinning.
        num (int): Number of log-spaced samples requested beyond the head. Duplicates
            after rounding to integers are dropped, so fewer may be returned.

    Returns:
        np.ndarray: Sorted, unique indices into the rank-abundance array.
    """
    if n <= head + num:
        return np.arange(n)
    tail = np.geomspace(head, n - 1, num=num).astype(int)
    return np.unique(np.concatenate([np.arange(head), tail]))


def _library_abundance(libraries, label_col="Library"):
    """Barcode index vs abundance, as drawn by
    `viral_library.plot_barcode_counts_and_percentage`.

    That function plots column 0 (barcode index) against column 1 (UMI count) of each
    library array. Libraries above 500k barcodes are subsampled to stay inside Excel's
    1,048,576-row limit, keeping the first 1,000 ranks in full and thinning the rest on
    a log scale to match the plotted axes (see :func:`_log_subsample_indices`).
    """
    frames = []
    for name, array in libraries.items():
        array = np.asarray(array)
        index, counts = array[:, 0], array[:, 1]
        if len(index) > ABUNDANCE_SUBSAMPLE_THRESHOLD:
            keep = _log_subsample_indices(len(index))
            index, counts = index[keep], counts[keep]
        frames.append(
            pd.DataFrame(
                {
                    label_col: name,
                    "Barcode_Index": index.astype(int),
                    "Barcode_Abundance": counts,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else None


def _unique_fraction(libraries, max_cells, label_col="Library"):
    """Fraction of uniquely labelled cells, as drawn by `plot_unique_label_fraction`.

    The evaluation grid is that function's
    ``np.logspace(0, log10(max_cells), dtype=int)``
    with its default of 50 points.
    """
    from brisc.manuscript_analysis import viral_library as virlib

    evaluation_points = np.logspace(0, np.log10(max_cells), dtype=int)
    frames = []
    for name, array in libraries.items():
        probability = virlib.probability_distribution(np.asarray(array))
        fractions = [virlib.fraction_unique(probability, n) for n in evaluation_points]
        frames.append(
            pd.DataFrame(
                {
                    label_col: name,
                    "Number_of_Labeled_Cells": evaluation_points,
                    "Fraction_Uniquely_Labeled": fractions,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else None


def _presynaptic_density(voxel_distances_sorted, cell_distances_sorted):
    """Cumulative isocortex cell density, as drawn by `plot_rabies_density`.

    Used by the supplementary version of the panel, which is drawn from the sorted
    distance arrays; the Figure 1 panel takes the curve the plotting call returned (see
    :func:`_presynaptic_density_sheet`).
    """
    radii = np.linspace(0, 2.0, 200)
    voxel_volume = 0.01**3
    densities = []
    for r in radii:
        n_voxels = np.searchsorted(voxel_distances_sorted, r, side="right")
        n_cells = np.searchsorted(cell_distances_sorted, r, side="right")
        densities.append(0.0 if n_voxels == 0 else n_cells / (n_voxels * voxel_volume))
    return pd.DataFrame(
        {
            "Distance_To_Injection_mm": radii,
            "Cell_Density_Per_mm3": densities,
        }
    )


def _hist_table(values, max_val=None, show_zero=False, group=None, group_col="Group"):
    """Integer histogram as drawn by `barcodes_in_cells.plot_hist`.

    `plot_hist` normalises over the full bincount (zero included) and only then slices
    to the displayed range, so the proportions do not sum to 1 when ``show_zero`` is
    False. Reproduced here exactly.
    """
    values = np.asarray(values)
    if max_val is None:
        max_val = int(values.max()) + 1
    min_val = 0 if show_zero else 1
    counts = np.bincount(values.astype(int), minlength=max_val + 1)
    props = counts / np.sum(counts)
    counts = counts[min_val : max_val + 1]
    props = props[min_val : max_val + 1]
    table = pd.DataFrame(
        {
            "Value": np.arange(min_val, max_val + 1),
            "Cell_Count": counts,
            "Proportion": props,
        }
    )
    if group is not None:
        table.insert(0, group_col, group)
    return table


def _matrix_sheet(matrix, index_name="Presynaptic_Group"):
    """A labelled matrix as a flat sheet with its index as the first column."""
    out = pd.DataFrame(matrix).copy()
    out.index.name = index_name
    return out.reset_index()


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

FIG1_PANELS = [
    "Fig 1d Library abundance",
    "Fig 1e Unique fraction",
    "Fig 1f Library comparison abund",
    "Fig 1g Library compar unique",
    "Fig 1h Starter spread sim",
    "Fig 1k Presynaptic density",
    "Fig 1m Starter positions",
    "Fig 1n Pairwise distances",
]


#: Keys of ``fig1_plotted_data`` that belong to an image panel: what they hold is the
#: annotation drawn over a micrograph (arrows, labels, legend keys), which is not Source
#: Data. Listed so that a key added later is noticed rather than silently dropped.
FIG1_IMAGE_KEYS = (
    "starter_confocal",
    "tail_vs_local_labels",
    "injection_legend",
)

#: How the two AAV-Cre delivery routes of panels o and p are named in the figure.
FIG1_DELIVERY_LABELS = {"local": "Intracerebral", "tail": "Intravenous"}

#: Number of cells per mm^3 in V1, the factor between the two x-axes of panel j.
FIG1J_V1_CELL_DENSITY = 150e3

#: The libraries made in this study. Panels h and i compare them against libraries
#: published by other laboratories, whose barcode counts are those groups' data to
#: distribute; only our own curves are written to the workbook.
FIG1_OWN_LIBRARIES = (
    "Plasmid library",
    "Virus library",
    "2 wells",
    "12 wells",
    "RV2",
    "RV35",
)

#: Sheet note for a panel that also draws libraries published by other laboratories.
FIG1_PUBLISHED_LIBRARY_NOTE = (
    "Note: the panel also compares libraries published by other laboratories. Those "
    "curves are not tabulated here, as the barcode counts behind them are not ours to "
    "distribute; they are available from the original publications ({libraries})."
)


def _own_libraries_only(drawn, sheet):
    """Drop the curves of libraries published by other laboratories.

    Args:
        drawn (dict): Plotted curves, keyed by the library label of the figure.
        sheet (str): Sheet name, used in the message naming what was left out.

    Returns:
        tuple: ``(kept, dropped)``, the curves made in this study and the labels of the
        ones left out, in the plotted order.
    """
    kept = {name: curve for name, curve in drawn.items() if name in FIG1_OWN_LIBRARIES}
    dropped = [name for name in drawn if name not in FIG1_OWN_LIBRARIES]
    if dropped:
        print(
            f"[Source Data] Fig 1: {sheet}: leaving out libraries published elsewhere: "
            f"{dropped}"
        )
    return kept, dropped


def build_fig1_source_data(fig1_plotted_data=None):
    """Build the panel dictionary for Figure 1 from what the figure actually drew.

    The notebook collects the return value of every Figure 1 plotting call in
    ``fig1_plotted_data``, so each sheet holds the arrays that were handed to
    matplotlib rather than a second, re-derived copy of them. Nothing that is not drawn
    enters the workbook: the printed 95%/99% unique-labelling estimates, the pairwise
    distances behind the panel p kernel density estimate and the micrograph annotations
    are all dropped. The libraries published by other laboratories, which panels h and i
    compare ours against, are left out too — see :func:`_own_libraries_only`.

    Args:
        fig1_plotted_data (dict): The notebook's ``fig1_plotted_data``. Curve panels
            (``"library_abundance"`` and friends, from
            `viral_library.plot_barcode_counts_and_percentage` and
            `viral_library.plot_unique_label_fraction`) hold one ``x``/``y`` entry per
            library; ``"starter_spread_simulation"`` comes from
            `start_density_sim.plot_starter_spread_sim`, ``"presynaptic_density"`` from
            `rabies_cell_counting.plot_rabies_density`, and ``"starter_positions"`` and
            ``"pairwise_distances"`` from the two `starter_cell_counting` panels.

    Returns:
        dict: Sheet name to DataFrame. Sheets needing a worksheet note carry it in
        ``DataFrame.attrs``.
    """
    plotted = fig1_plotted_data or {}
    # Plotted key, sheet, and what turns one into the other; in panel order.
    builders = (
        ("library_abundance", "Fig 1d Library abundance", _abundance_sheet),
        ("library_unique_fraction", "Fig 1e Unique fraction", _unique_fraction_sheet),
        (
            "library_comparison_abundance",
            "Fig 1f Library comparison abund",
            _abundance_sheet,
        ),
        (
            "library_comparison_unique_fraction",
            "Fig 1g Library compar unique",
            _unique_fraction_sheet,
        ),
        (
            "starter_spread_simulation",
            "Fig 1h Starter spread sim",
            _starter_spread_sheet,
        ),
        (
            "presynaptic_density",
            "Fig 1k Presynaptic density",
            _presynaptic_density_sheet,
        ),
        ("starter_positions", "Fig 1m Starter positions", _starter_positions_sheet),
        ("pairwise_distances", "Fig 1n Pairwise distances", _pairwise_distances_sheet),
    )
    # The panels that draw one curve per viral or plasmid library.
    library_keys = {key for key, _, build in builders if build in _LIBRARY_SHEETS}
    panels = {}
    for key, sheet, build in builders:
        drawn = plotted.get(key)
        if not drawn:
            continue
        dropped = []
        if key in library_keys:
            drawn, dropped = _own_libraries_only(drawn, sheet)
            if not drawn:
                continue
        table = build(drawn)
        if dropped:
            published = FIG1_PUBLISHED_LIBRARY_NOTE.format(libraries="; ".join(dropped))
            already = table.attrs.get(NOTE_ATTR)
            _note(table, f"{already} {published}" if already else published)
        panels[sheet] = table

    known = {key for key, _, _ in builders} | set(FIG1_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Fig 1: no sheet for plotted panels {unknown}")

    return panels


def _abundance_sheet(drawn):
    """Panels d/f/h — the rank-abundance curve of every library, as drawn.

    Only the libraries made in this study reach the sheet; see
    :func:`_own_libraries_only`.

    Libraries above 500k barcodes are subsampled to stay inside Excel's 1,048,576-row
    limit, keeping the first 1,000 ranks in full and thinning the rest on a log scale to
    match the plotted axes (see :func:`_log_subsample_indices`).
    """
    frames = []
    subsampled = False
    for library, curve in drawn.items():
        index = np.asarray(curve["x"])
        counts = np.asarray(curve["y"])
        if len(index) > ABUNDANCE_SUBSAMPLE_THRESHOLD:
            keep = _log_subsample_indices(len(index))
            index, counts = index[keep], counts[keep]
            subsampled = True
        frames.append(
            pd.DataFrame(
                {
                    "Library": library,
                    "Barcode_Index": index.astype(int),
                    "Barcode_Abundance": counts,
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    return _note(table, ABUNDANCE_SUBSAMPLE_NOTE) if subsampled else table


def _unique_fraction_sheet(drawn):
    """Panels e/g/i — proportion of uniquely labelled cells, on the plotted grid.

    Only the libraries made in this study reach the sheet; see
    :func:`_own_libraries_only`.
    """
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "Library": library,
                    "Number_Of_Infections": np.asarray(curve["x"]).astype(int),
                    "Proportion_Uniquely_Labeled": np.asarray(curve["y"], dtype=float),
                }
            )
            for library, curve in drawn.items()
        ],
        ignore_index=True,
    )


#: The sheet builders that take one curve per library, and so are filtered down to the
#: libraries made in this study.
_LIBRARY_SHEETS = (_abundance_sheet, _unique_fraction_sheet)


def _starter_spread_sheet(drawn):
    """Panel j — probability of starter-to-starter spread, with its dashed thresholds.

    The panel carries a second x-axis holding the same points as an absolute density,
    which is why the proportion appears twice. The two threshold columns are the
    constants of the dashed reference lines, so the crossing the panel marks is in the
    sheet.
    """
    frames = []
    for key, curve in drawn.items():
        if not key.startswith("n="):
            continue
        proportion = np.asarray(curve["x"], dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "Presynaptic_Cells_Per_Starter": int(key[len("n=") :]),
                    "Starter_Proportion": proportion,
                    "Starter_Density_Per_mm3": proportion * FIG1J_V1_CELL_DENSITY,
                    "Probability_Of_Spread": np.asarray(curve["y"], dtype=float),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)

    hline = drawn.get("hline_spread_probability") or {}
    vline = drawn.get("vline_density_threshold") or {}
    if "y" in hline:
        table["Spread_Probability_Threshold"] = hline["y"]
    if "x" in vline:
        table["Starter_Proportion_At_Threshold"] = vline["x"]
    return _note(
        table,
        "Note: Starter_Density_Per_mm3 is Starter_Proportion on the second x-axis of "
        "the panel, taking a V1 cell density of 150,000 cells/mm3. The last two "
        "columns are constants: they are the dashed horizontal and vertical lines "
        "marking where the 20-presynaptic-cells curve reaches a 5% probability of "
        "spread.",
    )


def _presynaptic_density_sheet(drawn):
    """Panel m — cumulative isocortex cell density around the injection site."""
    curve = drawn["cumulative_density"]
    return pd.DataFrame(
        {
            "Distance_To_Injection_mm": np.asarray(curve["x"], dtype=float),
            "Cell_Density_Per_mm3": np.asarray(curve["y"], dtype=float),
        }
    )


def _starter_positions_sheet(drawn):
    """Panel o — medio-lateral and antero-posterior starter positions, in mm."""
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "Delivery_Route": FIG1_DELIVERY_LABELS.get(route, route),
                    "Mediolateral_mm": np.asarray(scatter["x"], dtype=float),
                    "Anteroposterior_mm": np.asarray(scatter["y"], dtype=float),
                }
            )
            for route, scatter in drawn.items()
        ],
        ignore_index=True,
    )


def _pairwise_distances_sheet(drawn):
    """Panel p — the normalised kernel density estimate of pairwise distances.

    The pairwise distances themselves are not drawn, only their kernel density estimate
    and its median, so only those are written here.
    """
    frames = []
    for route, curve in drawn.items():
        frames.append(
            pd.DataFrame(
                {
                    "Delivery_Route": FIG1_DELIVERY_LABELS.get(route, route),
                    "Pairwise_Distance_mm": np.asarray(curve["x"], dtype=float),
                    "Normalised_Cell_Density": np.asarray(curve["y"], dtype=float),
                    "Median_Pairwise_Distance_mm": curve["median"]["x"],
                }
            )
        )
    return _note(
        pd.concat(frames, ignore_index=True),
        "Note: Median_Pairwise_Distance_mm is constant within a delivery route; it is "
        "the median of the distribution, drawn as a marker above each curve.",
    )


def export_fig1_source_data(output_path, **kwargs):
    panels = build_fig1_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=FIG1_PANELS
    )


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

FIG2_PANELS = [
    "Fig 2h UMAP by cluster",
    "Fig 2i UMAP barcoded cells",
    "Fig 2j Gene expression map",
]

#: Rows kept per worksheet when a panel holds more cells than Excel can store.
SHEET_ROW_LIMIT = 1000000

#: Sheet note for every worksheet holding a subsampled scatter of single cells.
CELL_SUBSAMPLE_NOTE = (
    "Note: this panel draws more cells than Microsoft Excel's limit of 1,048,576 rows "
    "per worksheet allows, so a random subsample of 1,000,000 cells (numpy seed 0) is "
    "given here; barcoded cells, where marked, are all kept. The complete, "
    "un-subsampled dataset is deposited on Figshare (see figshare_dataset.md)."
)


def _subsample_cells(table, keep=None, limit=SHEET_ROW_LIMIT, seed=0):
    """Randomly thin a per-cell table down to ``limit`` rows.

    Scatter panels of the whole dataset hold about two million cells, which no single
    worksheet can take. Rows flagged by ``keep`` are never dropped, so the rare
    highlighted population of a panel (the barcoded cells of Fig 2i) survives in full;
    the remaining budget is filled with a random sample of the rest, drawn with a fixed
    seed so the workbook is reproducible.

    Args:
        table (pandas.DataFrame): One row per plotted cell, in plotted order.
        keep (np.ndarray, optional): Boolean mask of rows to keep unconditionally.
        limit (int): Maximum number of rows in the returned table.
        seed (int): Seed of the random generator used for the subsample.

    Returns:
        tuple: ``(table, subsampled)``, the table (thinned if needed, otherwise the one
        passed in) and whether anything was dropped.
    """
    if len(table) <= limit:
        return table, False

    rng = np.random.default_rng(seed)
    keep = (
        np.zeros(len(table), dtype=bool)
        if keep is None
        else np.asarray(keep, dtype=bool)
    )
    n_free = limit - int(keep.sum())
    if n_free <= 0:  # the kept rows alone already fill the sheet
        selected = np.flatnonzero(keep)[:limit]
    else:
        candidates = np.flatnonzero(~keep)
        chosen = rng.choice(candidates, size=n_free, replace=False)
        selected = np.union1d(np.flatnonzero(keep), chosen)
    return table.iloc[np.sort(selected)].reset_index(drop=True), True


def build_fig2_source_data(fig2_plotted_data=None):
    """Build the panel dictionary for Figure 2 from what the figure actually drew.

    The notebook collects the return value of every Figure 2 plotting call in
    ``fig2_plotted_data``, so the panels here are built from the plotted arrays
    themselves rather than re-derived from the AnnData objects. Nothing that is not
    drawn (cell identifiers, unplotted genes, unplotted cells) enters the workbook.

    Args:
        fig2_plotted_data (dict): The notebook's ``fig2_plotted_data``, with keys
            ``"umap_clusters"`` (from `cell_typing.plot_cell_clusters`, panel h),
            ``"umap_barcoded_cells"`` (from `cell_typing.plot_umap_barcoded_cells`,
            panel i) and ``"gene_expression"`` (from
            `gene_transcript_plots.plot_gene_expression_mosaic`, panel j).

    Returns:
        dict: Sheet name to DataFrame. Sheets that had to be subsampled carry their
        note in ``DataFrame.attrs``.
    """
    plotted = fig2_plotted_data or {}
    panels = {}

    clusters = (plotted.get("umap_clusters") or {}).get("clusters")
    if clusters is not None:
        panels["Fig 2h UMAP by cluster"] = _umap_cluster_sheet(clusters)

    barcoded = plotted.get("umap_barcoded_cells")
    if barcoded is not None:
        panels["Fig 2i UMAP barcoded cells"] = _umap_barcoded_sheet(barcoded)

    gene_expression = plotted.get("gene_expression")
    if gene_expression:
        panels["Fig 2j Gene expression map"] = _gene_expression_sheet(gene_expression)

    return panels


def _umap_cluster_sheet(clusters):
    """Panel h — every cell of the UMAP, with the cluster it is coloured by."""
    table = pd.DataFrame(
        {
            "UMAP_1": np.asarray(clusters["x"], dtype=float),
            "UMAP_2": np.asarray(clusters["y"], dtype=float),
            "Cluster": np.asarray(clusters["cluster"]),
        }
    )
    table, subsampled = _subsample_cells(table)
    return _note(table, CELL_SUBSAMPLE_NOTE) if subsampled else table


def _umap_barcoded_sheet(barcoded):
    """Panel i — the same UMAP, split into the two groups the panel draws.

    `plot_umap_barcoded_cells` draws the non-barcoded cells in light grey and then the
    barcoded ones in black, so both groups are one sheet with the group as a column.
    """
    labels = {"non_barcoded": "Non-barcoded", "barcoded": "Barcoded"}
    frames = [
        pd.DataFrame(
            {
                "Cell_Group": labels.get(group, group),
                "UMAP_1": np.asarray(drawn["x"], dtype=float),
                "UMAP_2": np.asarray(drawn["y"], dtype=float),
            }
        )
        for group, drawn in barcoded.items()
    ]
    table = pd.concat(frames, ignore_index=True)
    table, subsampled = _subsample_cells(table, keep=table["Cell_Group"] == "Barcoded")
    return _note(table, CELL_SUBSAMPLE_NOTE) if subsampled else table


def _gene_expression_sheet(gene_expression):
    """Panel j — one row per cell, one column per gene of the mosaic.

    Every subplot of the mosaic draws the same cells at the same coordinates and only
    the colouring gene changes, so the ten panels are one wide sheet rather than ten
    stacked copies of the coordinates. `plot_gene_expression_mosaic` returns its arrays
    sorted by expression (it draws faint cells first), so the sort is undone with the
    ``order`` it reports to line the genes up cell by cell.
    """
    coordinates = None
    expression = {}
    for gene, drawn in gene_expression.items():
        x = np.asarray(drawn["x"], dtype=float)
        y = np.asarray(drawn["y"], dtype=float)
        expr = np.asarray(drawn["expression"], dtype=float)
        if "order" in drawn:  # undo the low-expression-first draw order
            unsort = np.argsort(np.asarray(drawn["order"]))
            x, y, expr = x[unsort], y[unsort], expr[unsort]
        if coordinates is None:
            coordinates = (gene, x, y)
        else:
            first, x0, y0 = coordinates
            if not (np.array_equal(x, x0) and np.array_equal(y, y0)):
                raise ValueError(
                    f"Fig 2j: cells of gene {gene!r} do not line up with {first!r}; "
                    "the mosaic must plot the same cells for every gene."
                )
        expression[gene] = expr

    _, x, y = coordinates
    table = pd.DataFrame({"ARA_Z_px": x, "ARA_Y_px": y})
    for gene, expr in expression.items():
        table[gene] = expr
    table, subsampled = _subsample_cells(table)
    note = (
        "Note: coordinates are Allen CCF positions in 10 um voxels, as plotted. Gene "
        "columns hold the raw transcript count of each cell, which the panel maps to "
        "colour and opacity between zero and the 95th percentile of the non-zero "
        "counts of that gene."
    )
    if subsampled:
        note = f"{note} {CELL_SUBSAMPLE_NOTE}"
    return _note(table, note)


def export_fig2_source_data(output_path, **kwargs):
    panels = build_fig2_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=FIG2_PANELS
    )


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

FIG3_PANELS = [
    "Fig 3a Barcodes per cell",
    "Fig 3b Match to library",
    "Fig 3c Starters per barcode",
    "Fig 3d Presynaptic per barcode",
    "Fig 3e Spots per cell",
    "Fig 3f mCherry vs presynaptic",
]


#: Keys of ``fig3_plotted_data`` that belong to an image panel: the two-channel
#: micrograph of the barcode spots inside cells (panel e inset), which is pixel data and
#: not Source Data, so `sensitivity.plot_cells_spots` returns nothing. Listed so that a
#: key added later is noticed rather than silently dropped.
FIG3_IMAGE_KEYS = ("spots_in_cells_image",)

#: How the two cell populations of panel a are labelled in the figure, in drawing order.
FIG3A_POPULATIONS = ("Presynaptic cells", "Starter cells")


def build_fig3_source_data(fig3_plotted_data=None):
    """Build the panel dictionary for Figure 3 from what the figure actually drew.

    The notebook collects the return value of every Figure 3 plotting call in
    ``fig3_plotted_data``, so each sheet holds the arrays that were handed to matplotlib
    rather than a second, re-derived copy of them. Nothing that is not drawn enters the
    workbook: the cell and barcode identifiers, the printed match-to-library and
    regression statistics, the counts of the histograms that do not annotate their bars
    and the micrograph of panel e are all dropped.

    Args:
        fig3_plotted_data (dict): The notebook's ``fig3_plotted_data``. The histogram
            panels (``"barcodes_per_cell_presynaptic"``,
            ``"barcodes_per_cell_starter"``, ``"starters_per_barcode"`` and
            ``"spots_per_cell"``) come from `barcodes_in_cells.plot_hist`,
            ``"match_to_library"`` from `match_to_library.plot_matches_to_library`,
            ``"presynaptic_per_barcode"`` from
            `barcodes_in_cells.plot_presyn_per_barcode` and
            ``"mcherry_vs_presynaptic"`` from
            `mcherry_intensity.plot_mcherry_intensity_presyn`.

    Returns:
        dict: Sheet name to DataFrame. Sheets needing a worksheet note carry it in
        ``DataFrame.attrs``.
    """
    plotted = fig3_plotted_data or {}
    # Plotted keys, sheet, and what turns the drawn arrays into it; in panel order.
    builders = (
        (
            ("barcodes_per_cell_presynaptic", "barcodes_per_cell_starter"),
            "Fig 3a Barcodes per cell",
            _barcodes_per_cell_sheet,
        ),
        (("match_to_library",), "Fig 3b Match to library", _match_to_library_sheet),
        (
            ("starters_per_barcode",),
            "Fig 3c Starters per barcode",
            _starters_per_barcode_sheet,
        ),
        (
            ("presynaptic_per_barcode",),
            "Fig 3d Presynaptic per barcode",
            _presyn_per_barcode_sheet,
        ),
        (("spots_per_cell",), "Fig 3e Spots per cell", _spots_per_cell_sheet),
        (
            ("mcherry_vs_presynaptic",),
            "Fig 3f mCherry vs presynaptic",
            _mcherry_sheet,
        ),
    )
    panels = {
        sheet: build(*[plotted[key] for key in keys])
        for keys, sheet, build in builders
        if all(plotted.get(key) for key in keys)
    }

    known = {key for keys, _, _ in builders for key in keys} | set(FIG3_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Fig 3: no sheet for plotted panels {unknown}")

    return panels


def _stairs_sheet(drawn, value_col, proportion_col, count_col):
    """One `barcodes_in_cells.plot_hist` histogram, as its bars were drawn.

    The bar of an integer value covers ``value +/- 0.5``, so the bin edges add nothing
    over the value itself and are left out. The per-bar counts are written only when the
    panel annotates its bars with them.
    """
    histogram = drawn["histogram"]
    table = pd.DataFrame(
        {
            value_col: np.asarray(histogram["x"]).astype(int),
            proportion_col: np.asarray(histogram["y"], dtype=float),
        }
    )
    if "counts" in histogram:
        table[count_col] = np.asarray(histogram["counts"]).astype(int)
    return table


def _barcodes_per_cell_sheet(presynaptic, starter):
    """Panel a — barcodes per cell, presynaptic cells above starter cells."""
    frames = []
    for population, drawn in zip(FIG3A_POPULATIONS, (presynaptic, starter)):
        table = _stairs_sheet(
            drawn, "Barcodes_Per_Cell", "Proportion_Of_Barcodes", "Cell_Count"
        )
        table.insert(0, "Cell_Population", population)
        frames.append(table)
    return _note(
        pd.concat(frames, ignore_index=True),
        "Note: Cell_Count is the number of cells in each bar, written above it in the "
        "panel; Proportion_Of_Barcodes is that count over all cells of the population, "
        "which is the quantity drawn (the axis label of the published panel).",
    )


def _match_to_library_sheet(drawn):
    """Panel b — the three normalised histograms of library reads per barcode.

    The first row is the bin of barcodes absent from the library, which the panel draws
    as a separate bar left of the logarithmic axis; the viral library curve has no such
    bar, hence the missing value.
    """
    series = {
        "In_Situ_Barcode_Proportion": drawn["in_situ"],
        "Random_Barcode_Proportion": drawn["random"],
        "Library_Read_Proportion": drawn["viral_library"],
    }
    edges = np.asarray(drawn["viral_library"]["bin_edges"], dtype=float)
    table = pd.DataFrame(
        {
            "Bin_Min_Reads": np.insert(edges[:-1], 0, 0.0),
            "Bin_Max_Reads": np.insert(edges[1:], 0, edges[0]),
        }
    )
    for column, curve in series.items():
        values = np.asarray(curve["y"], dtype=float)
        zero_bin = curve.get("zero_bin_y", np.nan)
        table[column] = np.insert(values, 0, zero_bin)
    table["Library_Total_Reads"] = drawn["viral_library"]["total_reads_in_library"]
    return _note(
        table,
        "Note: bins are numbers of reads in the viral library. The first row is the "
        "bin of barcodes with no read in the library, drawn as a separate bar left of "
        "the logarithmic axis; the viral library curve has no such bar. "
        "Library_Total_Reads is constant: it is the total number of reads in the "
        "library, the factor turning the read axis into the proportion of unique reads "
        "the panel is labelled with.",
    )


def _starters_per_barcode_sheet(drawn):
    """Panel c — starter cells per barcode, with the counts annotating the bars."""
    return _note(
        _stairs_sheet(
            drawn, "Starters_Per_Barcode", "Proportion_Of_Barcodes", "Barcode_Count"
        ),
        "Note: Barcode_Count is the number of barcodes in each bar, written above it "
        "in the panel; Proportion_Of_Barcodes is that count over all barcodes.",
    )


def _presyn_per_barcode_sheet(drawn):
    """Panel d — presynaptic cells per barcode, orphan and non-orphan barcodes.

    Each barcode type is normalised by its own number of barcodes, as drawn.
    """
    frames = []
    for barcode_type, curve in drawn.items():
        edges = np.asarray(curve["bin_edges"], dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "Barcode_Type": barcode_type,
                    "Bin_Min": np.insert(edges[:-1], 0, 0.0),
                    "Bin_Max": np.insert(edges[1:], 0, 0.0),
                    "Proportion_Of_Barcodes": np.insert(
                        np.asarray(curve["y"], dtype=float), 0, curve["zero_bin_y"]
                    ),
                }
            )
        )
    return _note(
        pd.concat(frames, ignore_index=True),
        "Note: the first row of each barcode type is the proportion of barcodes with "
        "zero presynaptic cells, drawn as a separate bar left of the logarithmic axis.",
    )


def _spots_per_cell_sheet(drawn):
    """Panel e — barcode spots per cell, with the dotted detection threshold."""
    table = _stairs_sheet(
        drawn, "Barcode_Spots_Per_Cell", "Proportion_Of_Cells", "Cell_Count"
    )
    threshold = drawn["histogram"].get("min_spots_threshold")
    if threshold is None:
        return table
    table["Min_Spots_Threshold"] = threshold
    return _note(
        table,
        "Note: Min_Spots_Threshold is constant; it is the dotted vertical line of the "
        "panel, below which cells are not called barcoded.",
    )


def _mcherry_sheet(drawn):
    """Panel f — presynaptic cells against starter mCherry fluorescence, and its fit.

    Both are on the natural-logarithm axes the panel draws them on. The regression
    statistics the notebook prints are not drawn and are left out.
    """
    scatter = drawn["starter_cells"]
    frames = [
        pd.DataFrame(
            {
                "Series_Type": "Individual starter cell",
                "Log_mCherry_Fluorescence": np.asarray(scatter["x"], dtype=float),
                "Log_Presynaptic_Cells": np.asarray(scatter["y"], dtype=float),
            }
        )
    ]
    fit = drawn.get("robust_fit")
    if fit is not None:
        table = pd.DataFrame(
            {
                "Series_Type": "Robust fit",
                "Log_mCherry_Fluorescence": np.asarray(fit["x"], dtype=float),
                "Log_Presynaptic_Cells": np.asarray(fit["y"], dtype=float),
            }
        )
        for column, key in (
            ("Fit_CI_Lower", "ci_lower"),
            ("Fit_CI_Upper", "ci_upper"),
        ):
            if key in fit:
                table[column] = np.asarray(fit[key], dtype=float)
        frames.append(table)
    return _note(
        pd.concat(frames, ignore_index=True),
        "Note: both columns are natural logarithms, as plotted; the panel labels the "
        "axes with the fluorescence and cell numbers themselves. The presynaptic count "
        "of a starter includes the starter itself, hence the '+ 1' of the axis label. "
        "Rows of type 'Robust fit' are the fitted line seaborn drew and the bounds of "
        "its bootstrap confidence band; they are empty for the individual cells.",
    )


def export_fig3_source_data(output_path, **kwargs):
    panels = build_fig3_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=FIG3_PANELS
    )


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------

FIG4_PANELS = [
    "Fig 4a_1 Coronal cell positions",
    "Fig 4a_2 Flatmap cell positions",
    "Fig 4b Cortical depth VISp",
    "Fig 4c Starters per presynaptic",
    "Fig 4d Multibarcoded starters",
    "Fig 4e Example barcodes",
    "Fig 4f_1 Relative coords observed",
    "Fig 4f_2 Relative coords shuffled",
    "Fig 4g ML KDE vs shuffle",
]


#: Keys of ``fig4_plotted_data`` that belong to an image panel. Figure 4 has none: the
#: atlas contours of panels a and f, and the flatmap outline of panels b and f, are
#: images drawn inside a data-bearing panel rather than panels of their own, so no key
#: reports them. The tuple is kept so that a key added later is noticed rather than
#: silently dropped.
FIG4_IMAGE_KEYS = ()

#: How the two cell populations are named in the sheets of panels a, b, c and f.
FIG4_STARTER_LABEL = "Starter cell"
FIG4_PRESYNAPTIC_LABEL = "Presynaptic cell"

#: Name given to the grey background of every barcoded cell in panel f.
FIG4_ALL_CELLS_LABEL = "All barcoded cells"

#: The stacked series of panel e, in the order they are stacked.
FIG4E_SERIES = (
    "Barcode 1",
    "Barcode 2",
    "Barcode 3",
    "Barcode 4",
    "Any 2",
    "Any 3",
    "Any 4",
)


def build_fig4_source_data(fig4_plotted_data=None):
    """Build the panel dictionary for Figure 4 from what the figure actually drew.

    The notebook collects the return value of every Figure 4 plotting call in
    ``fig4_plotted_data``, so each sheet holds the arrays that were handed to matplotlib
    rather than a second, re-derived copy of them. Nothing that is not drawn enters the
    workbook: the cell identifiers, the barcode sets behind a starter's connections, the
    unplotted coordinate of the relative positions (the antero-posterior one, panels g
    and h draw only medio-lateral against depth) and the statistics the notebook prints
    are all dropped.

    Args:
        fig4_plotted_data (dict): The notebook's ``fig4_plotted_data``, with keys
            ``"all_cells_coronal"`` and ``"all_cells_flatmap"`` (the two panels of
            `spatial_plots_rabies.plot_all_rv_cells`), ``"cortical_depth"`` (from
            `spatial_plots_rabies.plot_layer_distribution`),
            ``"starters_per_presynaptic"`` (from `barcodes_in_cells.plot_hist`),
            ``"multibarcoded_starters"`` (from
            `barcodes_in_cells.plot_multibarcoded_starters`), ``"example_barcodes"``
            (from `spatial_plots_rabies.plot_example_barcodes`),
            ``"relative_coors_observed"`` and ``"relative_coors_shuffled"`` (two
            calls to `distance_between_cells.plot_relative_coors`) and ``"ml_kde"``, the
            observed
            kernel density estimate and shuffle band the notebook draws itself.

    Returns:
        dict: Sheet name to DataFrame. Sheets needing a worksheet note carry it in
        ``DataFrame.attrs``.
    """
    plotted = fig4_plotted_data or {}
    # Plotted key, sheet, and what turns one into the other; in panel order.
    builders = (
        (
            "all_cells_coronal",
            "Fig 4a_1 Coronal cell positions",
            _coronal_positions_sheet,
        ),
        (
            "all_cells_flatmap",
            "Fig 4a_2 Flatmap cell positions",
            _flatmap_positions_sheet,
        ),
        ("cortical_depth", "Fig 4b Cortical depth VISp", _cortical_depth_sheet),
        (
            "starters_per_presynaptic",
            "Fig 4c Starters per presynaptic",
            _starters_per_presynaptic_sheet,
        ),
        (
            "multibarcoded_starters",
            "Fig 4d Multibarcoded starters",
            _multibarcoded_sheet,
        ),
        ("example_barcodes", "Fig 4e Example barcodes", _example_barcodes_sheet),
        (
            "relative_coors_observed",
            "Fig 4f_1 Relative coords observed",
            _relative_coors_sheet,
        ),
        (
            "relative_coors_shuffled",
            "Fig 4f_2 Relative coords shuffled",
            _relative_coors_sheet,
        ),
        ("ml_kde", "Fig 4g ML KDE vs shuffle", _ml_kde_sheet),
    )
    panels = {
        sheet: build(plotted[key]) for key, sheet, build in builders if plotted.get(key)
    }

    known = {key for key, _, _ in builders} | set(FIG4_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Fig 4: no sheet for plotted panels {unknown}")

    return panels


def _cell_positions_sheet(drawn, xcol, ycol, note):
    """Panels a/b — every barcoded cell of one space, coloured by cortical area.

    The starter cells are the same rows redrawn in black on top of the area colours, so
    they are a flag rather than a second copy of the coordinates.
    """
    table = pd.DataFrame(
        {
            xcol: np.asarray(drawn["x"], dtype=float),
            ycol: np.asarray(drawn["y"], dtype=float),
            "Cortical_Area": np.asarray(drawn["cortical_area"]),
            "Is_Starter": np.asarray(drawn["is_starter"], dtype=bool),
        }
    )
    table, subsampled = _subsample_cells(table, keep=table["Is_Starter"].values)
    if subsampled:
        note = f"{note} {CELL_SUBSAMPLE_NOTE}"
    return _note(table, note)


def _coronal_positions_sheet(drawn):
    """Panel a — the coronal section, in the 10 um voxels of the atlas."""
    return _cell_positions_sheet(
        drawn,
        "ARA_Z_px",
        "ARA_Y_px",
        "Note: coordinates are Allen CCF positions in 10 um voxels, as plotted "
        "(ARA_Z_px is medio-lateral, drawn on a reversed axis; ARA_Y_px is "
        "dorso-ventral, drawn downwards). Every cell is drawn in the colour of its "
        "cortical area and the cells flagged Is_Starter are redrawn in black on top. "
        "The atlas outlines are an image and are not tabulated.",
    )


def _flatmap_positions_sheet(drawn):
    """Panel b — the same cells on the cortical flatmap."""
    return _cell_positions_sheet(
        drawn,
        "Flatmap_X",
        "Flatmap_Y",
        "Note: coordinates are positions on the Allen cortical flatmap in 10 um "
        "voxels, as plotted, with both axes reversed in the panel. Every cell is drawn "
        "in the colour of its cortical area and the cells flagged Is_Starter are "
        "redrawn in black on top. The flatmap outlines are an image and are not "
        "tabulated.",
    )


def _cortical_depth_sheet(drawn):
    """Panel c — the cortical depth of every barcoded cell of VISp, by population.

    The panel draws these depths as the two halves of a split violin. The dashed layer
    boundaries it draws over them are Allen atlas averages rather than measurements of
    this dataset, so they are left out, as are the atlas outlines of panels a, b and f.
    """
    frames = []
    for key, series in (
        ("starter_cells", FIG4_STARTER_LABEL),
        ("presynaptic_cells", FIG4_PRESYNAPTIC_LABEL),
    ):
        population = drawn[key]
        frames.append(
            pd.DataFrame(
                {
                    "Series": series,
                    "Cortical_Depth_um": np.asarray(population["values"], dtype=float),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    table, subsampled = _subsample_cells(table)
    note = (
        "Note: one row per barcoded cell of VISp, holding the cortical depth the panel "
        "draws it at. The panel shows the two populations as the two halves of a split "
        "violin, a kernel density estimate of these depths (Gaussian kernel, bandwidth "
        "adjusted by 0.5) whose width is normalised, so its horizontal axis is in "
        "arbitrary units. The dashed horizontal lines of the panel are the average "
        "cortical layer boundaries of the Allen atlas, reference geometry rather than "
        "data of this experiment, and are deliberately not tabulated."
    )
    if subsampled:
        note = f"{note} {CELL_SUBSAMPLE_NOTE}"
    return _note(table, note)


def _starters_per_presynaptic_sheet(drawn):
    """Panel d — how many starter cells each presynaptic cell is connected to."""
    return _note(
        _stairs_sheet(drawn, "Connected_Starters", "Proportion_Of_Cells", "Cell_Count"),
        "Note: each bar covers its integer value +/- 0.5. Cell_Count is the number of "
        "presynaptic cells in the bar, written above it in the panel; "
        "Proportion_Of_Cells is that count over all presynaptic cells, including the "
        "cells connected to no starter, which the panel does not draw.",
    )


def _multibarcoded_sheet(drawn):
    """Panel e — the presynaptic cells of every multi-barcoded starter, by barcode.

    One column per stacked series, in stacking order, so each row is one bar of the
    panel. The bar order is the panel's own: starters sorted by their total number of
    presynaptic cells.
    """
    series = [key for key in FIG4E_SERIES if key in drawn]
    series += [key for key in drawn if key not in FIG4E_SERIES]
    first = drawn[series[0]]
    table = pd.DataFrame(
        {"Starter_Rank": np.asarray(first["x"], dtype=int)},
    )
    for key in series:
        table[key.replace(" ", "_")] = np.asarray(drawn[key]["y"], dtype=float)
    return _note(
        table,
        "Note: one row per bar of the panel, ordered as the panel orders them. The "
        "Barcode_N columns hold the number of presynaptic cells carrying only the Nth "
        "most abundant barcode of that starter; the Any_N columns hold the number "
        "carrying N of its barcodes. The bars are stacked in that column order.",
    )


def _example_barcodes_sheet(drawn):
    """Panel f — the cells of each example barcode, in both spaces.

    The coronal and the flatmap panel draw the same cells, so a cell is one row holding
    both coordinate pairs. The grey background of every barcoded cell is drawn first and
    is the first block of rows. The lines the panel draws from a starter to each of its
    presynaptic cells join the coordinates of those rows and add nothing.
    """
    frames = []
    for key, series in drawn.items():
        is_all_cells = key == "all_cells"
        starter = np.asarray(series.get("is_starter", []), dtype=bool)
        if is_all_cells:
            point_type = np.full(len(series["x"]), FIG4_ALL_CELLS_LABEL)
        else:
            point_type = np.where(starter, FIG4_STARTER_LABEL, FIG4_PRESYNAPTIC_LABEL)
        frames.append(
            pd.DataFrame(
                {
                    "Barcode": FIG4_ALL_CELLS_LABEL if is_all_cells else key,
                    "Point_Type": point_type,
                    "ARA_Z_px": np.asarray(series["x"], dtype=float),
                    "ARA_Y_px": np.asarray(series["y"], dtype=float),
                    "Flatmap_X": np.asarray(series["flatmap_x"], dtype=float),
                    "Flatmap_Y": np.asarray(series["flatmap_y"], dtype=float),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    table, subsampled = _subsample_cells(
        table, keep=(table["Barcode"] != FIG4_ALL_CELLS_LABEL).values
    )
    note = (
        "Note: coordinates are Allen CCF positions and cortical flatmap positions in "
        "10 um voxels, as plotted; the same cell is drawn at both, once on the coronal "
        "panel and once on the flatmap panel. The rows whose Barcode is "
        f"'{FIG4_ALL_CELLS_LABEL}' are the grey background of all barcoded cells, "
        "drawn without distinguishing starters. The lines of the panel join each "
        "starter to the cells of the same barcode."
    )
    if subsampled:
        note = f"{note} {CELL_SUBSAMPLE_NOTE}"
    return _note(table, note)


def _relative_coors_sheet(drawn):
    """Panels g/h — position of every presynaptic cell relative to its starter.

    Only the two coordinates the panel draws are written: the antero-posterior offset is
    in the plotted array but on neither axis.
    """
    scatter = drawn["relative_coors"]
    table = pd.DataFrame(
        {
            "Relative_ML_Location_mm": np.asarray(scatter["x"], dtype=float),
            "Relative_Cortical_Depth_mm": np.asarray(scatter["y"], dtype=float),
        }
    )
    table, subsampled = _subsample_cells(table)
    return _note(table, CELL_SUBSAMPLE_NOTE) if subsampled else table


def _ml_kde_sheet(drawn):
    """Panel i — the observed medio-lateral positions and the shuffle band.

    The observed curve is a kernel density estimate of one value per presynaptic cell,
    so those values are what is written; the shuffle band is drawn from its percentiles
    on the notebook's grid, which is a different number of rows, so each series keeps
    its own rows.
    """
    observed = drawn["observed"]
    shuffle = drawn["shuffle"]
    frames = [
        pd.DataFrame(
            {
                "Series": "Observed",
                "Relative_ML_Location_mm": np.asarray(observed["values"], dtype=float),
                "Shuffle_Density_Lower": np.nan,
                "Shuffle_Density_Upper": np.nan,
            }
        ),
        pd.DataFrame(
            {
                "Series": "Shuffle",
                "Relative_ML_Location_mm": np.asarray(shuffle["x"], dtype=float),
                "Shuffle_Density_Lower": np.asarray(shuffle["lower"], dtype=float),
                "Shuffle_Density_Upper": np.asarray(shuffle["upper"], dtype=float),
            }
        ),
    ]
    table = pd.concat(frames, ignore_index=True)
    table, subsampled = _subsample_cells(
        table, keep=(table["Series"] == "Shuffle").values
    )
    bandwidth = observed.get("bw_method")
    note = (
        "Note: the Observed rows are the medio-lateral position of every presynaptic "
        "cell relative to its starter, the same values as the Relative_ML_Location_mm "
        "column of the panel g sheet; the panel draws their kernel density estimate "
        f"(Gaussian kernel, bandwidth method {bandwidth}). The Shuffle rows hold the "
        "2.5th and 97.5th percentile of the same estimate over the barcode-shuffled "
        "nulls, on the grid the notebook evaluates them on, drawn as the grey band."
    )
    if subsampled:
        note = f"{note} {CELL_SUBSAMPLE_NOTE}"
    return _note(table, note)


def export_fig4_source_data(output_path, **kwargs):
    panels = build_fig4_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=FIG4_PANELS
    )


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------

FIG5_PANELS = [
    "Fig 5a Presyn pos by layer",
    "Fig 5b Counts matrix",
    "Fig 5c Mean input fraction",
    "Fig 5d Input fraction by layer",
    "Fig 5e Connectivity diagram CI",
    "Fig 5e Input vs shuffle",
    "Fig 5f Mean output fraction",
    "Fig 5g Output vs shuffle",
    "Fig 5h Inhibitory marker dotplot",
    "Fig 5i_1 Interneuron counts",
    "Fig 5i_2 Interneuron input fract",
    "Fig 5j Interneuron diagram CI",
]


#: Keys of ``fig5_plotted_data`` that belong to an image panel. Figure 5 has none —
#: every one of its panels is data-bearing — so the tuple is empty. It is kept so that
#: an image panel added later is declared here instead of being reported as a plotted
#: panel without a sheet.
FIG5_IMAGE_KEYS = ()

#: Point types of panel a, keyed by their key in the plotted element.
FIG5A_POINT_TYPES = {"presynaptic": "Presynaptic cell", "starter": "Starter cell"}


def _sheet_index_name(label, default="Presynaptic_Group"):
    """An axis label of the figure as a column name (``"Presynaptic layer"`` -> ...)."""
    words = str(label or "").split()
    return "_".join(word.capitalize() for word in words) if words else default


def _fig5h_dotplot_sheet(drawn):
    """Panel h — inhibitory cell-type marker dotplot values, as scanpy computed them."""
    colors = pd.DataFrame(drawn["dot_color_df"])
    sizes = pd.DataFrame(drawn["dot_size_df"])
    long_colors = (
        colors.rename_axis("Cell_Type")
        .reset_index()
        .melt(id_vars="Cell_Type", var_name="Gene", value_name="Scaled_Mean_Expression")
    )
    long_sizes = (
        sizes.rename_axis("Cell_Type")
        .reset_index()
        .melt(id_vars="Cell_Type", var_name="Gene", value_name="Fraction_Expressing")
    )
    table = long_colors.merge(long_sizes, on=["Cell_Type", "Gene"], how="left")
    return _note(
        table,
        "Note: Scaled_Mean_Expression is the dot colour, the mean expression rescaled "
        "to 0-1 within each gene (scanpy's standard_scale='var'), not the raw mean. "
        "Fraction_Expressing is the dot size, i.e. the percentage of cells expressing "
        "the gene in each cell type.",
    )


def build_fig5_source_data(fig5_plotted_data=None):
    """Build the panel dictionary for Figure 5 from what the figure actually drew.

    The notebook collects the return value of every Figure 5 plotting call in
    ``fig5_plotted_data``, so each sheet holds the numbers that were handed to
    matplotlib. Nothing that is not drawn enters the workbook: the 1,000 shuffles and
    1,000 bootstrap replicates behind the panels, the row sums of the count matrices,
    the antero-posterior offset of the panel a presynaptic cells and the confidence
    bounds of panel d (which draws none) are all dropped.

    Args:
        fig5_plotted_data (dict): The notebook's ``fig5_plotted_data``. The matrix
            panels (``"counts"``, ``"input_fraction"``, ``"output_fraction"`` and their
            ``"interneuron_"`` counterparts) come from
            `connectivity_matrices.plot_area_by_area_connectivity`, the two bubble plots
            from `connectivity_matrices.bubble_plot`, the two connectivity diagrams from
            `connectivity_matrices.connectivity_diagram_mpl`,
            ``"input_fraction_by_layer"`` from
            `bootstrapping.plot_confidence_intervals` and
            ``"presynaptic_positions"`` from the scatter loop of the figure cell.

    Returns:
        dict: Sheet name to DataFrame. Sheets needing a worksheet note carry it in
        ``DataFrame.attrs``.
    """
    plotted = fig5_plotted_data or {}
    # Plotted key, sheet, and what turns one into the other; in panel order.
    builders = (
        (
            "presynaptic_positions",
            "Fig 5a Presyn pos by layer",
            _presyn_positions_sheet,
        ),
        ("counts", "Fig 5b Counts matrix", _fig5_matrix_sheet),
        ("input_fraction", "Fig 5c Mean input fraction", _fig5_matrix_sheet),
        (
            "input_fraction_by_layer",
            "Fig 5d Input fraction by layer",
            _input_fraction_ci_sheet,
        ),
        (
            "connectivity_diagram",
            "Fig 5e Connectivity diagram CI",
            _diagram_edges_sheet,
        ),
        ("input_vs_shuffle", "Fig 5e Input vs shuffle", _bubble_sheet),
        ("output_fraction", "Fig 5f Mean output fraction", _fig5_matrix_sheet),
        ("output_vs_shuffle", "Fig 5g Output vs shuffle", _bubble_sheet),
        (
            "inhibitory_dotplot",
            "Fig 5h Inhibitory marker dotplot",
            _fig5h_dotplot_sheet,
        ),
        ("interneuron_counts", "Fig 5i_1 Interneuron counts", _fig5_matrix_sheet),
        (
            "interneuron_input_fraction",
            "Fig 5i_2 Interneuron input fract",
            _fig5_matrix_sheet,
        ),
        (
            "interneuron_diagram",
            "Fig 5j Interneuron diagram CI",
            _diagram_edges_sheet,
        ),
    )
    panels = {
        sheet: build(plotted[key]) for key, sheet, build in builders if plotted.get(key)
    }

    known = {key for key, _, _ in builders} | set(FIG5_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Fig 5: no sheet for plotted panels {unknown}")

    return panels


def _fig5_matrix_sheet(drawn):
    """Panels b/c/g/i/j — a connectivity matrix exactly as colour-mapped.

    The matrix keeps the row and column order of the panel, so the sheet can be read as
    the panel is read: one row per presynaptic group, one column per starter group. The
    count panels also print the number of starter cells under each column, which is
    appended as a last row.
    """
    element = drawn["matrix"]
    matrix = pd.DataFrame(element["matrix"])
    index_name = _sheet_index_name(element.get("ylabel"))
    table = _matrix_sheet(matrix, index_name=index_name)

    starter_counts = element.get("starter_counts")
    if starter_counts is None:
        return table
    starter_counts = pd.Series(starter_counts)
    row = {index_name: "Starter cell count"}
    for column in table.columns[1:]:
        row[column] = starter_counts.get(column, np.nan)
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    return _note(
        table,
        "Note: the last row is not part of the matrix. It is the number of starter "
        "cells printed under each column of the panel, next to the 'N starters:' "
        "label.",
    )


def _presyn_positions_sheet(drawn):
    """Panel a — one row per point of the per-starter-layer scatters.

    Only the two plotted coordinates are given: the panel plots the medio-lateral offset
    of each presynaptic cell from its starter against its absolute cortical depth, so
    the antero-posterior offset, which the notebook also computes, is not Source Data.
    """
    frames = []
    for layer, groups in drawn.items():
        for key, label in FIG5A_POINT_TYPES.items():
            scatter = groups.get(key)
            if scatter is None:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "Starter_Layer": layer,
                        "Point_Type": label,
                        "Relative_ML_um": np.asarray(scatter["x"], dtype=float),
                        "Cortical_Depth_um": np.asarray(scatter["y"], dtype=float),
                    }
                )
            )
    table = pd.concat(frames, ignore_index=True)
    note = (
        "Note: Relative_ML_um is the medio-lateral offset from the starter cell, so it "
        "is 0 for every starter cell; Cortical_Depth_um is an absolute depth below the "
        "pia. The antero-posterior offset is not plotted and is therefore not given. "
        "The dashed horizontal lines of the panel are the average cortical layer "
        "boundaries of the Allen atlas, reference geometry rather than a measurement "
        "of this dataset, and are deliberately not tabulated."
    )
    return _note(table, note)


def _input_fraction_ci_sheet(drawn):
    """Panel d — the input fraction of every starter cell, with the means drawn on top.

    One sub-panel per starter layer, one distribution per presynaptic layer within it.
    ``Drawn_As`` says how the panel shows that distribution: as a violin when it holds
    more than seven starter cells, as a jittered scatter of the raw values otherwise.
    The bootstrap confidence intervals are not in the sheet because the panel draws
    none — only the individual values and the mean.
    """
    frames = []
    for starter, element in drawn.items():
        order = list(element["presynaptic_order"])
        drawn_as = element.get("drawn_as", {})
        for presyn in order:
            values = np.asarray(element["values"].get(presyn, []), dtype=float)
            frames.append(
                pd.DataFrame(
                    {
                        "Series_Type": "Individual",
                        "Starter_Layer": starter,
                        "Presynaptic_Layer": presyn,
                        "Input_Fraction": values,
                        "Drawn_As": drawn_as.get(presyn, ""),
                    }
                )
            )
        frames.append(
            pd.DataFrame(
                {
                    "Series_Type": "Mean",
                    "Starter_Layer": starter,
                    "Presynaptic_Layer": order,
                    "Input_Fraction": [element["means"][presyn] for presyn in order],
                    "Drawn_As": "mean line",
                }
            )
        )
    return _note(
        pd.concat(frames, ignore_index=True),
        "Note: rows with Series_Type 'Individual' are one starter cell each; rows with "
        "Series_Type 'Mean' are the black lines drawn over each distribution. Drawn_As "
        "gives how the individual values are shown ('violin' for more than seven "
        "starter cells, 'points' otherwise). The panel draws no confidence interval, "
        "so none is given.",
    )


def _diagram_edges_sheet(drawn):
    """Panels e/k — one row per arrow of the connectivity diagram.

    Arrow width encodes the input fraction and arrow colour the width of its bootstrap
    confidence interval, so both are given. Only connections above the panel's input
    fraction cutoff are drawn, and only those are in the sheet.
    """
    element = drawn["edges"]
    edges = pd.DataFrame(element["edges"])
    table = edges.rename(
        columns={
            "starter": "Starter_Group",
            "presynaptic": "Presynaptic_Group",
            "input_fraction": "Input_Fraction",
            "ci_lower": "CI_Lower",
            "ci_upper": "CI_Upper",
            "ci_width": "CI_Width",
        }
    )[
        [
            "Starter_Group",
            "Presynaptic_Group",
            "Input_Fraction",
            "CI_Lower",
            "CI_Upper",
            "CI_Width",
        ]
    ]
    cutoff = element.get("min_fraction_cutoff")
    return _note(
        table,
        "Note: one row per arrow drawn, from the presynaptic group to the starter "
        f"group. Only connections with an input fraction of at least {cutoff} are "
        "drawn, and only those are listed. CI_Lower and CI_Upper are the 2.5th and "
        "97.5th percentiles of 1,000 bootstrap resamplings of the starter cells; "
        "CI_Width is their difference, which the colour of the arrow encodes.",
    )


def _bubble_sheet(drawn):
    """Panels f/h — observed-versus-shuffle bubble plot.

    Bubble area encodes the absolute log2 ratio of the observed connectivity over the
    shuffled one, bubble colour the signed -log10 of the FDR-corrected p-value, and a
    black outline marks the cells below the significance level. The 1,000 shuffles
    themselves are not drawn and are not in the sheet.
    """
    element = drawn["bubbles"]
    bubbles = pd.DataFrame(element["bubbles"])
    table = pd.DataFrame(
        {
            "Presynaptic_Group": bubbles["y_label"].to_numpy(),
            "Starter_Group": bubbles["x_label"].to_numpy(),
            "Log2_Observed_Over_Shuffle": bubbles["log_ratio"].to_numpy(),
            "FDR_Corrected_P_Value": bubbles["p_value"].to_numpy(),
            "Signed_Log10_P_Value": bubbles["color_value"].to_numpy(),
            "Significant": bubbles["significant"].to_numpy(),
        }
    )
    return _note(
        table,
        "Note: one row per bubble. Bubble area is proportional to "
        "|Log2_Observed_Over_Shuffle|; bubble colour is Signed_Log10_P_Value, the sign "
        "of the log ratio times -log10 of the FDR-corrected p-value. Significant is "
        f"True where FDR_Corrected_P_Value < {element.get('alpha')}, the cells the "
        "panel outlines in black.",
    )


def export_fig5_source_data(output_path, **kwargs):
    panels = build_fig5_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=FIG5_PANELS
    )


# ---------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------

FIG6_PANELS = [
    "Fig 6b Starter positions",
    "Fig 6c Presynaptic positions",
    "Fig 6c Smoothed starter map",
    "Fig 6e Starter vs presyn ML",
    "Fig 6e Running average and CI",
    "Fig 6f Azimuth running avg",
]

#: Keys of ``fig6_plotted_data`` that belong to an image panel. The flatmap outlines of
#: panels b, c and d and the retinotopy inset of panel d are atlas images, not
#: measurements, so they have no worksheet. Listed so that a key added later is noticed
#: rather than silently dropped.
FIG6_IMAGE_KEYS = ("retinotopy_map",)

#: Panel b draws the presynaptic cells of panel c again, in grey behind the starters.
FIG6B_NOTE = (
    "Note: the starter cells are drawn coloured by their medio-lateral position "
    "(Starter_ML_Position_mm, turbo_r colour map, -1 to 1 mm). The grey cloud drawn "
    "behind them is the presynaptic population of panel c; it is not repeated here, "
    "see the 'Fig 6c Presynaptic positions' worksheet."
)

#: The smoothed map is an image whose opacity carries the local data support.
FIG6D_NOTE = (
    "Note: Gaussian-smoothed mean starter medio-lateral position (mm) of the "
    "presynaptic cells at each flatmap location, drawn as the panel image. Column "
    "headers are flatmap X, the first column is flatmap Y; rows run from the bottom of "
    "the panel upwards. Blank cells lie outside the area covered by presynaptic cells "
    "and are not drawn. Inside it, the panel additionally fades pixels supported by "
    "few cells (opacity proportional to the summed Gaussian weight, saturating at 50), "
    "which is a display property of the image rather than a measurement."
)

#: What the shaded band and the dashed line of panel e are.
FIG6E_NOTE = (
    "Note: Shuffle_Lower and Shuffle_Upper are the 2.5th and 97.5th percentiles of "
    "1,000 bootstrap resamplings of the starter cells, drawn as the shaded band; the "
    "individual resamplings are not drawn and are not given. "
    "Mean_Starter_ML_Position_mm is constant: it is the mean starter position over all "
    "presynaptic cells, drawn as the dashed horizontal line."
)

#: Panel f draws the running average only, not the per-cell azimuths behind it.
FIG6F_NOTE = (
    "Note: the panel draws the Gaussian-weighted running average of the receptive "
    "field azimuth of the presynaptic cells against their medio-lateral position. The "
    "per-cell azimuth values that go into the average are not drawn in this panel and "
    "so are not given here; the azimuth map they are read from is the Allen atlas "
    "retinotopy shown as the inset of panel d."
)


def build_fig6_source_data(fig6_plotted_data=None):
    """Build the panel dictionary for Figure 6 from what the figure actually drew.

    The notebook collects every drawn Figure 6 series in ``fig6_plotted_data``, so each
    sheet holds the arrays that were handed to matplotlib rather than a second,
    re-derived copy of them. Nothing that is not drawn enters the workbook: the cell
    identifiers, the flatmap coordinates of cells outside the plotted panels, the
    individual bootstrap resamplings behind the shuffle band of panel e, the per-cell
    receptive-field azimuths that only enter panel f through their running average and
    the atlas images are all dropped.

    Args:
        fig6_plotted_data (dict): The notebook's ``fig6_plotted_data``, with keys
            ``"starter_positions"`` (panel b), ``"presynaptic_positions"`` (panel c),
            ``"smoothed_starter_map"`` (panel d), ``"starter_vs_presyn_ml"`` (panel e,
            both its scatter and its running average) and
            ``"azimuth_running_average"`` (panel f). Image-only keys are listed in
            :data:`FIG6_IMAGE_KEYS`.

    Returns:
        dict: Sheet name to DataFrame. Sheets needing a worksheet note carry it in
        ``DataFrame.attrs``.
    """
    plotted = fig6_plotted_data or {}
    # Plotted key, sheet, and what turns one into the other; in panel order. Panel e
    # gives two sheets, its scatter and its running average, from the same key.
    builders = (
        (
            "starter_positions",
            "Fig 6b Starter positions",
            partial(
                _longrange_flatmap_sheet,
                value="Starter_ML_Position_mm",
                note=FIG6B_NOTE,
            ),
        ),
        (
            "presynaptic_positions",
            "Fig 6c Presynaptic positions",
            partial(_longrange_flatmap_sheet, value="Starter_ML_Position_mm"),
        ),
        (
            "smoothed_starter_map",
            "Fig 6c Smoothed starter map",
            partial(_longrange_smoothed_map_sheet, note=FIG6D_NOTE),
        ),
        (
            "starter_vs_presyn_ml",
            "Fig 6e Starter vs presyn ML",
            partial(_longrange_scatter_sheet, value="Starter_ML_Position_mm"),
        ),
        (
            "starter_vs_presyn_ml",
            "Fig 6e Running average and CI",
            partial(
                _longrange_running_average_sheet,
                value="Running_Average_Starter_ML_mm",
                mean_column="Mean_Starter_ML_Position_mm",
                note=FIG6E_NOTE,
            ),
        ),
        (
            "azimuth_running_average",
            "Fig 6f Azimuth running avg",
            partial(
                _longrange_running_average_sheet,
                value="Running_Average_Azimuth_deg",
                note=FIG6F_NOTE,
            ),
        ),
    )
    panels = {
        sheet: build(plotted[key]) for key, sheet, build in builders if plotted.get(key)
    }

    known = {key for key, _, _ in builders} | set(FIG6_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Fig 6: no sheet for plotted panels {unknown}")

    return panels


# The four sheet builders below serve both long-range figures: Figure 6, along the
# medio-lateral axis and azimuth, and the supplementary reviewer figure, along the
# antero-posterior axis and elevation (see
# :func:`brisc.source_data.supplementary.build_suppfig_reviewer_source_data`). The two
# figures draw the same panels of different quantities, so only the name of the value
# column changes; the presynaptic position is the same medio-lateral flatmap axis in
# both, hence the shared ``Presynaptic_ML_mm``.

#: Name of the presynaptic-position column of the two graph panels, in both figures.
LONGRANGE_AXIS_COLUMN = "Presynaptic_ML_mm"


def _longrange_flatmap_sheet(drawn, value, note=None):
    """Panels b and c — flatmap positions coloured by the starter value, as drawn.

    Every series the panel scattered is stacked into one table. Panel b's grey cloud is
    the presynaptic population of panel c, which the notebook leaves out of panel b's
    plotted data rather than repeating it.
    """
    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "Flatmap_X": np.asarray(series["x"], dtype=float),
                    "Flatmap_Y": np.asarray(series["y"], dtype=float),
                    value: np.asarray(series["c"], dtype=float),
                }
            )
            for series in drawn.values()
        ],
        ignore_index=True,
    )
    return table if note is None else _note(table, note)


def _longrange_smoothed_map_sheet(drawn, note=None):
    """Panel d — the smoothed starter-value map, as the image it is drawn as.

    One row per image row and one column per image column, the flatmap coordinates of
    the panel's ``extent`` as the row and column labels. Pixels the panel draws fully
    transparent are left blank: they are outside the area covered by presynaptic cells,
    where the smoothed value is not shown.
    """
    panel = drawn["smoothed_map"]
    image = np.asarray(panel["image"], dtype=float)
    alpha = panel.get("alpha")
    if alpha is not None:
        image = np.where(np.asarray(alpha, dtype=float) > 0, image, np.nan)
    x0, x1, y0, y1 = panel["extent"]
    table = pd.DataFrame(image)
    table.columns = np.linspace(x0, x1, image.shape[1]).round(1)
    table.insert(0, "Flatmap_Y", np.linspace(y0, y1, image.shape[0]).round(1))
    return table if note is None else _note(table, note)


def _longrange_scatter_sheet(drawn, value, note=None):
    """Panel e — one point per presynaptic cell, its position against its starter's.

    The running average, its shuffle band and the mean-position line of the same panel
    are on their own worksheet: they are given on the evaluation grid of the average,
    not per cell.
    """
    cells = drawn["cells"]
    table = pd.DataFrame(
        {
            LONGRANGE_AXIS_COLUMN: np.asarray(cells["x"], dtype=float),
            value: np.asarray(cells["y"], dtype=float),
        }
    )
    return table if note is None else _note(table, note)


def _longrange_running_average_sheet(drawn, value, mean_column=None, note=None):
    """Panels e and f — a running average, with the band and mean line if drawn.

    ``value`` is the ``Running_Average_...`` column the panel plots and ``mean_column``
    the constant column of the dashed mean line, which only panel e draws.
    """
    curve = drawn["running_average"]
    table = pd.DataFrame(
        {
            LONGRANGE_AXIS_COLUMN: np.asarray(curve["x"], dtype=float),
            value: np.asarray(curve["y"], dtype=float),
        }
    )
    if curve.get("shuffle_lower") is not None:
        table["Shuffle_Lower"] = np.asarray(curve["shuffle_lower"], dtype=float)
        table["Shuffle_Upper"] = np.asarray(curve["shuffle_upper"], dtype=float)
    if mean_column is not None and curve.get("mean_starter_position") is not None:
        table[mean_column] = float(curve["mean_starter_position"])
    return table if note is None else _note(table, note)


def export_fig6_source_data(output_path, **kwargs):
    panels = build_fig6_source_data(**kwargs)
    notes = {
        name: table.attrs[NOTE_ATTR]
        for name, table in panels.items()
        if NOTE_ATTR in getattr(table, "attrs", {})
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=FIG6_PANELS
    )
