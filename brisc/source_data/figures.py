"""Source Data builders and exporters for the main manuscript figures.

Each ``build_figN_source_data`` takes the variables the figure notebook actually plots
and returns one DataFrame per data-bearing panel, holding the numbers as drawn (same
evaluation grid, same transform). Pure-image panels — micrographs, atlas outlines,
schematics — have no worksheet.

The ``FIGn_PANELS`` lists are passed to :func:`~brisc.source_data.io.save_excel_sheets`
so that a panel that fails to build is reported instead of silently skipped.
"""

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
    "Fig 1f Rescue scaling abundance",
    "Fig 1g Rescue scaling unique",
    "Fig 1h Library comparison abund",
    "Fig 1i Library compar unique",
    "Fig 1j Starter spread sim",
    "Fig 1m Presynaptic density",
    "Fig 1o Starter positions",
    "Fig 1p Pairwise distances",
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


def build_fig1_source_data(fig1_plotted_data=None):
    """Build the panel dictionary for Figure 1 from what the figure actually drew.

    The notebook collects the return value of every Figure 1 plotting call in
    ``fig1_plotted_data``, so each sheet holds the arrays that were handed to
    matplotlib rather than a second, re-derived copy of them. Nothing that is not drawn
    enters the workbook: the printed 95%/99% unique-labelling estimates, the pairwise
    distances behind the panel p kernel density estimate and the micrograph annotations
    are all dropped.

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
            "rescue_scaling_abundance",
            "Fig 1f Rescue scaling abundance",
            _abundance_sheet,
        ),
        (
            "rescue_scaling_unique_fraction",
            "Fig 1g Rescue scaling unique",
            _unique_fraction_sheet,
        ),
        (
            "library_comparison_abundance",
            "Fig 1h Library comparison abund",
            _abundance_sheet,
        ),
        (
            "library_comparison_unique_fraction",
            "Fig 1i Library compar unique",
            _unique_fraction_sheet,
        ),
        (
            "starter_spread_simulation",
            "Fig 1j Starter spread sim",
            _starter_spread_sheet,
        ),
        (
            "presynaptic_density",
            "Fig 1m Presynaptic density",
            _presynaptic_density_sheet,
        ),
        ("starter_positions", "Fig 1o Starter positions", _starter_positions_sheet),
        ("pairwise_distances", "Fig 1p Pairwise distances", _pairwise_distances_sheet),
    )
    panels = {
        sheet: build(plotted[key]) for key, sheet, build in builders if plotted.get(key)
    }

    known = {key for key, _, _ in builders} | set(FIG1_IMAGE_KEYS)
    unknown = [key for key in plotted if key not in known]
    if unknown:
        print(f"[Source Data] !! Fig 1: no sheet for plotted panels {unknown}")

    return panels


def _abundance_sheet(drawn):
    """Panels d/f/h — the rank-abundance curve of every library, as drawn.

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
    """Panels e/g/i — proportion of uniquely labelled cells, on the plotted grid."""
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


def build_fig3_source_data(
    cells_df=None,
    barcodes_df=None,
    in_situ_barcode_matches=None,
    random_barcode_matches=None,
    rv35_library=None,
    good_cells=None,
    valid_mcherry=None,
):
    """Build the panel dictionary for Figure 3.

    Args:
        cells_df (pd.DataFrame): Barcoded cells with ``n_unique_barcodes`` and
            ``is_starter`` (panel a).
        barcodes_df (pd.DataFrame): Per-barcode ``n_starters``/``n_presynaptic``
            (panels c, d).
        in_situ_barcode_matches (pd.DataFrame): In-situ barcode matches (panel b).
        random_barcode_matches (pd.DataFrame): Random control matches (panel b).
        rv35_library (pd.DataFrame): Viral library read counts (panel b).
        good_cells (pd.DataFrame): Rabies-positive cells with ``spot_count`` (panel e).
        valid_mcherry (pd.DataFrame): Starter mCherry measurements (panel f).
    """
    panels = {}

    if cells_df is not None and "n_unique_barcodes" in cells_df.columns:
        presynaptic = cells_df[~cells_df["is_starter"].astype(bool)]
        starters = cells_df[cells_df["is_starter"].astype(bool)]
        panels["Fig 3a Barcodes per cell"] = pd.concat(
            [
                _hist_table(
                    presynaptic["n_unique_barcodes"].values,
                    max_val=6,
                    group="Presynaptic cells",
                    group_col="Cell_Population",
                ),
                _hist_table(
                    starters["n_unique_barcodes"].values,
                    max_val=6,
                    group="Starter cells",
                    group_col="Cell_Population",
                ),
            ],
            ignore_index=True,
        ).rename(columns={"Value": "Barcodes_Per_Cell"})

    if in_situ_barcode_matches is not None and random_barcode_matches is not None:
        panels["Fig 3b Match to library"] = _match_to_library_table(
            in_situ_barcode_matches, random_barcode_matches, rv35_library
        )

    if barcodes_df is not None:
        if "n_starters" in barcodes_df.columns:
            panels["Fig 3c Starters per barcode"] = _hist_table(
                barcodes_df["n_starters"].values, show_zero=True
            ).rename(
                columns={
                    "Value": "Starters_Per_Barcode",
                    "Cell_Count": "Barcode_Count",
                    "Proportion": "Proportion_Of_Barcodes",
                }
            )
        if "n_presynaptic" in barcodes_df.columns:
            panels["Fig 3d Presynaptic per barcode"] = _presyn_per_barcode_table(
                barcodes_df
            )

    if good_cells is not None and "spot_count" in good_cells.columns:
        panels["Fig 3e Spots per cell"] = _hist_table(
            good_cells["spot_count"].values, max_val=40, show_zero=True
        ).rename(
            columns={
                "Value": "Barcode_Spots_Per_Cell",
                "Proportion": "Proportion_Of_Cells",
            }
        )

    if valid_mcherry is not None and "intensity_mean-0" in valid_mcherry.columns:
        intensity = valid_mcherry["intensity_mean-0"].values
        n_presynaptic = valid_mcherry["n_presynaptic"].values
        panels["Fig 3f mCherry vs presynaptic"] = pd.DataFrame(
            {
                "Starter_Cell_ID": valid_mcherry.index.astype(str),
                "mCherry_Fluorescence_AU": intensity,
                "Number_Of_Presynaptic_Cells": n_presynaptic,
                "Log_mCherry_Fluorescence": np.log(intensity.astype(float)),
                "Log_Number_Of_Presynaptic": np.log(n_presynaptic.astype(float)),
            }
        )

    return panels


def _match_to_library_table(in_situ, random, rv35_library):
    """Panel b — the three normalised histograms drawn by `plot_matches_to_library`."""
    bin_edges = np.logspace(0, 6, num=20)
    bin_edges = np.insert(bin_edges, 0, 0)

    in_situ_counts = (
        in_situ["ham_lib_bc_counts"]
        .where(in_situ["ham_min_edit_distance"] <= 0, 0)
        .values
    )
    random_counts = (
        random["lib_bc_counts"].where(random["min_edit_distance"] <= 0, 0).values
    )

    in_situ_hist, _ = np.histogram(in_situ_counts, bins=bin_edges)
    random_hist, _ = np.histogram(random_counts, bins=bin_edges)
    in_situ_hist = in_situ_hist / np.sum(in_situ_hist)
    random_hist = random_hist / np.sum(random_hist)

    table = pd.DataFrame(
        {
            "Bin_Min_Reads": bin_edges[:-1],
            "Bin_Max_Reads": bin_edges[1:],
            "In_Situ_Barcode_Proportion": in_situ_hist,
            "Random_Barcode_Proportion": random_hist,
        }
    )

    if rv35_library is not None:
        sequences = np.flip(np.asarray(rv35_library["counts"]))
        edge_positions = sequences.searchsorted(bin_edges)
        library = np.zeros(len(bin_edges) - 1)
        for i in range(len(bin_edges) - 1):
            start, stop = edge_positions[i : i + 2]
            library[i] = sequences[start:stop].sum()
        table["Library_Read_Proportion"] = library / np.sum(library)
        table["Library_Total_Reads"] = np.sum(rv35_library["counts"])
    return table


def _presyn_per_barcode_table(barcodes_df):
    """Panel d — log-spaced histograms of presynaptic cells per barcode."""
    orphan = barcodes_df[barcodes_df["n_starters"] == 0]["n_presynaptic"].values
    non_orphan = barcodes_df[barcodes_df["n_starters"] > 0]["n_presynaptic"].values
    max_n = max(orphan.max(), non_orphan.max())
    bins = 10 ** (np.arange(0, np.log10(max_n), 0.16))

    frames = []
    for label, values in (
        ("Orphan barcodes", orphan),
        ("Non-orphan barcodes", non_orphan),
    ):
        hist, edges = np.histogram(values, bins=bins)
        frames.append(
            pd.DataFrame(
                {
                    "Barcode_Type": label,
                    "Bin_Min": np.insert(edges[:-1], 0, 0.0),
                    "Bin_Max": np.insert(edges[1:], 0, 0.0),
                    "Proportion_Of_Barcodes": np.insert(
                        hist / len(values), 0, (values == 0).sum() / len(values)
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def export_fig3_source_data(output_path, **kwargs):
    panels = build_fig3_source_data(**kwargs)
    notes = {
        "Fig 3d Presynaptic per barcode": (
            "The first row of each barcode type is the proportion of barcodes with "
            "zero "
            "presynaptic cells, drawn as a separate bar left of the log axis."
        )
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=FIG3_PANELS
    )


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------

FIG4_PANELS = [
    "Fig 4a Coronal cell positions",
    "Fig 4b Flatmap cell positions",
    "Fig 4c Cortical depth VISp",
    "Fig 4d Starters per presynaptic",
    "Fig 4e Multibarcoded starters",
    "Fig 4f Example barcodes",
    "Fig 4g Relative coords observed",
    "Fig 4h Relative coords shuffled",
    "Fig 4i ML KDE vs shuffle",
]

FIG4_EXAMPLE_BARCODES = ("GCTTCATGCAATTG", "GCTCTTCCTTAATA", "ATAAATAAGGCGCT")


def build_fig4_source_data(
    cells_df=None,
    presynaptic=None,
    multibarcoded_starters=None,
    relative_presyn_coords_flatmap=None,
    all_shuffled_distances_flatmap=None,
    x_grid=None,
    ml_kde_shuffled=None,
    bw_method=0.05,
    example_barcodes=FIG4_EXAMPLE_BARCODES,
):
    """Build the panel dictionary for Figure 4.

    Args:
        cells_df (pd.DataFrame): All barcoded cells with atlas and flatmap coordinates
            (panels a, b, c, f).
        presynaptic (pd.DataFrame): Presynaptic cells with ``n_starters`` (panel d).
        multibarcoded_starters (pd.DataFrame): Output of
            `barcodes_in_cells.analyze_multibarcoded_starters` (panel e).
        relative_presyn_coords_flatmap (np.ndarray): (N, 3) observed relative
            coordinates in flatmap pixels (panels g, i).
        all_shuffled_distances_flatmap (list): Per-shuffle (N, 3) relative coordinates
            (panels h, i).
        x_grid (np.ndarray): KDE evaluation grid of panel i.
        ml_kde_shuffled (np.ndarray): (n_shuffles, len(x_grid)) shuffled KDEs (panel i).
        bw_method (float): KDE bandwidth used by the panel.
        example_barcodes (tuple): The barcodes drawn in panel f.
    """
    panels = {}

    if cells_df is not None:
        panels["Fig 4a Coronal cell positions"] = pd.DataFrame(
            {
                "Cell_ID": cells_df.index.astype(str),
                "ARA_Y": cells_df["ara_y"].values,
                "ARA_Z": cells_df["ara_z"].values,
                "Cortical_Area": cells_df["cortical_area"].values,
                "Is_Starter": cells_df["is_starter"].values,
            }
        )
        panels["Fig 4b Flatmap cell positions"] = pd.DataFrame(
            {
                "Cell_ID": cells_df.index.astype(str),
                "Flatmap_X": cells_df["flatmap_x"].values,
                "Flatmap_Y": cells_df["flatmap_y"].values,
                "Cortical_Area": cells_df["cortical_area"].values,
                "Is_Starter": cells_df["is_starter"].values,
            }
        )
        visp = cells_df[cells_df["cortical_area"] == "VISp"]
        panels["Fig 4c Cortical depth VISp"] = pd.DataFrame(
            {
                "Cell_ID": visp.index.astype(str),
                "Cortical_Depth_um": visp["normalised_layers"].values * 10,
                "Is_Starter": visp["is_starter"].values,
            }
        )
        panels["Fig 4f Example barcodes"] = _example_barcodes_table(
            cells_df, example_barcodes
        )

    if presynaptic is not None and "n_starters" in presynaptic.columns:
        panels["Fig 4d Starters per presynaptic"] = _hist_table(
            presynaptic["n_starters"].values, max_val=4
        ).rename(
            columns={"Value": "Connected_Starters", "Proportion": "Proportion_Of_Cells"}
        )

    if multibarcoded_starters is not None:
        panels["Fig 4e Multibarcoded starters"] = _multibarcoded_table(
            multibarcoded_starters
        )

    if relative_presyn_coords_flatmap is not None:
        panels["Fig 4g Relative coords observed"] = _relative_coords_table(
            relative_presyn_coords_flatmap
        )
    if all_shuffled_distances_flatmap is not None:
        panels["Fig 4h Relative coords shuffled"] = _relative_coords_table(
            all_shuffled_distances_flatmap[1]
        )

    if (
        relative_presyn_coords_flatmap is not None
        and x_grid is not None
        and ml_kde_shuffled is not None
    ):
        from scipy.stats import gaussian_kde

        observed = relative_presyn_coords_flatmap[:, 0] / 100
        observed = observed[~np.isnan(observed)]
        low, high = np.percentile(ml_kde_shuffled, [2.5, 97.5], axis=0)
        panels["Fig 4i ML KDE vs shuffle"] = pd.DataFrame(
            {
                "Relative_ML_Location_mm": x_grid,
                "Observed_Density": gaussian_kde(observed, bw_method=bw_method)(x_grid),
                "Shuffle_Density_Lower": low,
                "Shuffle_Density_Upper": high,
            }
        )

    return panels


def _relative_coords_table(coords):
    """Panels g/h — relative M-L and depth of each presynaptic cell, in mm."""
    coords = np.asarray(coords) / 100
    return pd.DataFrame(
        {
            "Relative_ML_Location_mm": coords[:, 0],
            "Relative_Cortical_Depth_mm": coords[:, 2],
        }
    )


def _example_barcodes_table(cells_df, barcodes):
    """Panel f — the cells carrying each example barcode, in both spaces."""
    frames = []
    for barcode in barcodes:
        carries = cells_df["all_barcodes"].apply(lambda bcs: barcode in bcs)
        subset = cells_df[carries]
        frames.append(
            pd.DataFrame(
                {
                    "Barcode": barcode,
                    "Cell_ID": subset.index.astype(str),
                    "Is_Starter": subset["is_starter"].values,
                    "ARA_Y": subset["ara_y"].values,
                    "ARA_Z": subset["ara_z"].values,
                    "Flatmap_X": subset["flatmap_x"].values,
                    "Flatmap_Y": subset["flatmap_y"].values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _multibarcoded_table(multibarcoded_starters):
    """Panel e — presynaptic cells attributable to each barcode of a starter."""
    rows = []
    for _, row in multibarcoded_starters.iterrows():
        per_barcode = sorted(row["n_presyn_per_barcode"], reverse=True)
        for rank, count in enumerate(per_barcode, start=1):
            rows.append(
                {
                    "Starter_Cell_ID": row["starter_cell_id"],
                    "Barcode_Rank": rank,
                    "Presynaptic_Cells_With_Barcode_Only": count,
                    "Total_Presynaptic_Cells": row["n_presyn"],
                }
            )
        counts = np.asarray(row["barcode_counts"])
        for n_shared, count in enumerate(counts):
            if n_shared < 2:
                continue
            rows.append(
                {
                    "Starter_Cell_ID": row["starter_cell_id"],
                    "Barcode_Rank": f"shared by {n_shared}",
                    "Presynaptic_Cells_With_Barcode_Only": count,
                    "Total_Presynaptic_Cells": row["n_presyn"],
                }
            )
    return pd.DataFrame(rows)


def export_fig4_source_data(output_path, **kwargs):
    panels = build_fig4_source_data(**kwargs)
    return save_excel_sheets(panels, output_path, expected=FIG4_PANELS)


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------

FIG5_PANELS = [
    "Fig 5a Presyn pos by layer",
    "Fig 5b Counts matrix",
    "Fig 5c Mean input fraction",
    "Fig 5d Input fraction by layer",
    "Fig 5e Connectivity diagram CI",
    "Fig 5f Input vs shuffle",
    "Fig 5g Mean output fraction",
    "Fig 5h Output vs shuffle",
    "Fig 5i Interneuron counts",
    "Fig 5j Interneuron input fract",
    "Fig 5k Interneuron diagram CI",
]


def build_fig5_source_data(
    starters_df=None,
    counts_df=None,
    starter_counts=None,
    presynaptic_counts=None,
    mean_input_fraction=None,
    mean_input_frac_df=None,
    fractions_df=None,
    lower_df=None,
    upper_df=None,
    input_fraction_log_ratio=None,
    input_fraction_pval=None,
    output_fraction=None,
    output_fraction_log_ratio=None,
    output_fraction_pval=None,
    inh_counts_df=None,
    inh_starter_counts=None,
    inh_presynaptic_counts=None,
    inh_mean_input_fraction=None,
    inh_lower_df=None,
    inh_upper_df=None,
):
    """Build the panel dictionary for Figure 5.

    Every argument is a variable of `figure5_connectivity_matrices.ipynb`; the matrices
    are written with their layer / cell-type labels as the first column.
    """
    panels = {}

    if starters_df is not None:
        panels["Fig 5a Presyn pos by layer"] = _presyn_positions_by_layer(starters_df)

    if counts_df is not None:
        panels["Fig 5b Counts matrix"] = _counts_matrix_sheet(
            counts_df, starter_counts, presynaptic_counts
        )

    if mean_input_fraction is not None:
        panels["Fig 5c Mean input fraction"] = _matrix_sheet(mean_input_fraction)

    if fractions_df is not None:
        panels["Fig 5d Input fraction by layer"] = _input_fraction_points(
            fractions_df, mean_input_frac_df
        )

    if mean_input_fraction is not None and lower_df is not None:
        panels["Fig 5e Connectivity diagram CI"] = _matrix_with_ci(
            mean_input_fraction, lower_df, upper_df
        )

    if input_fraction_log_ratio is not None:
        panels["Fig 5f Input vs shuffle"] = _log_ratio_table(
            input_fraction_log_ratio, input_fraction_pval
        )

    if output_fraction is not None:
        panels["Fig 5g Mean output fraction"] = _matrix_sheet(output_fraction)

    if output_fraction_log_ratio is not None:
        panels["Fig 5h Output vs shuffle"] = _log_ratio_table(
            output_fraction_log_ratio, output_fraction_pval
        )

    if inh_counts_df is not None:
        panels["Fig 5i Interneuron counts"] = _counts_matrix_sheet(
            inh_counts_df,
            inh_starter_counts,
            inh_presynaptic_counts,
            index_name="Presynaptic_Cell_Type",
        )

    if inh_mean_input_fraction is not None:
        panels["Fig 5j Interneuron input fract"] = _matrix_sheet(
            inh_mean_input_fraction, index_name="Presynaptic_Cell_Type"
        )
        if inh_lower_df is not None:
            panels["Fig 5k Interneuron diagram CI"] = _matrix_with_ci(
                inh_mean_input_fraction,
                inh_lower_df,
                inh_upper_df,
                index_name="Presynaptic_Cell_Type",
            )

    return panels


def _counts_matrix_sheet(
    counts, starter_counts, presynaptic_counts, index_name="Presynaptic_Group"
):
    """A connectivity count matrix with its starter and presynaptic marginals."""
    table = _matrix_sheet(counts, index_name=index_name)
    if presynaptic_counts is not None:
        table["Total_Presynaptic_Cells"] = np.asarray(presynaptic_counts)
    if starter_counts is not None:
        starter_counts = pd.Series(starter_counts)
        row = {index_name: "Starter cell count"}
        for column in table.columns[1:]:
            row[column] = starter_counts.get(column, np.nan)
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    return table


def _presyn_positions_by_layer(starters_df):
    """Panel a — presynaptic positions relative to their starter, by layer."""
    scale = 10  # flatmap pixels to microns
    frames = []
    for layer, group in starters_df.groupby("cortical_layer", observed=True):
        relative = np.hstack(group["presynaptic_coors_relative"].values)[0]
        absolute = np.hstack(group["presynaptic_coors"].values)[0]
        frames.append(
            pd.DataFrame(
                {
                    "Starter_Layer": layer,
                    "Point_Type": "Presynaptic cell",
                    "Relative_ML_um": relative[:, 1] * scale,
                    "Relative_AP_um": relative[:, 0] * scale,
                    "Cortical_Depth_um": absolute[:, 2] * scale,
                }
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    "Starter_Layer": layer,
                    "Point_Type": "Starter cell",
                    "Relative_ML_um": 0.0,
                    "Relative_AP_um": np.nan,
                    "Cortical_Depth_um": group["flatmap_z_normalised"].values * scale,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _input_fraction_points(fractions_df, mean_input_frac_df=None):
    """Panel d — per-starter input fractions, plus the group means drawn on top."""
    grouping = "cortical_layer"
    value_cols = [c for c in fractions_df.columns if c != grouping]
    points = fractions_df.melt(
        id_vars=grouping,
        value_vars=value_cols,
        var_name="Presynaptic_Layer",
        value_name="Input_Fraction",
    )
    points = points.rename(columns={grouping: "Starter_Layer"})
    points.insert(0, "Series_Type", "Individual")
    frames = [points]
    if mean_input_frac_df is not None:
        means = (
            pd.DataFrame(mean_input_frac_df)
            .rename_axis("Presynaptic_Layer")
            .reset_index()
            .melt(
                id_vars="Presynaptic_Layer",
                var_name="Starter_Layer",
                value_name="Input_Fraction",
            )
        )
        means.insert(0, "Series_Type", "Mean")
        frames.append(means)
    return pd.concat(frames, ignore_index=True)


def _matrix_with_ci(matrix, lower, upper, index_name="Presynaptic_Group"):
    """A matrix with its bootstrap confidence interval, in long form."""

    def _long(df, name):
        return (
            pd.DataFrame(df)
            .rename_axis(index_name)
            .reset_index()
            .melt(id_vars=index_name, var_name="Starter_Group", value_name=name)
        )

    table = _long(matrix, "Input_Fraction")
    table["CI_Lower"] = _long(lower, "CI_Lower")["CI_Lower"]
    if upper is not None:
        table["CI_Upper"] = _long(upper, "CI_Upper")["CI_Upper"]
    return table


def _log_ratio_table(log_ratio, pvalues):
    """Panels f/h — observed-vs-shuffle log ratio and its FDR-corrected p-value."""
    table = (
        pd.DataFrame(log_ratio)
        .rename_axis("Presynaptic_Group")
        .reset_index()
        .melt(
            id_vars="Presynaptic_Group",
            var_name="Starter_Group",
            value_name="Log2_Observed_Over_Shuffle",
        )
    )
    if pvalues is not None:
        long_p = (
            pd.DataFrame(pvalues)
            .rename_axis("Presynaptic_Group")
            .reset_index()
            .melt(
                id_vars="Presynaptic_Group",
                var_name="Starter_Group",
                value_name="FDR_Corrected_P_Value",
            )
        )
        table["FDR_Corrected_P_Value"] = long_p["FDR_Corrected_P_Value"]
    return table


def export_fig5_source_data(output_path, **kwargs):
    panels = build_fig5_source_data(**kwargs)
    return save_excel_sheets(panels, output_path, expected=FIG5_PANELS)


# ---------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------

FIG6_PANELS = [
    "Fig 6b Starter positions",
    "Fig 6c Presynaptic positions",
    "Fig 6d Smoothed starter map",
    "Fig 6e Starter vs presyn ML",
    "Fig 6e Running average and CI",
    "Fig 6f Presynaptic azimuth",
    "Fig 6f Azimuth running avg",
]


def build_long_range_source_data(
    prefix="Fig 6",
    v1_starter_cells=None,
    starter_panel_values=None,
    presy_xy=None,
    presyn_panel_values=None,
    starter_value_label="Starter_ML_Position_mm",
    presynaptic_axis_values=None,
    ctx_img=None,
    ctx_mask=None,
    xlim=(150, 1050),
    ylim=(810, 1330),
    running_average_x=None,
    running_average=None,
    conf_int=None,
    mean_position=None,
    retinotopy=None,
    retinotopy_label="Receptive_Field_Azimuth_deg",
    retinotopy_name="Azimuth",
    retinotopy_running_average=None,
    panel_letters=("b", "c", "d", "e", "f"),
):
    """Shared builder for the two long-range figures.

    Figure 6 and the supplementary reviewer figure have the same five panels: starter
    and presynaptic flatmap positions, a smoothed map of starter position, the
    starter-vs-presynaptic scatter with its running average and shuffle band, and a
    retinotopy running average. Only the quantity carried by the colour axis differs.

    Every value argument is expected in the units the panel draws; only the flatmap
    coordinates are raw.
    """
    b, c, d, e, f = panel_letters
    panels = {}

    if v1_starter_cells is not None and starter_panel_values is not None:
        v1_xy = v1_starter_cells[["flatmap_x", "flatmap_y"]].values.T
        panels[f"{prefix} {b} Starter positions"] = pd.DataFrame(
            {
                "Starter_Cell_ID": v1_starter_cells.index.astype(str),
                "Flatmap_X": v1_xy[0],
                "Flatmap_Y": v1_xy[1],
                starter_value_label: np.asarray(starter_panel_values),
            }
        )

    if presy_xy is not None and presyn_panel_values is not None:
        panels[f"{prefix} {c} Presynaptic positions"] = pd.DataFrame(
            {
                "Flatmap_X": presy_xy[0],
                "Flatmap_Y": presy_xy[1],
                starter_value_label: np.asarray(presyn_panel_values),
            }
        )

    if ctx_img is not None:
        image = np.asarray(ctx_img, dtype=float)
        if ctx_mask is not None:
            image = image * np.asarray(ctx_mask, dtype=float)
        table = pd.DataFrame(image)
        table.columns = np.linspace(xlim[0], xlim[1], image.shape[1]).round(1)
        table.insert(
            0, "Flatmap_Y", np.linspace(ylim[0], ylim[1], image.shape[0]).round(1)
        )
        panels[f"{prefix} {d} Smoothed starter map"] = table

    if presynaptic_axis_values is not None and presyn_panel_values is not None:
        panels[f"{prefix} {e} Starter vs presyn ML"] = pd.DataFrame(
            {
                "Presynaptic_ML_mm": np.asarray(presynaptic_axis_values),
                starter_value_label: np.asarray(presyn_panel_values),
            }
        )

    if running_average_x is not None and running_average is not None:
        table = pd.DataFrame(
            {
                "Presynaptic_ML_mm": np.asarray(running_average_x),
                f"Running_Average_{starter_value_label}": np.asarray(running_average),
            }
        )
        if conf_int is not None:
            conf_int = np.asarray(conf_int)
            table["Shuffle_Lower"] = conf_int[0]
            table["Shuffle_Upper"] = conf_int[1]
        if mean_position is not None:
            table[f"Mean_{starter_value_label}"] = mean_position
        panels[f"{prefix} {e} Running average and CI"] = table

    if retinotopy is not None and presynaptic_axis_values is not None:
        panels[f"{prefix} {f} Presynaptic {retinotopy_name.lower()}"] = pd.DataFrame(
            {
                "Presynaptic_ML_mm": np.asarray(presynaptic_axis_values),
                retinotopy_label: np.asarray(retinotopy),
            }
        )

    if running_average_x is not None and retinotopy_running_average is not None:
        panels[
            f"{prefix} {f} {retinotopy_name.capitalize()} running avg"
        ] = pd.DataFrame(
            {
                "Presynaptic_ML_mm": np.asarray(running_average_x),
                retinotopy_label: np.asarray(retinotopy_running_average),
            }
        )

    return panels


def build_fig6_source_data(
    v1_starter_cells=None,
    presy_xy=None,
    presy_azel=None,
    relative_starter_pos=None,
    center_abs=None,
    scale=0.01,
    ctx_img=None,
    ctx_mask=None,
    xlim=(150, 1050),
    ylim=(810, 1330),
    x_calc=None,
    pres_vs_start_kde=None,
    conf_int=None,
    azi_kde=None,
    mean_position=None,
):
    """Build the panel dictionary for Figure 6.

    Args are the variables of `figure6_long_range.ipynb`. ``relative_starter_pos`` and
    ``presy_xy`` are already restricted to the cells with valid coordinates, in the same
    order, so they line up row by row.
    """

    def rel_pos(x):
        return -(np.asarray(x) - center_abs)

    starter_values = None
    if v1_starter_cells is not None and center_abs is not None:
        starter_values = rel_pos(v1_starter_cells["flatmap_x"].values) * scale

    return build_long_range_source_data(
        prefix="Fig",
        panel_letters=("6b", "6c", "6d", "6e", "6f"),
        v1_starter_cells=v1_starter_cells,
        starter_panel_values=starter_values,
        presy_xy=presy_xy,
        presyn_panel_values=(
            None
            if relative_starter_pos is None
            else np.asarray(relative_starter_pos) * scale
        ),
        starter_value_label="Starter_ML_Position_mm",
        presynaptic_axis_values=(
            None
            if presy_xy is None or center_abs is None
            else rel_pos(presy_xy[0]) * scale
        ),
        ctx_img=ctx_img,
        ctx_mask=ctx_mask,
        xlim=xlim,
        ylim=ylim,
        running_average_x=(
            None if x_calc is None or center_abs is None else rel_pos(x_calc) * scale
        ),
        running_average=(
            None if pres_vs_start_kde is None else np.asarray(pres_vs_start_kde) * scale
        ),
        conf_int=None if conf_int is None else np.asarray(conf_int) * scale,
        mean_position=None if mean_position is None else mean_position * scale,
        retinotopy=None if presy_azel is None else np.asarray(presy_azel)[0],
        retinotopy_label="Receptive_Field_Azimuth_deg",
        retinotopy_running_average=azi_kde,
    )


def export_fig6_source_data(output_path, **kwargs):
    panels = build_fig6_source_data(**kwargs)
    notes = {
        "Fig 6d Smoothed starter map": (
            "Gaussian-smoothed mean starter medio-lateral position (mm) over the "
            "flatmap. Column headers are flatmap X, the first column is flatmap Y."
        )
    }
    return save_excel_sheets(
        panels, output_path, notes_dict=notes, expected=FIG6_PANELS
    )
