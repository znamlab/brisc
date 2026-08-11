"""Redraw manuscript panels from the exported Source Data workbooks.

Everything here reads *only* the ``.xlsx`` files produced by
:mod:`brisc.source_data.figures` and :mod:`brisc.source_data.supplementary` --
never the raw data. A panel that cannot be drawn from its sheet alone is a sign
that the sheet is missing something.

The plot kind is inferred from column names and dtypes rather than from a
hand-written spec per panel, so sheets added later are handled without changes
here. What cannot be read off the numbers -- logarithmic axes, an equal aspect
ratio, inverted atlas axes, the colour maps of a mosaic -- is declared per panel
in :data:`PANEL_SPECS`, matched on the sheet name.
"""

import re
from functools import partial
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brisc.source_data.io import read_source_data_workbook

#: Maximum number of levels of a categorical column still usable as a hue.
MAX_HUE_LEVELS = 12

#: Column pairs that are known to be 2D coordinates of the same space.
COORD_PAIRS = [
    ("UMAP_1", "UMAP_2"),
    ("flatmap_x", "flatmap_y"),
    ("Relative_X_um", "Relative_Y_um"),
    ("azimuth", "elevation"),
]

#: Columns that identify a row rather than carry a plottable value.
ID_PATTERN = re.compile(r"(^|_)id$|^barcode$|^main_barcode$", re.IGNORECASE)

#: Confidence-band column names, drawn as a shaded region around the curve.
_CI_LOWER = re.compile(r"(lower|ci_low|_low)$", re.IGNORECASE)
_CI_UPPER = re.compile(r"(upper|ci_up|_high)$", re.IGNORECASE)

FIGSIZE = (4.5, 3.6)


def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _plottable_numeric_columns(df):
    """Numeric columns excluding row identifiers."""
    return [
        c for c in df.columns if _is_numeric(df[c]) and not ID_PATTERN.search(str(c))
    ]


def _categorical_columns(df):
    return [
        c
        for c in df.columns
        if (not _is_numeric(df[c])) and not ID_PATTERN.search(str(c))
    ]


def _find_hue(df, exclude=()):
    """First categorical column with a legend-sized number of levels."""
    for col in _categorical_columns(df):
        if col in exclude:
            continue
        n = df[col].nunique(dropna=True)
        if 2 <= n <= MAX_HUE_LEVELS:
            return col
    return None


def _groups(df, hue):
    """Yield ``(label, sub_df)`` pairs, or a single unlabelled group."""
    if hue is None:
        yield None, df
    else:
        for label, sub in df.groupby(hue, observed=True, sort=False):
            yield label, sub


def _apply_spec(ax, spec):
    """Apply the declared axis scales, aspect ratio and inversions of a panel."""
    if not spec:
        return
    for axis in ("x", "y"):
        scale = spec.get(f"{axis}scale")
        if scale:
            getattr(ax, f"set_{axis}scale")(scale)
        if spec.get(f"invert_{axis}"):
            getattr(ax, f"invert_{axis}axis")()
    if spec.get("aspect"):
        ax.set_aspect(spec["aspect"])
    if spec.get("frameon") is False:
        ax.set_axis_off()


def _maybe_log(ax, values, axis, spec=None):
    """Use a log scale when the data are positive and span >2 decades.

    A scale declared for the panel in :data:`PANEL_SPECS` always wins, so a panel drawn
    on log axes in the manuscript keeps them even where the exported range happens to
    be narrow.
    """
    if spec and spec.get(f"{axis}scale"):
        return
    vals = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna())
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    positive = vals[vals > 0]
    if positive.size < 0.99 * vals.size or positive.size == 0:
        return
    if positive.max() / positive.min() > 100:
        getattr(ax, f"set_{axis}scale")("log")


def _monotonic_x_candidates(df, numeric_cols, hue):
    """Numeric columns that increase monotonically within every group."""
    candidates = []
    for col in numeric_cols:
        ok = True
        for _, sub in _groups(df, hue):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().values
            if len(vals) < 3 or not np.all(np.diff(vals) > 0):
                ok = False
                break
        if ok:
            candidates.append(col)
    return candidates


def _finish(fig, sheet_name, source_name, n_rows):
    fig.suptitle(sheet_name, fontsize=11)
    footer = f"{source_name} — {n_rows:,} rows" if source_name else f"{n_rows:,} rows"
    fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize=6, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))


def _text_panel(message, sheet_name, source_name, n_rows):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=9)
    ax.set_axis_off()
    _finish(fig, sheet_name, source_name, n_rows)
    return fig


def _plot_bar_from_bins(df, sheet_name, source_name):
    """Histogram table with explicit bin edges (e.g. Fig 1k)."""
    lo = next(c for c in df.columns if "bin_min" in str(c).lower())
    hi = next(c for c in df.columns if "bin_max" in str(c).lower())
    count_col = next(
        c
        for c in _plottable_numeric_columns(df)
        if c not in (lo, hi) and "center" not in str(c).lower()
    )
    left = pd.to_numeric(df[lo], errors="coerce")
    width = pd.to_numeric(df[hi], errors="coerce") - left
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        left, pd.to_numeric(df[count_col], errors="coerce"), width=width, align="edge"
    )
    ax.set_xlabel(str(lo).replace("_Bin_Min", ""))
    ax.set_ylabel(str(count_col))
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_scatter(df, xcol, ycol, sheet_name, source_name, spec=None):
    hue = _find_hue(df, exclude=(xcol, ycol))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, sub in _groups(df, hue):
        ax.scatter(sub[xcol], sub[ycol], s=3, alpha=0.5, label=str(label))
    ax.set_xlabel(str(xcol))
    ax.set_ylabel(str(ycol))
    _maybe_log(ax, df[xcol], "x", spec)
    _maybe_log(ax, df[ycol], "y", spec)
    _apply_spec(ax, spec)
    if hue is not None:
        ax.legend(title=str(hue), fontsize=6, title_fontsize=6, markerscale=3)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_heatmap(matrix, sheet_name, source_name, n_rows, xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(max(4.5, 0.35 * matrix.shape[1] + 2), 3.8))
    im = ax.imshow(matrix.values.astype(float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([str(i) for i in matrix.index], fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _finish(fig, sheet_name, source_name, n_rows)
    return fig


def _plot_lines(df, xcol, ycols, hue, sheet_name, source_name, spec=None):
    n = len(ycols)
    fig, axes = plt.subplots(n, 1, figsize=(FIGSIZE[0], 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, ycol in zip(axes, ycols):
        for label, sub in _groups(df, hue):
            ax.plot(sub[xcol], sub[ycol], lw=1, label=str(label))
        ax.set_ylabel(str(ycol), fontsize=8)
        _maybe_log(ax, df[ycol], "y", spec)
        _apply_spec(ax, spec)
    _maybe_log(axes[0], df[xcol], "x", spec)
    axes[-1].set_xlabel(str(xcol))
    if hue is not None:
        axes[0].legend(title=str(hue), fontsize=6, title_fontsize=6)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_band(df, xcol, ycols, lower, upper, hue, sheet_name, source_name):
    """A curve (or curves) with a shaded confidence band."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, sub in _groups(df, hue):
        ax.fill_between(
            sub[xcol], sub[lower], sub[upper], alpha=0.25, lw=0, color="0.5"
        )
        for ycol in ycols:
            suffix = f" ({label})" if label is not None else ""
            ax.plot(sub[xcol], sub[ycol], lw=1, label=f"{ycol}{suffix}")
    ax.set_xlabel(str(xcol))
    ax.legend(fontsize=6)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_points_and_means(df, valcol, sheet_name, source_name):
    """Individual observations with the group means drawn on top (e.g. Fig 5d)."""
    group_cols = [
        c
        for c in _categorical_columns(df)
        if c != "Series_Type" and df[c].nunique(dropna=True) <= 30
    ]
    xcol = group_cols[0] if group_cols else None
    huecol = group_cols[1] if len(group_cols) > 1 else None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    categories = list(pd.unique(df[xcol].dropna())) if xcol else [None]
    positions = {c: i for i, c in enumerate(categories)}
    rng = np.random.default_rng(0)
    for series_type, sub in df.groupby("Series_Type", observed=True):
        for label, group in _groups(sub, huecol):
            x = (
                np.array([positions.get(v, np.nan) for v in group[xcol]], dtype=float)
                if xcol
                else np.zeros(len(group))
            )
            if str(series_type).lower().startswith("individ"):
                ax.scatter(
                    x + rng.uniform(-0.15, 0.15, len(x)),
                    group[valcol],
                    s=4,
                    alpha=0.4,
                    label=(
                        f"{label} (individual)" if label is not None else "individual"
                    ),
                )
            else:
                ax.scatter(
                    x,
                    group[valcol],
                    s=30,
                    marker="_",
                    label=f"{label} (mean)" if label is not None else "mean",
                )
    if xcol:
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([str(c) for c in categories], rotation=45, fontsize=6)
        ax.set_xlabel(str(xcol))
    ax.set_ylabel(str(valcol))
    ax.legend(fontsize=5, ncol=2)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_categorical_bar(df, catcol, valcol, sheet_name, source_name):
    fig, ax = plt.subplots(figsize=(max(4.5, 0.25 * len(df) + 2), 3.6))
    ax.bar(np.arange(len(df)), pd.to_numeric(df[valcol], errors="coerce"))
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([str(v) for v in df[catcol]], rotation=90, fontsize=6)
    ax.set_xlabel(str(catcol))
    ax.set_ylabel(str(valcol))
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_hist_grid(df, numeric_cols, hue, sheet_name, source_name):
    n = len(numeric_cols)
    ncols = 1 if n == 1 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(FIGSIZE[0] * ncols * 0.8, 2.4 * nrows)
    )
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for ax, col in zip(axes, numeric_cols):
        vals = pd.to_numeric(df[col], errors="coerce")
        finite = vals[np.isfinite(vals)]
        if finite.empty:
            ax.set_axis_off()
            continue
        bins = np.histogram_bin_edges(finite, bins=min(50, max(10, finite.nunique())))
        for label, sub in _groups(df, hue):
            sub_vals = pd.to_numeric(sub[col], errors="coerce")
            ax.hist(
                sub_vals[np.isfinite(sub_vals)],
                bins=bins,
                histtype="step",
                lw=1,
                label=str(label),
            )
        ax.set_xlabel(str(col), fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=6)
        _maybe_log(ax, finite, "x")
    for ax in axes[n:]:
        ax.set_axis_off()
    if hue is not None:
        axes[0].legend(title=str(hue), fontsize=6, title_fontsize=6)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


# ---------------------------------------------------------------------------
# Figure 1 -- panels whose geometry cannot be inferred from the sheet alone
# ---------------------------------------------------------------------------

#: Height/width ratio of each Figure 1 panel's axes box, keyed by panel letter. Figure 1
#: is laid out at 17.4 x 20.0 cm with `fig.add_axes`, so a panel of width ``w`` and
#: height ``h`` in figure fractions has a box aspect of ``h * 20.0 / (w * 17.4)``. The
#: redrawn panels keep those proportions.
FIG1_BOX_ASPECT = {
    "1d": 1.15,  # [0.13, 0.13]
    "1e": 1.15,
    "1f": 1.07,  # [0.14, 0.13]
    "1g": 1.07,
    "1h": 1.07,
    "1i": 1.07,
    "1j": 1.59,  # [0.13, 0.18]
    "1m": 1.15,  # [0.10, 0.10]
    "1p": 0.72,  # [0.24, 0.15]
}

#: Colours of the library curves, in the order the libraries appear in the sheet, which
#: is the order the notebook passed them to the plotting call. Panels d/e, f/g and h/i
#: draw the same libraries in the same colours.
FIG1_LIBRARY_COLORS = {
    "1d": ("dodgerblue", "darkorange"),
    "1e": ("dodgerblue", "darkorange"),
    "1f": ("darkorchid", "darkorange"),
    "1g": ("darkorchid", "darkorange"),
    "1h": (
        "dodgerblue",
        "darkgreen",
        "brown",
        "teal",
        "darkgrey",
        "violet",
        "darkorchid",
        "darkorange",
    ),
}
FIG1_LIBRARY_COLORS["1i"] = FIG1_LIBRARY_COLORS["1h"]

#: Colours of the three simulated presynaptic-cell numbers of panel j, in sheet order.
FIG1J_COLORS = ("lightsalmon", "tomato", "red")

#: Colour and drawing order of the two AAV-Cre delivery routes of panels o and p.
FIG1_ROUTE_COLORS = {"Intracerebral": "yellowgreen", "Intravenous": "midnightblue"}
FIG1_ROUTE_ZORDER = {"Intracerebral": 1, "Intravenous": 5}


#: Trailing words of a column name that are a unit, and so belong in brackets.
UNIT_WORDS = ("mm", "um", "px", "deg", "AU")


def _axis_label(name):
    """A column name as an axis label (``"Distance_To_Injection_mm"`` -> as the figure).

    Only the leading word keeps its capital; the others are lower-cased unless they are
    not plain Title-case words, which leaves units and acronyms (``mm3``, ``ML``) alone.
    A trailing unit is bracketed, as in the figure.
    """
    words = str(name).replace("_", " ").split()
    words = [
        word if i == 0 or word != word.capitalize() else word.lower()
        for i, word in enumerate(words)
    ]
    if len(words) > 1 and words[-1] in UNIT_WORDS and words[-2] != "per":
        words[-1] = f"({words[-1]})"
    return " ".join(words)


def _decade_ticks(upper, n_max=4):
    """At most ``n_max`` whole-decade ticks from 1 up to ``upper``."""
    decades = int(round(np.log10(upper)))
    step = max(1, int(np.ceil(decades / (n_max - 1))))
    return 10.0 ** np.arange(0, decades + 1, step)


def _fig1_figsize(box_aspect, width=3.4):
    """Figure size fitting an axes box of the given aspect, with room for the labels.

    The box aspect is fixed, so the figure has to be tall enough for it or the panel ends
    up floating in white space: about 1.2 inch of the width goes to the y label and
    ticks, and 1.8 inch of the height to the title, x label and footer.
    """
    return (width, float(np.clip((width - 1.2) * box_aspect + 1.8, 2.6, 6.2)))


def _style_fig1_axes(
    ax,
    xlabel=None,
    ylabel=None,
    xscale="linear",
    yscale="linear",
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    box_aspect=None,
    aspect=None,
):
    """Give a redrawn panel the scales, limits and proportions of the published one."""
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlabel is not None:
        ax.set_xlabel(_axis_label(xlabel), fontsize=9)
    if ylabel is not None:
        ax.set_ylabel(_axis_label(ylabel), fontsize=9)
    if aspect is not None:
        ax.set_aspect(aspect)
    elif box_aspect is not None:
        ax.set_box_aspect(box_aspect)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_fig1_curves(
    df,
    sheet_name,
    source_name,
    xcol,
    ycol,
    group_col=None,
    colors=(),
    drawstyle="default",
    legend=True,
    **axes_options,
):
    """A line panel of Figure 1, on the axes of the published panel.

    Colours are taken in the order the series appear in the sheet, which is the order
    the notebook handed them to the plotting call.
    """
    palette = list(colors)
    fig, ax = plt.subplots(figsize=_fig1_figsize(axes_options.get("box_aspect", 1.0)))
    for i, (label, sub) in enumerate(_groups(df, group_col)):
        ax.plot(
            sub[xcol],
            sub[ycol],
            lw=1.2,
            drawstyle=drawstyle,
            color=palette[i % len(palette)] if palette else None,
            label=str(label),
        )
    _style_fig1_axes(ax, xlabel=xcol, ylabel=ycol, **axes_options)
    if legend and group_col is not None:
        ax.legend(fontsize=6, frameon=False, handlelength=1)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_starter_spread_sim(df, sheet_name, source_name):
    """Fig 1j -- the spread simulation, with its density axis and dashed thresholds."""
    box_aspect = FIG1_BOX_ASPECT["1j"]
    fig, ax = plt.subplots(figsize=_fig1_figsize(box_aspect))
    for i, (n, sub) in enumerate(_groups(df, "Presynaptic_Cells_Per_Starter")):
        ax.plot(
            sub["Starter_Proportion"],
            sub["Probability_Of_Spread"],
            lw=1.2,
            color=FIG1J_COLORS[i % len(FIG1J_COLORS)],
            label=f"{int(n)}",
        )
    for column, line in (
        ("Spread_Probability_Threshold", ax.axhline),
        ("Starter_Proportion_At_Threshold", ax.axvline),
    ):
        if column in df.columns:  # constant columns: the dashed reference lines
            line(float(df[column].iloc[0]), linestyle="dashed", color="black", lw=0.8)

    _style_fig1_axes(
        ax,
        xlabel="Proportion of starter neurons",
        ylabel="Probability of spread\nbetween starter neurons",
        xscale="log",
        yscale="log",
        xlim=(1e-4, 1),
        xticks=[1e-4, 1e-3, 1e-2, 1e-1, 1],
        box_aspect=box_aspect,
    )

    if "Starter_Density_Per_mm3" in df.columns:
        # The second x-axis of the panel; its factor is the ratio the sheet gives.
        factor = float(
            np.median(df["Starter_Density_Per_mm3"] / df["Starter_Proportion"])
        )
        density_axis = ax.secondary_xaxis(
            "top", functions=(lambda v: v * factor, lambda v: v / factor)
        )
        density_axis.set_xscale("log")
        density_axis.set_xlabel(
            "Density of starter neurons (mm$^{-3}$)", fontsize=9, labelpad=2
        )
        density_axis.set_xticks([1e2, 1e3, 1e4, 1e5])
        density_axis.tick_params(axis="both", which="major", labelsize=7)

    ax.legend(
        title="Presynaptic cells\nper starter",
        fontsize=6,
        title_fontsize=6,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        handlelength=1,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_starter_positions(df, sheet_name, source_name):
    """Fig 1o -- starter positions of both delivery routes, on equal axes."""
    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    for route, sub in _groups(df, "Delivery_Route"):
        ax.scatter(
            sub["Mediolateral_mm"],
            sub["Anteroposterior_mm"],
            s=4,
            alpha=0.5,
            linewidths=0,
            color=FIG1_ROUTE_COLORS.get(route, "0.5"),
            zorder=FIG1_ROUTE_ZORDER.get(route, 1),
            label=str(route),
        )
    _style_fig1_axes(
        ax,
        xlabel="ML position (mm)",
        ylabel="AP position (mm)",
        xticks=[-0.4, 0, 0.4],
        yticks=[-0.4, 0, 0.4],
        aspect="equal",
    )
    ax.legend(fontsize=6, frameon=False, markerscale=2)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_pairwise_distances(df, sheet_name, source_name):
    """Fig 1p -- the pairwise-distance densities, each with its median marker."""
    box_aspect = FIG1_BOX_ASPECT["1p"]
    fig, ax = plt.subplots(figsize=_fig1_figsize(box_aspect, width=4.4))
    for route, sub in _groups(df, "Delivery_Route"):
        color = FIG1_ROUTE_COLORS.get(route, "0.5")
        (line,) = ax.plot(
            sub["Pairwise_Distance_mm"],
            sub["Normalised_Cell_Density"],
            lw=1.2,
            color=color,
            label=str(route),
        )
        line.set_clip_on(False)
        if "Median_Pairwise_Distance_mm" in sub.columns:
            ax.plot(
                sub["Median_Pairwise_Distance_mm"].iloc[0],
                1.05,
                marker="v",
                color=color,
                markersize=4,
                clip_on=False,
            )
    _style_fig1_axes(
        ax,
        xlabel="Pairwise distance (mm)",
        ylabel="Normalised cell density",
        xlim=(0, 1),
        ylim=(0, 1),
        xticks=[0, 0.5, 1],
        yticks=[0, 1],
        box_aspect=box_aspect,
    )
    ax.legend(fontsize=6, frameon=False, handlelength=1, loc="upper right")
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _fig1_abundance_draw(panel):
    """The draw callable of the rank-abundance panel ``panel`` (``"1d"``, ...)."""
    return partial(
        _plot_fig1_curves,
        xcol="Barcode_Index",
        ycol="Barcode_Abundance",
        group_col="Library",
        colors=FIG1_LIBRARY_COLORS[panel],
        drawstyle="steps-pre",
        xscale="log",
        yscale="log",
        xlim=(1, 1e8),
        ylim=(0.8, 1e6),
        xticks=np.logspace(0, 8, 9),
        yticks=np.logspace(0, 6, 7),
        box_aspect=FIG1_BOX_ASPECT[panel],
    )


def _fig1_unique_draw(panel, max_cells):
    """The draw callable of the unique-fraction panel ``panel`` (``"1e"``, ...)."""
    return partial(
        _plot_fig1_curves,
        xcol="Number_Of_Infections",
        ycol="Proportion_Uniquely_Labeled",
        group_col="Library",
        colors=FIG1_LIBRARY_COLORS[panel],
        xscale="log",
        xlim=(1, max_cells),
        ylim=(0.5, 1.02),
        # Whole decades, else the tick labels of the log axis are dropped as unlabelled
        # powers.
        xticks=_decade_ticks(max_cells),
        yticks=[0.5, 0.75, 1.0],
        box_aspect=FIG1_BOX_ASPECT[panel],
    )


# ---------------------------------------------------------------------------
# Figure 2 -- panels whose geometry cannot be inferred from the sheet alone
# ---------------------------------------------------------------------------

#: Colour maps of the Fig 2j mosaic, in the plotted gene order.
FIG2J_CMAPS = [
    "GnBu",
    "Oranges",
    "Greys",
    "Blues",
    "Reds",
    "pink_r",
    "Purples",
    "Greens",
    "Wistia",
    "RdPu",
]

#: Coronal window and layout of the Fig 2j mosaic, as drawn by the notebook.
FIG2J_XLIM = (1100, 567)
FIG2J_YLIM = (420, 0)
FIG2J_NCOLS = 5

#: Expression percentile of the non-zero counts mapped to full opacity in Fig 2j.
FIG2J_PERCENTILE = 95

#: Colours of the two groups of Fig 2i, as drawn by `plot_umap_barcoded_cells`.
FIG2I_COLORS = {"Non-barcoded": "lightgrey", "Barcoded": "black"}
FIG2I_SIZES = {"Non-barcoded": 1.0, "Barcoded": 0.5}


def _categorical_palette(n):
    """`n` distinct colours, in the order scanpy assigns cluster colours."""
    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return [base[i % len(base)] for i in range(n)]


def _plot_umap_clusters(df, sheet_name, source_name):
    """Fig 2h -- the UMAP coloured by cluster, with the labels on the data."""
    import matplotlib.patheffects as path_effects

    clusters = list(pd.unique(df["Cluster"].dropna()))
    colors = dict(zip(clusters, _categorical_palette(len(clusters))))

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    for cluster, sub in df.groupby("Cluster", observed=True, sort=False):
        ax.scatter(
            sub["UMAP_1"],
            sub["UMAP_2"],
            s=1,
            alpha=0.5,
            linewidths=0,
            color=colors.get(cluster, "0.5"),
            rasterized=True,
        )
        label = ax.text(
            sub["UMAP_1"].median(),
            sub["UMAP_2"].median(),
            str(cluster),
            ha="center",
            va="center",
            fontsize=5,
            zorder=5,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=1, foreground="white")]
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_umap_barcoded(df, sheet_name, source_name):
    """Fig 2i -- the same UMAP, barcoded cells in black over the rest in grey."""
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    # Drawn in the order of the figure: the grey background first.
    for group in ("Non-barcoded", "Barcoded"):
        sub = df[df["Cell_Group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["UMAP_1"],
            sub["UMAP_2"],
            s=FIG2I_SIZES.get(group, 1.0),
            linewidths=0,
            color=FIG2I_COLORS.get(group, "0.5"),
            label=f"{group} cells",
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend(loc="lower right", fontsize=6, frameon=False, markerscale=6)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_gene_expression_mosaic(df, sheet_name, source_name):
    """Fig 2j -- one coronal section per gene, coloured and faded by expression."""
    genes = [c for c in df.columns if c not in ("ARA_Z_px", "ARA_Y_px")]
    ncols = min(FIG2J_NCOLS, len(genes))
    nrows = int(np.ceil(len(genes) / ncols))

    x = pd.to_numeric(df["ARA_Z_px"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["ARA_Y_px"], errors="coerce").to_numpy()

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            2.1 * ncols,
            # the coronal window is drawn with an equal aspect ratio
            2.1
            * nrows
            * abs(FIG2J_YLIM[0] - FIG2J_YLIM[1])
            / abs(FIG2J_XLIM[0] - FIG2J_XLIM[1])
            + 0.4,
        ),
    )
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for ax, gene, cmap_name in zip(axes, genes, FIG2J_CMAPS * len(genes)):
        expression = pd.to_numeric(df[gene], errors="coerce").to_numpy(dtype=float)
        nonzero = expression[expression > 0]
        vmax = max(
            np.percentile(nonzero, FIG2J_PERCENTILE) if nonzero.size else 1.0, 1e-9
        )
        rgba = plt.get_cmap(cmap_name)(mcolors.Normalize(vmin=0, vmax=vmax)(expression))
        rgba[:, 3] = np.clip(expression / vmax, 0, 1)  # zero-count cells invisible
        order = np.argsort(expression)  # faint cells first, as in the figure
        ax.scatter(
            x[order], y[order], s=0.3, c=rgba[order], linewidths=0, rasterized=True
        )
        ax.set_title(gene, fontsize=7, fontstyle="italic", pad=1.5)
        ax.set_xlim(*FIG2J_XLIM)
        ax.set_ylim(*FIG2J_YLIM)
        ax.set_aspect("equal")
        ax.set_axis_off()
    for ax in axes[len(genes) :]:
        ax.set_axis_off()
    _finish(fig, sheet_name, source_name, len(df))
    return fig


# ---------------------------------------------------------------------------
# Per-panel drawing specifications
# ---------------------------------------------------------------------------

#: What the numbers of a sheet cannot say about how its panel is drawn, matched on a
#: lower-case substring of the sheet name (first match wins, so specific keys go
#: first). ``draw`` replaces the inferred plot entirely; the other keys refine it:
#: ``xscale``/``yscale``, ``aspect``, ``invert_x``/``invert_y`` and ``frameon``.
PANEL_SPECS = [
    # Figure 1, whose panels are drawn on the axes of the published figure.
    ("1d library abundance", {"kind": "line", "draw": _fig1_abundance_draw("1d")}),
    ("1e unique fraction", {"kind": "line", "draw": _fig1_unique_draw("1e", 1e6)}),
    (
        "1f rescue scaling abundance",
        {"kind": "line", "draw": _fig1_abundance_draw("1f")},
    ),
    (
        "1g rescue scaling unique",
        {"kind": "line", "draw": _fig1_unique_draw("1g", 1e4)},
    ),
    (
        "1h library comparison abund",
        {"kind": "line", "draw": _fig1_abundance_draw("1h")},
    ),
    (
        "1i library compar unique",
        {"kind": "line", "draw": _fig1_unique_draw("1i", 1e6)},
    ),
    ("1j starter spread", {"kind": "line", "draw": _plot_starter_spread_sim}),
    (
        "1m presynaptic density",
        {
            "kind": "line",
            "draw": partial(
                _plot_fig1_curves,
                xcol="Distance_To_Injection_mm",
                ycol="Cell_Density_Per_mm3",
                colors=("black",),
                legend=False,
                xlim=(0, 2),
                ylim=(0, None),
                box_aspect=FIG1_BOX_ASPECT["1m"],
            ),
        },
    ),
    ("1o starter positions", {"kind": "scatter", "draw": _plot_starter_positions}),
    ("1p pairwise distances", {"kind": "line", "draw": _plot_pairwise_distances}),
    ("umap by cluster", {"kind": "scatter", "draw": _plot_umap_clusters}),
    ("umap barcoded cells", {"kind": "scatter", "draw": _plot_umap_barcoded}),
    ("gene expression map", {"kind": "mosaic", "draw": _plot_gene_expression_mosaic}),
]

#: The same, matched on the columns a sheet holds rather than on its name, for
#: quantities that appear in several figures under different sheet names. A sheet
#: matches when it holds every listed column.
COLUMN_SPECS = [
    # Rank-abundance curves are drawn on log-log axes (Fig 1d, f, h; Supp 4a).
    (("Barcode_Index", "Barcode_Abundance"), {"xscale": "log", "yscale": "log"}),
    # Unique-labelling curves are drawn against a logarithmic number of cells.
    (("Number_of_Labeled_Cells", "Fraction_Uniquely_Labeled"), {"xscale": "log"}),
]


def _spec_for(sheet_name, df=None):
    """The :data:`PANEL_SPECS` or :data:`COLUMN_SPECS` entry of a sheet, or ``None``."""
    name = str(sheet_name).lower()
    for key, spec in PANEL_SPECS:
        if key in name:
            return spec
    if df is not None:
        for columns, spec in COLUMN_SPECS:
            if all(column in df.columns for column in columns):
                return spec
    return None


def plot_sheet(df, sheet_name, source_name=None):
    """Draw one sheet, choosing the plot kind from its columns.

    Args:
        df (pandas.DataFrame): Sheet contents, as returned by
            :func:`~brisc.source_data.io.read_source_data_workbook`.
        sheet_name (str): Name of the sheet, used as the figure title.
        source_name (str, optional): Workbook name, shown in the figure footer.

    Returns:
        tuple: ``(matplotlib.figure.Figure, kind)`` where ``kind`` is one of
        ``"bar"``, ``"scatter"``, ``"heatmap"``, ``"line"``, ``"hist"`` or
        ``"empty"``.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return _text_panel("Sheet is empty", sheet_name, source_name, 0), "empty"

    spec = _spec_for(sheet_name, df)
    if spec is not None and "draw" in spec:
        return spec["draw"](df, sheet_name, source_name), spec.get("kind", "custom")

    numeric = _plottable_numeric_columns(df)
    categorical = _categorical_columns(df)
    names = [str(c).lower() for c in df.columns]

    # 1) Histogram table with explicit bin edges
    if any("bin_min" in n for n in names) and any("bin_max" in n for n in names):
        return _plot_bar_from_bins(df, sheet_name, source_name), "bar"

    # 2) Known 2D coordinate pairs
    for xcol, ycol in COORD_PAIRS:
        if xcol in df.columns and ycol in df.columns:
            return (
                _plot_scatter(df, xcol, ycol, sheet_name, source_name, spec),
                "scatter",
            )
    for col in df.columns:
        if str(col).endswith("_x") and f"{str(col)[:-2]}_y" in df.columns:
            ycol = f"{str(col)[:-2]}_y"
            if _is_numeric(df[col]) and _is_numeric(df[ycol]):
                return (
                    _plot_scatter(df, col, ycol, sheet_name, source_name, spec),
                    "scatter",
                )

    # 3) Long (category, category, value) table -> pivot to a heatmap
    if len(categorical) == 2 and len(numeric) == 1:
        matrix = df.pivot_table(
            index=categorical[0], columns=categorical[1], values=numeric[0]
        )
        return (
            _plot_heatmap(
                matrix,
                sheet_name,
                source_name,
                len(df),
                xlabel=str(categorical[1]),
                ylabel=str(categorical[0]),
            ),
            "heatmap",
        )

    hue = _find_hue(df)

    # 3b) Curve with a confidence band
    lower = next((c for c in df.columns if _CI_LOWER.search(str(c))), None)
    upper = next((c for c in df.columns if _CI_UPPER.search(str(c))), None)
    if lower is not None and upper is not None and len(numeric) >= 3:
        rest = [c for c in numeric if c not in (lower, upper)]
        band_x = _monotonic_x_candidates(df, rest, hue)
        if band_x:
            xcol = band_x[0]
            ycols = [c for c in rest if c != xcol]
            return (
                _plot_band(df, xcol, ycols, lower, upper, hue, sheet_name, source_name),
                "band",
            )

    # 3c) Individual points with the group mean drawn on top
    if "Series_Type" in df.columns and len(numeric) >= 1:
        return (
            _plot_points_and_means(df, numeric[-1], sheet_name, source_name),
            "points",
        )

    monotonic = _monotonic_x_candidates(df, numeric, hue)

    # 4) Wide matrix: label column followed by many numeric columns
    if not monotonic and len(numeric) >= 3 and not _is_numeric(df[df.columns[0]]):
        matrix = df.set_index(df.columns[0])[numeric]
        return (
            _plot_heatmap(
                matrix, sheet_name, source_name, len(df), ylabel=str(df.columns[0])
            ),
            "heatmap",
        )

    # 5) Monotonic x -> one line panel per remaining numeric column
    if monotonic:
        xcol = monotonic[0]
        ycols = [c for c in numeric if c != xcol]
        if ycols:
            return (
                _plot_lines(df, xcol, ycols, hue, sheet_name, source_name, spec),
                "line",
            )

    # 6) Small aggregate table -> bar
    if len(numeric) == 1 and len(categorical) == 1 and len(df) <= 50:
        return (
            _plot_categorical_bar(
                df, categorical[0], numeric[0], sheet_name, source_name
            ),
            "bar",
        )

    # 7) Two free numeric columns -> scatter
    if len(numeric) == 2:
        return (
            _plot_scatter(df, numeric[0], numeric[1], sheet_name, source_name, spec),
            "scatter",
        )

    # 8) Anything else with numbers -> histogram grid
    if numeric:
        return _plot_hist_grid(df, numeric, hue, sheet_name, source_name), "hist"

    # 9) Nothing plottable
    msg = "No numeric column to plot\ncolumns: " + ", ".join(str(c) for c in df.columns)
    return _text_panel(msg, sheet_name, source_name, len(df)), "empty"


def slugify(name):
    """Turn a sheet name into a safe file stem (``"Fig 1d"`` -> ``"Fig_1d"``)."""
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    return slug or "sheet"


def plot_workbook(xlsx_path, output_dir, dpi=200, verbose=True):
    """Render every sheet of one workbook to a PNG.

    Args:
        xlsx_path (str or Path): Source Data workbook.
        output_dir (str or Path): Directory for the PNGs (created if missing).
        dpi (int): Resolution of the saved figures.
        verbose (bool): Print one line per sheet.

    Returns:
        list: Paths of the written PNGs.
    """
    xlsx_path = Path(xlsx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sheets = read_source_data_workbook(xlsx_path)
    written = []
    for sheet_name, df in sheets.items():
        target = output_dir / f"{slugify(sheet_name)}.png"
        try:
            fig, kind = plot_sheet(df, sheet_name, source_name=xlsx_path.name)
        except Exception as e:  # a bad sheet must not abort the workbook
            print(f"  [!] {sheet_name}: could not plot ({type(e).__name__}: {e})")
            continue
        try:
            fig.savefig(target, dpi=dpi)
        finally:
            plt.close(fig)
        written.append(target)
        if verbose:
            n_rows = 0 if df is None else len(df)
            print(
                f"  {output_dir.name}/{target.name}  [{kind}, {n_rows:,} rows, "
                f"{0 if df is None else len(df.columns)} cols]"
            )
    return written


def plot_all_workbooks(
    source_data_dir, output_dir=None, dpi=200, pattern="Source_Data_*.xlsx"
):
    """Render every workbook in a Source Data directory.

    Args:
        source_data_dir (str or Path): Directory holding the ``.xlsx`` files.
        output_dir (str or Path, optional): Root for the panels. Defaults to
            ``source_data_dir / "panels"``.
        dpi (int): Resolution of the saved figures.
        pattern (str): Glob used to find the workbooks.

    Returns:
        dict: Mapping of workbook path to the list of PNGs written for it.
    """
    source_data_dir = Path(source_data_dir)
    output_dir = Path(output_dir) if output_dir else source_data_dir / "panels"

    workbooks = sorted(source_data_dir.glob(pattern))
    if not workbooks:
        print(f"[Panels] No workbook matching {pattern!r} in {source_data_dir}")
        return {}

    results = {}
    for xlsx_path in workbooks:
        stem = xlsx_path.stem
        if stem.startswith("Source_Data_"):
            stem = stem[len("Source_Data_") :]
        print(f"\n--- {xlsx_path.name} ---")
        results[xlsx_path] = plot_workbook(xlsx_path, output_dir / stem, dpi=dpi)
    return results
