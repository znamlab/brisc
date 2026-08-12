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
# Panel geometry shared by the redrawn figures
# ---------------------------------------------------------------------------

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


def _panel_figsize(box_aspect, width=3.4):
    """Figure size fitting an axes box of the given aspect, with room for the labels.

    The box aspect is fixed, so the figure has to be tall enough for it or the panel
    ends up floating in white space: about 1.2 inch of the width goes to the y label and
    ticks, and 1.8 inch of the height to the title, x label and footer.
    """
    return (width, float(np.clip((width - 1.2) * box_aspect + 1.8, 2.6, 6.2)))


def _style_panel_axes(
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


def _plot_panel_curves(
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
    """A one-line-per-series panel, on the axes of the published panel.

    ``colors`` is either a mapping of series label to colour, which is what a panel
    whose sheet holds a subset of the drawn series needs, or a sequence taken in the
    order the series appear in the sheet — the order the notebook passed them in.
    """
    by_name = isinstance(colors, dict)
    palette = dict(colors) if by_name else list(colors)
    fig, ax = plt.subplots(figsize=_panel_figsize(axes_options.get("box_aspect", 1.0)))
    for i, (label, sub) in enumerate(_groups(df, group_col)):
        if by_name:
            color = palette.get(label)
        else:
            color = palette[i % len(palette)] if palette else None
        ax.plot(
            sub[xcol],
            sub[ycol],
            lw=1.2,
            drawstyle=drawstyle,
            color=color,
            label=str(label),
        )
    _style_panel_axes(ax, xlabel=xcol, ylabel=ycol, **axes_options)
    if legend and group_col is not None:
        ax.legend(fontsize=6, frameon=False, handlelength=1)
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

#: Colour of every library curve, by the library label the sheet carries. Keyed by name
#: rather than by position because the workbook holds only the libraries made in this
#: study: panels h and i also draw published libraries, which are not tabulated, so
#: their colours are listed here only to keep ours on the right hue. Panels d/e, f/g
#: and h/i draw the same libraries in the same colours.
FIG1_LIBRARY_COLORS = {
    "1d": {"Plasmid library": "dodgerblue", "Virus library": "darkorange"},
    "1f": {"2 wells": "darkorchid", "12 wells": "darkorange"},
    "1h": {
        "Clark, 2021": "dodgerblue",
        "Saunders, 2022": "darkgreen",
        "Zhang, 2024": "brown",
        "Tan, 2025 N2c": "teal",
        "Shin, 2024 N2c": "darkgrey",
        "Shin, 2024 SADB19": "violet",
        "RV2": "darkorchid",
        "RV35": "darkorange",
    },
}
FIG1_LIBRARY_COLORS["1e"] = FIG1_LIBRARY_COLORS["1d"]
FIG1_LIBRARY_COLORS["1g"] = FIG1_LIBRARY_COLORS["1f"]
FIG1_LIBRARY_COLORS["1i"] = FIG1_LIBRARY_COLORS["1h"]

#: Colours of the three simulated presynaptic-cell numbers of panel j, in sheet order.
FIG1J_COLORS = ("lightsalmon", "tomato", "red")

#: Colour and drawing order of the two AAV-Cre delivery routes of panels o and p.
FIG1_ROUTE_COLORS = {"Intracerebral": "yellowgreen", "Intravenous": "midnightblue"}
FIG1_ROUTE_ZORDER = {"Intracerebral": 1, "Intravenous": 5}


def _plot_starter_spread_sim(df, sheet_name, source_name):
    """Fig 1j -- the spread simulation, with its density axis and dashed thresholds."""
    box_aspect = FIG1_BOX_ASPECT["1j"]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect))
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

    _style_panel_axes(
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
    _style_panel_axes(
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
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect, width=4.4))
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
    _style_panel_axes(
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


def _abundance_draw(colors, box_aspect):
    """The draw callable of a rank-abundance panel, on the axes Figure 1 established.

    Supplementary Figure 4a draws the same quantity on the same axes, with its own
    colours and proportions, which is why the two are arguments rather than a panel
    letter looked up in :data:`FIG1_LIBRARY_COLORS` and :data:`FIG1_BOX_ASPECT`.
    """
    return partial(
        _plot_panel_curves,
        xcol="Barcode_Index",
        ycol="Barcode_Abundance",
        group_col="Library",
        colors=colors,
        drawstyle="steps-pre",
        xscale="log",
        yscale="log",
        xlim=(1, 1e8),
        ylim=(0.8, 1e6),
        xticks=np.logspace(0, 8, 9),
        yticks=np.logspace(0, 6, 7),
        box_aspect=box_aspect,
    )


def _unique_draw(colors, box_aspect, max_cells):
    """The draw callable of a unique-fraction panel, on the axes Figure 1 established.

    Shared with Supplementary Figure 4b; see :func:`_abundance_draw` for why the colours
    and proportions are passed in.
    """
    return partial(
        _plot_panel_curves,
        xcol="Number_Of_Infections",
        ycol="Proportion_Uniquely_Labeled",
        group_col="Library",
        colors=colors,
        xscale="log",
        xlim=(1, max_cells),
        ylim=(0.5, 1.02),
        # Whole decades, else the tick labels of the log axis are dropped as unlabelled
        # powers.
        xticks=_decade_ticks(max_cells),
        yticks=[0.5, 0.75, 1.0],
        box_aspect=box_aspect,
    )


def _fig1_abundance(panel):
    """The rank-abundance draw callable of Figure 1 ``panel`` (``"1d"``, ...)."""
    return _abundance_draw(FIG1_LIBRARY_COLORS[panel], FIG1_BOX_ASPECT[panel])


def _fig1_unique(panel, max_cells):
    """The unique-fraction draw callable of Figure 1 ``panel`` (``"1e"``, ...)."""
    return _unique_draw(FIG1_LIBRARY_COLORS[panel], FIG1_BOX_ASPECT[panel], max_cells)


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
    # `tight_layout` leaves the rows touching, because an axis-off panel reports no
    # height for its title: open a gap so a gene name cannot land on the row above it.
    if nrows > 1:
        fig.subplots_adjust(hspace=0.12)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 -- panels whose geometry cannot be inferred from the sheet alone
# ---------------------------------------------------------------------------

#: Height/width ratio of each Figure 3 panel's axes box, keyed by panel letter. Figure 3
#: is laid out at 17.4 x 17.4 cm with `fig.add_axes`, so a panel of width ``w`` and
#: height ``h`` in figure fractions has a box aspect of ``h / w``. The redrawn panels
#: keep those proportions; panel b is the two axes of its broken y axis.
FIG3_BOX_ASPECT = {
    "3a": 0.62,  # [0.13, 0.08], one per cell population
    "3b": 0.61,  # [0.23, 0.14]
    "3b_top": 0.15,  # [0.23, 0.14 / 4]
    "3c": 0.91,  # [0.22, 0.20]
    "3d": 1.20,  # [0.15, 0.18]
    "3e": 1.20,  # [0.15, 0.18]
    "3f": 1.00,  # [0.18, 0.18]
}

#: Fill and edge colour of the barcode-count histograms of panels a, c and e.
FIG3_HIST_FACECOLOR = "slategray"
FIG3_HIST_EDGECOLOR = "black"

#: Colour of each series of panel b, and the line style telling the two barcode sets
#: apart, as drawn by `match_to_library.plot_matches_to_library`.
FIG3B_STYLE = {
    "Library_Read_Proportion": ("black", "-", "Viral library barcodes"),
    "In_Situ_Barcode_Proportion": ("dodgerblue", "-", "In situ barcodes"),
    "Random_Barcode_Proportion": ("dodgerblue", (0, (2, 1)), "Random barcodes"),
}

#: The bar of barcodes absent from the library sits here, left of the log axis, and its
#: centre carries the "0" tick.
FIG3B_ZERO_BAR = (0.03, 0.06)

#: The two y ranges of the broken y axis of panel b, bottom first, and their ticks.
FIG3B_YLIMS = ((0, 0.2), (0.6, 0.65))
FIG3B_YTICKS = ((0, 0.1, 0.2), (0.6, 0.65))

#: Proportions of unique library reads the x axis of panel b is labelled with.
FIG3B_XTICK_PROPORTIONS = (1e-8, 1e-5, 1e-2)

#: Colour of the two barcode types of panel d, and the bar holding the barcodes with no
#: presynaptic cell, drawn left of the log axis.
FIG3D_COLORS = {"Orphan barcodes": "darkorange", "Non-orphan barcodes": "dodgerblue"}
FIG3D_ZERO_BAR = (0.48, 0.69)
FIG3D_ALPHA = 0.5

#: Colour of the starter cells and of the robust fit of panel f.
FIG3F_COLOR = "darkslategray"


def _fig3_stairs(ax, values, proportions, counts=None, y_offset=0.05):
    """Draw one `barcodes_in_cells.plot_hist` histogram back onto ``ax``.

    Integer values are drawn as bars covering ``value +/- 0.5``; the counts, when the
    sheet holds them, are the numbers annotating the bars of the published panel.
    """
    values = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    proportions = np.asarray(pd.to_numeric(proportions, errors="coerce"), dtype=float)
    ax.stairs(
        proportions,
        np.append(values - 0.5, values[-1] + 0.5),
        fill=True,
        edgecolor=FIG3_HIST_EDGECOLOR,
        facecolor=FIG3_HIST_FACECOLOR,
        linewidth=0.5,
    )
    if counts is None:
        return
    for value, proportion, count in zip(values, proportions, np.asarray(counts)):
        ax.text(
            value - 0.2,
            proportion + y_offset,
            f"{int(count)}",
            ha="left",
            fontsize=5,
            color="black",
            alpha=0.8,
            rotation=35,
        )


def _plot_fig3_barcodes_per_cell(df, sheet_name, source_name):
    """Fig 3a -- barcodes per cell, presynaptic cells above starter cells."""
    box_aspect = FIG3_BOX_ASPECT["3a"]
    populations = list(pd.unique(df["Cell_Population"]))
    width = 3.4
    fig, axes = plt.subplots(
        len(populations),
        1,
        figsize=(width, (width - 1.2) * box_aspect * len(populations) + 1.8),
        sharex=True,
    )
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    values = pd.to_numeric(df["Barcodes_Per_Cell"], errors="coerce")
    for ax, population in zip(axes, populations):
        sub = df[df["Cell_Population"] == population]
        _fig3_stairs(
            ax,
            sub["Barcodes_Per_Cell"],
            sub["Proportion_Of_Barcodes"],
            sub["Cell_Count"] if "Cell_Count" in sub.columns else None,
        )
        _style_panel_axes(
            ax,
            ylabel="Proportion_Of_Barcodes",
            xlim=(values.min() - 0.5, values.max() + 0.5),
            ylim=(0, 1),
            xticks=np.arange(int(values.min()), int(values.max()) + 1),
            yticks=[0, 0.5, 1.0],
            box_aspect=box_aspect,
        )
        ax.text(
            values.max() + 0.5,
            1,
            str(population).replace(" ", "\n"),
            ha="right",
            va="top",
            fontsize=6,
        )
    axes[-1].set_xlabel(_axis_label("Barcodes_Per_Cell"), fontsize=9)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_fig3_match_to_library(df, sheet_name, source_name):
    """Fig 3b -- library reads per barcode, on the broken y axis of the panel.

    The three histograms are drawn on both halves of the broken axis, as the notebook
    does, so a bar taller than the lower range is still readable.
    """
    edges = np.append(
        pd.to_numeric(df["Bin_Min_Reads"], errors="coerce").to_numpy()[1:],
        pd.to_numeric(df["Bin_Max_Reads"], errors="coerce").to_numpy()[-1],
    )
    box_aspects = (FIG3_BOX_ASPECT["3b_top"], FIG3_BOX_ASPECT["3b"])
    width = 4.0
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(width, (width - 1.2) * sum(box_aspects) + 1.8),
        sharex=True,
        gridspec_kw=dict(height_ratios=list(box_aspects)),
    )
    for ax, box_aspect, ylim, yticks in zip(
        axes, box_aspects, FIG3B_YLIMS[::-1], FIG3B_YTICKS[::-1]
    ):
        for column, (color, linestyle, label) in FIG3B_STYLE.items():
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce").to_numpy()
            ax.stairs(
                values[1:],
                edges,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                fill=False,
                label=label,
            )
            if np.isfinite(values[0]):  # barcodes absent from the library
                ax.stairs(
                    [values[0]],
                    list(FIG3B_ZERO_BAR),
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.0,
                    fill=False,
                )
        _style_panel_axes(
            ax,
            xscale="log",
            xlim=(FIG3B_ZERO_BAR[0] * 0.6, edges[-1] * 1.5),
            ylim=ylim,
            yticks=list(yticks),
            box_aspect=box_aspect,
        )

    total_reads = float(
        pd.to_numeric(df["Library_Total_Reads"], errors="coerce").iloc[0]
    )
    ticks = [np.sqrt(FIG3B_ZERO_BAR[0] * FIG3B_ZERO_BAR[1])]
    ticks += [proportion * total_reads for proportion in FIG3B_XTICK_PROPORTIONS]
    labels = ["$0$"] + [
        f"$10^{{{int(np.log10(proportion))}}}$"
        for proportion in FIG3B_XTICK_PROPORTIONS
    ]
    axes[1].set_xticks(ticks, labels=labels)
    axes[1].set_xlabel(
        "Proportion of unique reads in viral library per barcode", fontsize=9
    )
    axes[1].set_ylabel("Proportion of barcodes / of unique reads", fontsize=9)
    axes[0].spines["bottom"].set_visible(False)
    axes[0].tick_params(axis="x", which="both", bottom=False)
    axes[0].legend(loc="upper right", fontsize=6, frameon=False)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_fig3_starters_per_barcode(df, sheet_name, source_name):
    """Fig 3c -- starter cells per barcode, with the counts above the bars."""
    box_aspect = FIG3_BOX_ASPECT["3c"]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect))
    values = pd.to_numeric(df["Starters_Per_Barcode"], errors="coerce")
    _fig3_stairs(
        ax,
        values,
        df["Proportion_Of_Barcodes"],
        df["Barcode_Count"] if "Barcode_Count" in df.columns else None,
    )
    _style_panel_axes(
        ax,
        xlabel="Starters_Per_Barcode",
        ylabel="Proportion_Of_Barcodes",
        xlim=(values.min() - 0.5, values.max() + 0.5),
        ylim=(0, 1),
        xticks=np.arange(int(values.min()), int(values.max()) + 1),
        box_aspect=box_aspect,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_fig3_presyn_per_barcode(df, sheet_name, source_name):
    """Fig 3d -- presynaptic cells per barcode, orphan against non-orphan barcodes."""
    box_aspect = FIG3_BOX_ASPECT["3d"]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect))
    upper = 1.0
    for barcode_type, sub in _groups(df, "Barcode_Type"):
        color = FIG3D_COLORS.get(barcode_type, "0.5")
        proportions = pd.to_numeric(
            sub["Proportion_Of_Barcodes"], errors="coerce"
        ).to_numpy()
        edges = np.append(
            pd.to_numeric(sub["Bin_Min"], errors="coerce").to_numpy()[1:],
            pd.to_numeric(sub["Bin_Max"], errors="coerce").to_numpy()[-1],
        )
        upper = max(upper, edges[-1])
        for fill, alpha, lw in ((True, FIG3D_ALPHA, 0), (False, 1, 1.0)):
            ax.stairs(
                proportions[1:],
                edges,
                fill=fill,
                color=color,
                lw=lw,
                alpha=alpha,
                label=str(barcode_type) if fill else None,
            )
            # the barcodes with no presynaptic cell, drawn left of the log axis
            ax.stairs(
                [proportions[0]],
                list(FIG3D_ZERO_BAR),
                fill=fill,
                color=color,
                lw=max(lw, 0.5),
                alpha=alpha,
            )
    _style_panel_axes(
        ax,
        xlabel="Presynaptic cells per barcode",
        ylabel="Proportion_Of_Barcodes",
        xscale="log",
        xlim=(FIG3D_ZERO_BAR[0], upper),
        box_aspect=box_aspect,
    )
    ax.set_xticks(
        [np.sqrt(FIG3D_ZERO_BAR[0] * FIG3D_ZERO_BAR[1]), 1, 10, 100],
        labels=["0", "1", "10", "100"],
    )
    ax.legend(fontsize=6, frameon=False, handlelength=1, loc="upper right")
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_fig3_spots_per_cell(df, sheet_name, source_name):
    """Fig 3e -- barcode spots per cell, with the dotted detection threshold."""
    box_aspect = FIG3_BOX_ASPECT["3e"]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect))
    values = pd.to_numeric(df["Barcode_Spots_Per_Cell"], errors="coerce")
    _fig3_stairs(ax, values, df["Proportion_Of_Cells"])
    if "Min_Spots_Threshold" in df.columns:  # a constant column: the dotted line
        ax.axvline(
            float(df["Min_Spots_Threshold"].iloc[0]),
            color="k",
            linestyle="dotted",
            lw=1,
        )
    _style_panel_axes(
        ax,
        xlabel="Barcode_Spots_Per_Cell",
        ylabel="Proportion_Of_Cells",
        xlim=(values.min() - 0.5, values.max() + 0.5),
        ylim=(0, 0.08),
        xticks=np.arange(0, values.max() + 10, 10),
        box_aspect=box_aspect,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_fig3_mcherry(df, sheet_name, source_name):
    """Fig 3f -- presynaptic cells against starter mCherry fluorescence, and its fit.

    Both axes hold natural logarithms, which the panel labels with the fluorescence and
    the cell numbers themselves.
    """
    box_aspect = FIG3_BOX_ASPECT["3f"]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect))
    cells = df[df["Series_Type"] == "Individual starter cell"]
    ax.scatter(
        cells["Log_mCherry_Fluorescence"],
        cells["Log_Presynaptic_Cells"],
        s=3,
        color=FIG3F_COLOR,
        edgecolor="black",
        linewidths=0.2,
        alpha=0.5,
    )
    fit = df[df["Series_Type"] == "Robust fit"]
    if not fit.empty:
        if {"Fit_CI_Lower", "Fit_CI_Upper"} <= set(fit.columns):
            ax.fill_between(
                fit["Log_mCherry_Fluorescence"],
                fit["Fit_CI_Lower"],
                fit["Fit_CI_Upper"],
                color=FIG3F_COLOR,
                alpha=0.15,
                lw=0,
            )
        ax.plot(
            fit["Log_mCherry_Fluorescence"],
            fit["Log_Presynaptic_Cells"],
            color=FIG3F_COLOR,
            lw=2,
        )
    _style_panel_axes(
        ax,
        xlabel="Starter mCherry fluorescence (AU)",
        ylabel="Number of presynaptic cells + 1",
        box_aspect=box_aspect,
    )
    ax.set_xticks(np.log([100, 1000]), labels=[100, 1000])
    ax.set_yticks(np.log([1, 10, 100]), labels=[1, 10, 100])
    _finish(fig, sheet_name, source_name, len(df))
    return fig


# ---------------------------------------------------------------------------
# Long-range panels -- Figure 6 and the supplementary reviewer figure
# ---------------------------------------------------------------------------

# The two long-range figures draw the same six panels of different quantities: Figure 6
# along the medio-lateral axis and receptive-field azimuth, the supplementary reviewer
# figure along the antero-posterior axis and elevation. Everything that differs between
# them lives in LONG_RANGE_STYLES, and the value column is resolved from the sheet, so
# the drawing functions below serve both.

#: Per-figure style of the long-range panels, as the two notebooks draw them.
#: ``windows`` is the flatmap window of panels b, c and d, ``(xlim, ylim)`` in flatmap
#: pixels and in the order the notebook sets them: both axes are inverted, so it keeps
#: the orientation of the published one, and panel b is the zoomed inset around V1.
#: ``box_aspect`` is the height/width ratio of the two graph panels' axes boxes; both
#: figures are laid out at 8.8 x 20.0 cm with `fig.add_axes`, so a panel of width ``w``
#: and height ``h`` in figure fractions has a box aspect of ``h * 20.0 / (w * 8.8)``.
#: ``retinotopy`` is the axis of panel f. The two figures share their flatmap windows,
#: proportions, presynaptic-position axis and curve colour; they differ in colour map,
#: value range, what the value is and which retinotopic quantity panel f averages.
LONG_RANGE_STYLES = {
    "fig6": dict(
        cmap="turbo_r",
        clim=(-1.0, 1.0),
        value_label="Starter ML position (mm)",
        axis_label="Presynaptic ML position (mm)",
        axis_lim=(-4.5, 4.5),
        axis_ticks=(-4, 0, 4),
        windows={
            "b": ((800, 400), (1200, 950)),
            "c": ((1050, 150), (1330, 810)),
            "d": ((1050, 150), (1330, 810)),
        },
        box_aspect={
            "e": 0.57,  # [0.8, 0.20]
            "f": 0.26,  # [0.8, 0.09]
        },
        curve_color="darkorchid",
        retinotopy=dict(
            label="Receptive field azimuth (degrees)",
            lim=(0, 60),
            ticks=(0, 30, 60),
        ),
    ),
    "reviewer": dict(
        cmap="turbo",
        clim=(8.5, 8.9),
        value_label="Starter AP position (mm)",
        axis_label="Presynaptic ML position (mm)",
        axis_lim=(-4.5, 4.5),
        axis_ticks=(-4, 0, 4),
        windows={
            "b": ((800, 400), (1200, 950)),
            "c": ((1050, 150), (1330, 810)),
            "d": ((1050, 150), (1330, 810)),
        },
        box_aspect={
            "e": 0.57,  # [0.8, 0.20]
            "f": 0.26,  # [0.8, 0.09]
        },
        curve_color="darkorchid",
        retinotopy=dict(
            label="Receptive field elevation (degrees)",
            lim=(-20, 20),
            ticks=(-20, 0, 20),
        ),
    ),
}

#: Name of the presynaptic-position column of the two graph panels, in both figures. The
#: quantity is the same in both: the medio-lateral flatmap position of the presynaptic
#: cells, in mm, which is what the panels label their x axis with.
LONG_RANGE_AXIS_COLUMN = "Presynaptic_ML_mm"

#: Flatmap coordinate columns, which every other numeric column of a flatmap sheet is
#: measured against.
LONG_RANGE_FLATMAP_COLUMNS = ("Flatmap_X", "Flatmap_Y")


def _long_range_value_column(df, exclude=()):
    """The value column of a long-range sheet: its numeric column that is not an axis.

    Resolved from the sheet rather than named, so that the two figures can label their
    value columns after the quantity they draw (starter medio-lateral position in
    Figure 6, antero-posterior position in the reviewer figure) without either figure's
    drawing code knowing the other's names.
    """
    for column in df.columns:
        if str(column) in set(exclude) or not _is_numeric(df[column]):
            continue
        return column
    raise KeyError(f"no value column in {list(df.columns)}")


def _long_range_prefixed_column(df, prefix):
    """The one column of a long-range sheet whose name starts with ``prefix``."""
    return next((c for c in df.columns if str(c).startswith(prefix)), None)


def _long_range_value_ticks(clim):
    """Colour-bar and value-axis ticks: the ends of the drawn range and its midpoint."""
    return [clim[0], (clim[0] + clim[1]) / 2, clim[1]]


def _long_range_flatmap_figsize(window, width=4.2):
    """Figure size holding the given flatmap window at an equal aspect ratio."""
    (x0, x1), (y0, y1) = window
    # the extra height is the title, the x label and the colour bar below it
    return (width, (width - 1.2) * abs(y1 - y0) / abs(x1 - x0) + 2.2)


def _long_range_colorbar(fig, mappable, ax, style):
    """The horizontal colour bar of the starter position, on the figure value range."""
    bar = fig.colorbar(mappable, ax=ax, orientation="horizontal", shrink=0.5, pad=0.22)
    bar.set_label(style["value_label"], fontsize=8)
    bar.set_ticks(_long_range_value_ticks(style["clim"]))
    bar.ax.tick_params(labelsize=7)
    return bar


def _plot_long_range_flatmap_scatter(
    df, sheet_name, source_name, style, panel, marker_size
):
    """Panels b and c -- flatmap cell positions coloured by the starter position."""
    style = LONG_RANGE_STYLES[style]
    window = style["windows"][panel]
    value = _long_range_value_column(df, exclude=LONG_RANGE_FLATMAP_COLUMNS)
    fig, ax = plt.subplots(figsize=_long_range_flatmap_figsize(window))
    points = ax.scatter(
        df["Flatmap_X"],
        df["Flatmap_Y"],
        c=pd.to_numeric(df[value], errors="coerce"),
        cmap=style["cmap"],
        vmin=style["clim"][0],
        vmax=style["clim"][1],
        s=marker_size,
        linewidths=0,
        alpha=0.8 if panel == "b" else 0.4,
        rasterized=True,
    )
    _style_panel_axes(
        ax,
        xlabel="Flatmap X",
        ylabel="Flatmap Y",
        xlim=window[0],
        ylim=window[1],
        aspect="equal",
    )
    _long_range_colorbar(fig, points, ax, style)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_long_range_smoothed_map(df, sheet_name, source_name, style, panel="d"):
    """Panel d -- the smoothed starter map, redrawn as the image the panel shows."""
    style = LONG_RANGE_STYLES[style]
    y = pd.to_numeric(df["Flatmap_Y"], errors="coerce").to_numpy(dtype=float)
    columns = [c for c in df.columns if c != "Flatmap_Y"]
    x = np.asarray([float(c) for c in columns], dtype=float)
    image = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    window = style["windows"][panel]
    fig, ax = plt.subplots(figsize=_long_range_flatmap_figsize(window))
    # The sheet is written bottom row first, as the panel draws it.
    picture = ax.imshow(
        image,
        cmap=style["cmap"],
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        vmin=style["clim"][0],
        vmax=style["clim"][1],
        interpolation="nearest",
    )
    _style_panel_axes(
        ax,
        xlabel="Flatmap X",
        ylabel="Flatmap Y",
        xlim=window[0],
        ylim=window[1],
        aspect="equal",
    )
    _long_range_colorbar(fig, picture, ax, style)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_long_range_starter_vs_presyn(df, sheet_name, source_name, style, panel="e"):
    """Panel e -- one point per presynaptic cell, on the axes of the published panel.

    The value axis is drawn on the figure's colour range, which is also the window the
    published panel sets. Its ticks are the ends of that range and its midpoint: the
    reviewer notebook asks for ``[8.5, 9]`` inside ``ylim=(8.5, 8.9)``, so its second
    tick never renders -- do not "fix" this back to the notebook's tick list.
    """
    style = LONG_RANGE_STYLES[style]
    box_aspect = style["box_aspect"][panel]
    value = _long_range_value_column(df, exclude=(LONG_RANGE_AXIS_COLUMN,))
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect, width=4.6))
    ax.scatter(
        df[LONG_RANGE_AXIS_COLUMN],
        pd.to_numeric(df[value], errors="coerce"),
        s=3,
        alpha=0.3,
        linewidths=0,
        color="k",
        rasterized=True,
    )
    _style_panel_axes(
        ax,
        xlabel=style["axis_label"],
        ylabel=style["value_label"],
        xlim=style["axis_lim"],
        ylim=style["clim"],
        xticks=list(style["axis_ticks"]),
        yticks=_long_range_value_ticks(style["clim"]),
        box_aspect=box_aspect,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_long_range_running_average(df, sheet_name, source_name, style, panel="e"):
    """Panel e -- the running average with its shuffle band and mean-position line."""
    style = LONG_RANGE_STYLES[style]
    box_aspect = style["box_aspect"][panel]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect, width=4.6))
    x = pd.to_numeric(df[LONG_RANGE_AXIS_COLUMN], errors="coerce")
    if {"Shuffle_Lower", "Shuffle_Upper"} <= set(df.columns):
        ax.fill_between(
            x,
            pd.to_numeric(df["Shuffle_Lower"], errors="coerce"),
            pd.to_numeric(df["Shuffle_Upper"], errors="coerce"),
            color=style["curve_color"],
            alpha=0.4,
            linewidth=0,
        )
    average = _long_range_prefixed_column(df, "Running_Average_")
    ax.plot(
        x,
        pd.to_numeric(df[average], errors="coerce"),
        color=style["curve_color"],
        lw=2,
    )
    mean_column = _long_range_prefixed_column(df, "Mean_")
    if mean_column is not None:  # constant column: the dashed line
        ax.axhline(
            float(df[mean_column].iloc[0]),
            color="k",
            linestyle="dashed",
            lw=1.5,
        )
    _style_panel_axes(
        ax,
        xlabel=style["axis_label"],
        ylabel=style["value_label"],
        xlim=style["axis_lim"],
        ylim=style["clim"],
        xticks=list(style["axis_ticks"]),
        yticks=_long_range_value_ticks(style["clim"]),
        box_aspect=box_aspect,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_long_range_retinotopy_average(df, sheet_name, source_name, style, panel="f"):
    """Panel f -- the running average of the receptive-field position of the map."""
    style = LONG_RANGE_STYLES[style]
    box_aspect = style["box_aspect"][panel]
    retinotopy = style["retinotopy"]
    average = _long_range_prefixed_column(df, "Running_Average_")
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect, width=4.6))
    ax.plot(
        pd.to_numeric(df[LONG_RANGE_AXIS_COLUMN], errors="coerce"),
        pd.to_numeric(df[average], errors="coerce"),
        color="k",
        lw=2,
    )
    _style_panel_axes(
        ax,
        xlabel=style["axis_label"],
        ylabel=retinotopy["label"],
        xlim=style["axis_lim"],
        ylim=retinotopy["lim"],
        xticks=list(style["axis_ticks"]),
        yticks=list(retinotopy["ticks"]),
        box_aspect=box_aspect,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


# ---------------------------------------------------------------------------
# Figure 5 -- panels whose geometry cannot be inferred from the sheet alone
# ---------------------------------------------------------------------------

#: Height/width ratio of the Figure 5 panels whose proportions are not fixed by their
#: own data. Figure 5 is laid out at 17.4 x 17.4 cm with `fig.add_axes`, so a panel of
#: width ``w`` and height ``h`` in figure fractions has a box aspect of ``h / w``. The
#: matrices, bubble plots and diagrams are all drawn with an equal aspect instead.
FIG5_BOX_ASPECT = {
    "5d": 0.18 / 0.17,  # [0.17, 0.18], five stacked sub-axes side by side
}

#: Colour map of every Figure 5 matrix panel, as passed to `seaborn.heatmap`.
FIG5_MATRIX_CMAP = "inferno"

#: Value range of each matrix panel. ``None`` is a limit the published panel let follow
#: the data, which `plot_area_by_area_connectivity` takes as 0.7 times the smallest
#: value for `vmin` and the largest value for `vmax`; both are recomputed from the
#: sheet.
FIG5_MATRIX_VLIM = {
    "5b": (None, None),
    "5c": (0.0, 0.40),
    "5g": (0.0, None),
    "5i": (None, None),
    "5j": (0.0, 0.70),
}

#: Colour-bar title of each matrix panel. The two count matrices have no colour bar.
FIG5_MATRIX_CBAR = {
    "5c": "Input\nfraction",
    "5g": "Output\nfraction",
    "5j": "Input\nfraction",
}

#: Window of every panel a scatter, in microns, as the figure sets it. The depth axis is
#: inverted: the pia is at the top.
FIG5A_XLIM = (-800, 800)
FIG5A_YLIM = (1000, -50)

#: Node layout and edge scaling of the two connectivity diagrams, keyed by panel. The
#: positions are arbitrary drawing coordinates rather than measurements, so they belong
#: here and not in the workbook.
FIG5_DIAGRAM = {
    "5e": dict(
        positions={
            "2/3": (0, 6),
            "4": (2, 5),
            "5": (0, 4),
            "6a": (2, 3),
            "6b": (0, 2),
        },
        radius=0.5,
        edge_width_scale=20,
        arrow_head_scale=30,
        vmin=0.0,
        vmax=0.4,
    ),
    "5k": dict(
        positions={
            "Pvalb": (0, 1.5),
            "Sst": (3, 1.5),
            "Vip": (1.5, 0),
            "Lamp5": (1.5, 3),
        },
        radius=0.6,
        edge_width_scale=10,
        arrow_head_scale=20,
        vmin=0.0,
        vmax=0.5,
    ),
}

#: Colour map of the confidence-interval width of the connectivity diagrams, and the
#: smallest input fraction they draw an arrow for.
FIG5_DIAGRAM_CMAP = "RdPu_r"
FIG5_DIAGRAM_CUTOFF = 0.2

#: Colour map, colour range, bubble scaling and significance level of the two bubble
#: plots, as the figure passes them to `bubble_plot`.
FIG5_BUBBLE_VLIM = (-2, 2)
FIG5_BUBBLE_SIZE_SCALE = 80
FIG5_BUBBLE_ALPHA = 0.05


def _fig5_matrix(df):
    """Split a Figure 5 matrix sheet into its matrix and its starter-count row."""
    label_col = df.columns[0]
    is_counts = df[label_col].astype(str).str.startswith("Starter cell count")
    matrix = df[~is_counts].set_index(label_col)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix.index = matrix.index.astype(str)
    starter_counts = None
    if is_counts.any():
        row = df[is_counts].iloc[0].drop(label_col)
        starter_counts = pd.to_numeric(row, errors="coerce")
    return matrix, starter_counts, label_col


def _plot_fig5_matrix(
    df, sheet_name, source_name, panel, value_format="{:.2f}", xlabel="Starter layer"
):
    """Panels b/c/g/i/j -- a connectivity matrix, on the colour scale of the figure.

    Drawn as the published panel is: `inferno`, white lines between the cells, a black
    frame, the value in every cell, the column labels on top and, for the count
    matrices, the number of starter cells under each column.
    """
    matrix, starter_counts, label_col = _fig5_matrix(df)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmin, vmax = FIG5_MATRIX_VLIM[panel]
    if vmin is None:
        vmin = float(np.min(finite)) * 0.7
    if vmax is None:
        vmax = float(np.max(finite))

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    image = ax.imshow(values, cmap=FIG5_MATRIX_CMAP, vmin=vmin, vmax=vmax)
    threshold = float(np.max(finite)) / 2
    for (row, column), value in np.ndenumerate(values):
        ax.text(
            column,
            row,
            value_format.format(value),
            ha="center",
            va="center",
            fontsize=7,
            color="white" if value < threshold else "black",
        )
    # White grid lines between the cells and a black frame, as the heatmap draws them.
    ax.set_xticks(np.arange(-0.5, matrix.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.set_axisbelow(False)  # the lines separate the cells, so they go over the image
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    _style_panel_axes(
        ax,
        xlabel=xlabel,
        ylabel=label_col,
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        aspect="equal",
    )
    ax.set_xticklabels([str(c) for c in matrix.columns])
    ax.set_yticklabels(list(matrix.index))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="both", which="major", length=0)

    if starter_counts is not None:
        for column, count in enumerate(starter_counts.to_numpy()):
            ax.text(
                column,
                matrix.shape[0] - 0.35,
                "" if not np.isfinite(count) else f"{count:.0f}",
                ha="center",
                va="top",
                fontsize=7,
                transform=ax.transData,
            )
        ax.text(
            -0.65,
            matrix.shape[0] - 0.35,
            "N starters:",
            ha="right",
            va="top",
            fontsize=7,
        )
    if panel in FIG5_MATRIX_CBAR:
        bar = fig.colorbar(image, ax=ax, shrink=0.35, aspect=8, pad=0.04)
        bar.ax.set_title(FIG5_MATRIX_CBAR[panel], fontsize=7, loc="left")
        bar.ax.tick_params(labelsize=7)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_fig5_presyn_positions(df, sheet_name, source_name):
    """Panel a -- the presynaptic cells of every starter layer, in the panel's window.

    One sub-panel per starter layer, presynaptic cells in dark red against the
    medio-lateral offset from their starter, the starters themselves in black at an
    offset of zero, on an inverted (pia at the top) equal-aspect depth axis.
    """
    layers = list(dict.fromkeys(df["Starter_Layer"].astype(str)))
    fig, axes = plt.subplots(1, len(layers), figsize=(7.6, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, layer in zip(axes, layers):
        sub = df[df["Starter_Layer"].astype(str) == layer]
        for point_type, color, size, alpha in (
            ("Presynaptic cell", "darkred", 3, 0.5),
            ("Starter cell", "black", 6, 0.3),
        ):
            points = sub[sub["Point_Type"] == point_type]
            ax.scatter(
                points["Relative_ML_um"],
                points["Cortical_Depth_um"],
                marker=".",
                s=size,
                color=color,
                alpha=alpha,
                linewidths=0,
            )
        _style_panel_axes(
            ax,
            xlim=FIG5A_XLIM,
            ylim=FIG5A_YLIM,
            xticks=[-800, 0, 800],
            aspect="equal",
        )
        ax.set_title(layer, fontsize=8)
    axes[0].set_ylabel(_axis_label("Cortical_Depth_um"), fontsize=9)
    axes[len(axes) // 2].set_xlabel(_axis_label("Relative_ML_um"), fontsize=9)
    fig.suptitle("")
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_fig5_input_fraction_ci(df, sheet_name, source_name):
    """Panel d -- the per-starter input fractions, redrawn by the plotting function.

    The sheet holds every individual value and every mean the panel draws, so the panel
    is redrawn by `bootstrapping.plot_confidence_intervals` itself, which decides
    between a violin and a jittered scatter exactly as it did for the figure.
    """
    from brisc.manuscript_analysis import bootstrapping as boot

    individual = df[df["Series_Type"] == "Individual"]
    means = df[df["Series_Type"] == "Mean"]
    presyn = list(dict.fromkeys(means["Presynaptic_Layer"].astype(str)))
    starters = list(dict.fromkeys(means["Starter_Layer"].astype(str)))
    mean_df = means.pivot_table(
        index="Presynaptic_Layer", columns="Starter_Layer", values="Input_Fraction"
    ).reindex(index=presyn, columns=starters)

    # One wide frame per starter layer; the columns are padded to a common length
    # because the sheet does not say which starter cell each value belongs to, and the
    # plotting function takes each column independently anyway.
    frames = []
    for starter in starters:
        group = individual[individual["Starter_Layer"].astype(str) == starter]
        columns = {
            layer: group.loc[
                group["Presynaptic_Layer"].astype(str) == layer, "Input_Fraction"
            ].to_numpy(dtype=float)
            for layer in presyn
        }
        height = max((len(values) for values in columns.values()), default=0)
        wide = pd.DataFrame(
            {
                layer: np.append(values, np.full(height - len(values), np.nan))
                for layer, values in columns.items()
            }
        )
        wide["Starter_Layer"] = starter
        frames.append(wide)
    points = pd.concat(frames, ignore_index=True)

    # `plot_confidence_intervals` divides the axes it is given into one sub-axes per
    # starter layer, so the rectangle below is the whole panel; its proportions are the
    # published ones.
    width, height = 4.8, 5.6
    box_width = 0.72
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes(
        [
            0.15,
            0.11,
            box_width,
            box_width * FIG5_BOX_ASPECT["5d"] * width / height,
        ]
    )
    boot.plot_confidence_intervals(
        mean_df,
        mean_df,
        mean_df,
        ax,
        label_fontsize=9,
        tick_fontsize=7,
        line_width=0.9,
        orientation="horizontal",
        individual_points_df=points,
        individual_points_grouping_col="Starter_Layer",
        jitter_width=0.15,
        point_size=3,
        point_alpha=0.4,
        show_violin=None,
        show_error=None,
    )
    fig.suptitle(sheet_name, fontsize=11)
    footer = f"{source_name} — {len(df):,} rows" if source_name else f"{len(df):,} rows"
    fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize=6, color="0.4")
    return fig


def _plot_fig5_bubbles(df, sheet_name, source_name, show_legend=True):
    """Panels f/h -- the observed-versus-shuffle bubble plot, as the figure draws it.

    The log ratios and p-values of the sheet are pivoted back to the two matrices
    `bubble_plot` takes, so the redrawn panel is the published one.
    """
    from brisc.manuscript_analysis import connectivity_matrices as conn_mat

    rows = list(dict.fromkeys(df["Presynaptic_Group"].astype(str)))
    columns = list(dict.fromkeys(df["Starter_Group"].astype(str)))

    def _pivot(column):
        return df.pivot_table(
            index="Presynaptic_Group", columns="Starter_Group", values=column
        ).reindex(index=rows, columns=columns)

    log_ratio = _pivot("Log2_Observed_Over_Shuffle")
    pvalues = _pivot("FDR_Corrected_P_Value")

    fig = plt.figure(figsize=(4.6 if show_legend else 3.8, 3.8))
    ax = fig.add_axes([0.16, 0.08, 0.5, 0.74])
    cbax = fig.add_axes([0.72, 0.5, 0.025, 0.16]) if show_legend else None
    conn_mat.bubble_plot(
        log_ratio,
        pvalues,
        alpha=FIG5_BUBBLE_ALPHA,
        size_scale=FIG5_BUBBLE_SIZE_SCALE,
        ax=ax,
        cbax=cbax,
        show_legend=show_legend,
        label_fontsize=9,
        tick_fontsize=7,
        vmin=FIG5_BUBBLE_VLIM[0],
        vmax=FIG5_BUBBLE_VLIM[1],
        bubble_lw=1,
    )
    fig.suptitle(sheet_name, fontsize=11)
    footer = f"{source_name} — {len(df):,} rows" if source_name else f"{len(df):,} rows"
    fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize=6, color="0.4")
    return fig


def _plot_fig5_diagram(df, sheet_name, source_name, panel):
    """Panels e/k -- the connectivity diagram, redrawn by the plotting function.

    The drawn arrows of the sheet are put back into the input-fraction and
    confidence-bound matrices `connectivity_diagram_mpl` takes, with a zero for every
    connection the panel does not draw, so the diagram comes out as published.
    """
    from brisc.manuscript_analysis import connectivity_matrices as conn_mat

    style = FIG5_DIAGRAM[panel]
    names = list(style["positions"])
    fraction = pd.DataFrame(0.0, index=names, columns=names)
    lower, upper = fraction.copy(), fraction.copy()
    for row in df.itertuples(index=False):
        presyn = str(row.Presynaptic_Group)
        starter = str(row.Starter_Group)
        if presyn not in names or starter not in names:
            continue
        fraction.loc[presyn, starter] = float(row.Input_Fraction)
        lower.loc[presyn, starter] = float(row.CI_Lower)
        upper.loc[presyn, starter] = float(row.CI_Upper)

    fig = plt.figure(figsize=(3.8, 4.0))
    ax = fig.add_axes([0.02, 0.05, 0.76, 0.82])
    cax = fig.add_axes([0.84, 0.12, 0.03, 0.18])
    conn_mat.connectivity_diagram_mpl(
        fraction,
        lower,
        upper,
        connection_names=names,
        positions=style["positions"],
        display_names=names,
        node_style=dict(facecolor="Lightgray", radius=style["radius"], fontsize=7),
        min_fraction_cutoff=FIG5_DIAGRAM_CUTOFF,
        ci_to_alpha=False,
        ci_cmap=FIG5_DIAGRAM_CMAP,
        edge_width_scale=style["edge_width_scale"],
        arrow_head_scale=style["arrow_head_scale"],
        arrow_style=dict(connectionstyle="Arc3, rad=-0.2", ec="none"),
        ax=ax,
        cax=cax,
        vmin=style["vmin"],
        vmax=style["vmax"],
    )
    fig.suptitle(sheet_name, fontsize=11)
    footer = f"{source_name} — {len(df):,} rows" if source_name else f"{len(df):,} rows"
    fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize=6, color="0.4")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 -- panels whose geometry cannot be inferred from the sheet alone
# ---------------------------------------------------------------------------

#: Height/width ratio of each Figure 4 panel's axes box, keyed by panel letter. Figure 4
#: is laid out at 17.4 x 17.4 cm with `fig.add_axes`, so a panel of width ``w`` and
#: height ``h`` in figure fractions has a box aspect of ``h / w``. The anatomical
#: panels (a, b, f, g, h) are drawn with an equal aspect instead and are not listed.
FIG4_BOX_ASPECT = {
    "4c": 1.875,  # [0.08, 0.15]
    "4d": 1.875,  # [0.08, 0.15]
    "4e": 1.50,  # [0.10, 0.15]
    "4i": 0.588,  # [0.17, 0.10]
}

#: Colour of every cortical area of panels a and b, as the notebook passes them.
FIG4_AREA_COLORS = {
    "AUDp": "limegreen",
    "AUDpo": "mediumseagreen",
    "AUDv": "springgreen",
    "RSP": "darkorchid",
    "TEa": "forestgreen",
    "ECT": "darkolivegreen",
    "TH": "orangered",
    "VISal": "aquamarine",
    "VISl": "darkturquoise",
    "VISli": "mediumaquamarine",
    "VISp": "deepskyblue",
    "VISpm": "royalblue",
}

#: Windows of the anatomical panels, in the order matplotlib takes them, so that the
#: axes the published panel inverts come out inverted here too.
FIG4A_XLIM = (1100, 570)
FIG4A_YLIM = (450, 0)
FIG4B_XLIM = (1200, 100)
FIG4B_YLIM = (1350, 740)
FIG4F_CORONAL_XLIM = (1100, 570)
FIG4F_CORONAL_YLIM = (450, 0)
FIG4F_FLATMAP_XLIM = (980, 250)
FIG4F_FLATMAP_YLIM = (1250, 850)

#: Cortical depth window of panel c, drawn downwards.
FIG4C_YLIM = (1000, 0)

#: Colour of each cell population of panel c, in the order `plot_layer_distribution`
#: stacks the two halves of the split violin.
FIG4C_COLORS = {"Starter cell": "black", "Presynaptic cell": "gray"}

#: Legend entry of the median line the redrawn panel c adds as a reading aid. The
#: published panel draws no such line: its violins are drawn with ``inner=None``.
FIG4C_MEDIAN_LABEL = "Median (not in the published panel)"

#: Window of panel i and the bandwidth method the notebook estimates its density with.
FIG4I_XLIM = (-5, 5)
FIG4I_BW_METHOD = 0.05

#: Colour of the stacked series of panel e, in stacking order.
FIG4E_COLORS = (
    "lightskyblue",
    "dodgerblue",
    "royalblue",
    "darkblue",
    "red",
    "orange",
    "orangered",
)

#: Colour of each example barcode of panel f, in the order the notebook passes them, and
#: of the grey background of all barcoded cells.
FIG4F_COLORS = ("dodgerblue", "forestgreen", "darkorange")
FIG4F_BACKGROUND = "gray"


def _plot_cell_positions(
    df, sheet_name, source_name, xcol, ycol, xlim, ylim, width=4.0
):
    """Panels a/b -- every barcoded cell coloured by area, starters in black on top."""
    span_x = abs(xlim[1] - xlim[0])
    span_y = abs(ylim[1] - ylim[0])
    fig, ax = plt.subplots(figsize=(width, width * span_y / span_x + 0.9))
    ax.scatter(
        df[xcol],
        df[ycol],
        s=1,
        alpha=0.3,
        linewidths=0,
        c=[FIG4_AREA_COLORS.get(a, "0.5") for a in df["Cortical_Area"]],
        rasterized=True,
    )
    starters = df[df["Is_Starter"].astype(bool)]
    ax.scatter(
        starters[xcol],
        starters[ycol],
        s=2,
        alpha=0.6,
        linewidths=0,
        c="black",
        rasterized=True,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    areas = [a for a in FIG4_AREA_COLORS if a in set(df["Cortical_Area"])]
    ax.legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", lw=0, color=FIG4_AREA_COLORS[a], label=a, ms=3
            )
            for a in areas
        ],
        loc="lower left",
        ncols=4,
        frameon=False,
        fontsize=5,
        handlelength=1,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_cortical_depth(df, sheet_name, source_name):
    """Fig 4c -- the cortical depth of every barcoded cell of VISp, by population.

    The published panel draws these depths as a split violin with ``inner=None``, so it
    carries no mean or median line. Here they are a jittered strip per population, with
    a short line at the median of each as a reading aid for this check panel only: it is
    not a feature of the figure and must not be added to it, and it is not in the
    workbook either -- it is computed here from the depths the sheet holds. The dashed
    layer boundaries of the published panel are Allen atlas averages, not data, so they
    are neither in the sheet nor redrawn here.
    """
    cells = df
    box_aspect = FIG4_BOX_ASPECT["4c"]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect, width=3.0))
    rng = np.random.default_rng(0)
    populations = [p for p in FIG4C_COLORS if p in set(cells["Series"])]
    for position, series in enumerate(populations):
        depths = (
            pd.to_numeric(
                cells.loc[cells["Series"] == series, "Cortical_Depth_um"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        if not depths.size:
            continue
        ax.scatter(
            position + rng.uniform(-0.25, 0.25, len(depths)),
            depths,
            s=2,
            alpha=0.5,
            linewidths=0,
            color=FIG4C_COLORS[series],
            rasterized=True,
            label=series,
        )
        ax.plot(
            [position - 0.38, position + 0.38],
            [np.median(depths)] * 2,
            color="crimson",
            lw=1.2,
            zorder=3,
            label=FIG4C_MEDIAN_LABEL if position == 0 else None,
        )
    _style_panel_axes(
        ax,
        ylabel="Cortical depth (um)",
        xlim=(-0.6, len(populations) - 0.4),
        ylim=FIG4C_YLIM,
        xticks=range(len(populations)),
        box_aspect=box_aspect,
    )
    ax.set_xticklabels(populations, fontsize=6)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.3, 1.0),
        frameon=False,
        fontsize=6,
        handlelength=1,
        markerscale=4,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_starters_per_presynaptic(df, sheet_name, source_name):
    """Fig 4d -- the number of starters a presynaptic cell is connected to."""
    box_aspect = FIG4_BOX_ASPECT["4d"]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect, width=2.8))
    values = df["Connected_Starters"].to_numpy(dtype=float)
    proportions = df["Proportion_Of_Cells"].to_numpy(dtype=float)
    ax.bar(
        values,
        proportions,
        width=1,
        edgecolor="black",
        facecolor="slategray",
        linewidth=0.5,
    )
    if "Cell_Count" in df.columns:  # the counts written above the bars
        for value, proportion, count in zip(values, proportions, df["Cell_Count"]):
            ax.text(
                value - 0.2,
                proportion + 0.05,
                f"{int(count)}",
                ha="left",
                fontsize=6,
                color="black",
                alpha=0.8,
                rotation=35,
            )
    _style_panel_axes(
        ax,
        xlabel="Connected starters",
        ylabel="Proportion of cells",
        xlim=(values.min() - 0.5, values.max() + 0.5),
        ylim=(0, 1),
        xticks=values,
        box_aspect=box_aspect,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_multibarcoded_starters(df, sheet_name, source_name):
    """Fig 4e -- presynaptic cells per barcode of every multi-barcoded starter."""
    box_aspect = FIG4_BOX_ASPECT["4e"]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect, width=3.0))
    x = df["Starter_Rank"].to_numpy(dtype=float)
    series = [c for c in df.columns if c != "Starter_Rank"]
    bottom = np.zeros(len(df))
    for column, color in zip(series, FIG4E_COLORS):
        height = df[column].to_numpy(dtype=float)
        ax.bar(
            x,
            height,
            bottom=bottom,
            width=1,
            color=color,
            label=str(column).replace("_", " "),
        )
        bottom = bottom + height
    _style_panel_axes(
        ax,
        xlabel="Starter cell",
        ylabel="# presynaptic cells",
        xticks=[x.min(), x.max()],
        box_aspect=box_aspect,
    )
    ax.legend(
        fontsize=5,
        loc="upper left",
        bbox_to_anchor=(0.55, 1.05),
        frameon=False,
        handlelength=1,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_example_barcodes(df, sheet_name, source_name):
    """Fig 4f -- the cells of each example barcode, coronal section and flatmap.

    The two panels of the figure draw the same cells, so both are redrawn here, each
    with its own window. The lines are the ones of the panel: from the starter of a
    barcode to every cell carrying it.
    """
    background = df["Barcode"] == "All barcoded cells"
    barcodes = [b for b in pd.unique(df.loc[~background, "Barcode"])]
    colors = dict(zip(barcodes, FIG4F_COLORS * (len(barcodes) // 3 + 1)))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))
    for ax, (xcol, ycol, xlim, ylim) in zip(
        axes,
        (
            ("ARA_Z_px", "ARA_Y_px", FIG4F_CORONAL_XLIM, FIG4F_CORONAL_YLIM),
            ("Flatmap_X", "Flatmap_Y", FIG4F_FLATMAP_XLIM, FIG4F_FLATMAP_YLIM),
        ),
    ):
        grey = df[background]
        ax.scatter(
            grey[xcol],
            grey[ycol],
            s=1,
            alpha=0.15,
            linewidths=0,
            c=FIG4F_BACKGROUND,
            rasterized=True,
        )
        for barcode in barcodes:
            cells = df[df["Barcode"] == barcode]
            color = colors[barcode]
            ax.scatter(
                cells[xcol],
                cells[ycol],
                s=4,
                linewidths=0,
                c=color,
                zorder=2,
                rasterized=True,
            )
            starter = cells[cells["Point_Type"] == "Starter cell"]
            for _, presynaptic in cells.iterrows():
                for _, origin in starter.iterrows():
                    ax.plot(
                        [origin[xcol], presynaptic[xcol]],
                        [origin[ycol], presynaptic[ycol]],
                        color=color,
                        lw=0.5,
                        alpha=0.5,
                        zorder=-3,
                        rasterized=True,
                    )
            ax.scatter(
                starter[xcol],
                starter[ycol],
                s=25,
                edgecolors="black",
                linewidths=1,
                c=color,
                zorder=3,
                label=barcode,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_axis_off()
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=5,
        ncols=2,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_relative_coors(df, sheet_name, source_name):
    """Figs 4g/h -- presynaptic positions relative to their starter, on equal axes."""
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.scatter(
        df["Relative_ML_Location_mm"],
        df["Relative_Cortical_Depth_mm"],
        s=1,
        alpha=0.05,
        color="black",
        edgecolors="none",
        rasterized=True,
    )
    _style_panel_axes(
        ax,
        xlabel="Relative M-L location (mm)",
        ylabel="Relative cortical\ndepth (mm)",
        xlim=(-5, 5),
        ylim=(-1, 1),
        yticks=[-1, 0, 1],
        aspect="equal",
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_ml_kde(df, sheet_name, source_name):
    """Fig 4i -- the observed medio-lateral density over the shuffle band.

    The sheet holds one relative position per presynaptic cell, so the kernel density
    estimate of the published panel is recomputed here with its bandwidth.
    """
    from scipy.stats import gaussian_kde

    box_aspect = FIG4_BOX_ASPECT["4i"]
    fig, ax = plt.subplots(figsize=_panel_figsize(box_aspect, width=3.4))
    shuffle = df[df["Series"] == "Shuffle"]
    ax.fill_between(
        shuffle["Relative_ML_Location_mm"],
        shuffle["Shuffle_Density_Lower"],
        shuffle["Shuffle_Density_Upper"],
        color="gray",
        lw=0,
        zorder=1,
        label="Shuffle",
    )
    observed = pd.to_numeric(
        df.loc[df["Series"] == "Observed", "Relative_ML_Location_mm"], errors="coerce"
    ).dropna()
    grid = np.linspace(*FIG4I_XLIM, 400)
    ax.plot(
        grid,
        gaussian_kde(observed.to_numpy(dtype=float), bw_method=FIG4I_BW_METHOD)(grid),
        color="black",
        lw=0.9,
        zorder=2,
        label="Observed",
    )
    _style_panel_axes(
        ax,
        xlabel="Relative M-L location (mm)",
        ylabel="Density",
        xlim=FIG4I_XLIM,
        ylim=(0, None),
        yticks=[0, 0.5],
        box_aspect=box_aspect,
    )
    ax.legend(fontsize=6, frameon=False, handlelength=1, loc="upper right")
    _finish(fig, sheet_name, source_name, len(df))
    return fig


# ---------------------------------------------------------------------------
# Supplementary Figures panel draw functions
# ---------------------------------------------------------------------------


def _plot_supp1d_starter_dilution(df, sheet_name, source_name):
    fig, ax = plt.subplots(figsize=_panel_figsize(1.0))
    for label, sub in _groups(df, "dilution"):
        ax.scatter(sub["dilution"], sub["density"], s=15, label=str(label))
    _style_panel_axes(
        ax,
        xlabel="Dilution",
        ylabel="Cell density (mm^-3)",
        yscale="log",
        box_aspect=1.0,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_supp5a_cells_per_section(df, sheet_name, source_name):
    fig, ax = plt.subplots(figsize=_panel_figsize(1.0))
    for label, sub in _groups(df, "Cell_Type"):
        ax.plot(
            sub["Section_Position_um"],
            sub["Cell_Count"],
            "o-",
            ms=3,
            label=str(label),
        )
    _style_panel_axes(
        ax, xlabel="Section position (um)", ylabel="Cell count", box_aspect=1.0
    )
    ax.legend(fontsize=6, frameon=False)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_supp5b_coronal_positions(df, sheet_name, source_name):
    fig, ax = plt.subplots(figsize=_panel_figsize(1.0))
    for label, sub in _groups(df, "Cell_Group"):
        ax.scatter(sub["ARA_Z_px"], sub["ARA_Y_px"], s=1, alpha=0.3, label=str(label))
    _style_panel_axes(
        ax,
        xlabel="ARA Z (px)",
        ylabel="ARA Y (px)",
        aspect="equal",
    )
    ax.invert_yaxis()
    ax.legend(fontsize=6, frameon=False)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_supp5c_dorsal_positions(df, sheet_name, source_name):
    fig, ax = plt.subplots(figsize=_panel_figsize(1.0))
    for label, sub in _groups(df, "Cell_Group"):
        ax.scatter(sub["ARA_X_px"], sub["ARA_Z_px"], s=1, alpha=0.3, label=str(label))
    _style_panel_axes(
        ax,
        xlabel="ARA X (px)",
        ylabel="ARA Z (px)",
        aspect="equal",
    )
    ax.invert_xaxis()
    ax.legend(fontsize=6, frameon=False)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_supp5d_dotplot(df, sheet_name, source_name):
    if "Mean_Expression" in df.columns and "Gene" in df.columns:
        matrix = df.pivot(index="Cluster", columns="Gene", values="Mean_Expression")
        return _plot_heatmap(
            matrix, sheet_name, source_name, len(df), xlabel="Gene", ylabel="Cluster"
        )
    fig, ax = plt.subplots(figsize=_panel_figsize(1.0))
    _finish(fig, sheet_name, source_name, len(df))
    return fig


#: Layout of the Supp 6 mosaic, as the notebook builds it: pairs of (coronal, depth
#: KDE) panels, four pairs per row, on a 17.4 x 12 cm figure.
SUPP6_PAIRS_PER_ROW = 4

#: Coronal window of every mosaic panel, both axes inverted as the figure draws them.
SUPP6_XLIM = (1100, 500)
SUPP6_YLIM = (420, 0)


def _plot_supp6_mosaic(df, sheet_name, source_name, ycol, xcol, kind):
    """One small panel per cluster, laid out as the published mosaic.

    Both Supp 6 sheets are per-cluster: the coronal scatter and the depth density beside
    it. Drawing them overlaid would hide exactly what the figure is about -- where each
    cluster sits -- so the redraw keeps the mosaic.
    """
    clusters = list(pd.unique(df["Cluster"].dropna()))
    ncols = min(SUPP6_PAIRS_PER_ROW, len(clusters))
    nrows = int(np.ceil(len(clusters) / ncols))
    palette = dict(zip(clusters, _categorical_palette(len(clusters))))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.0 * nrows + 0.8))
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for ax, cluster in zip(axes, clusters):
        sub = df[df["Cluster"] == cluster]
        color = palette.get(cluster, "0.5")
        if kind == "scatter":
            ax.scatter(
                sub[xcol],
                sub[ycol],
                s=0.4,
                alpha=0.4,
                linewidths=0,
                color=color,
                rasterized=True,
            )
            ax.set_xlim(*SUPP6_XLIM)
            ax.set_ylim(*SUPP6_YLIM)
            ax.set_aspect("equal")
        else:
            ax.plot(sub[xcol], sub[ycol], lw=1.0, color=color)
            ax.set_ylim(1000, 0)  # cortical depth, pia at the top as drawn
        ax.set_title(str(cluster), fontsize=6)
        ax.tick_params(labelsize=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(clusters) :]:
        ax.set_axis_off()
    _finish(fig, sheet_name, source_name, len(df))
    if nrows > 1:
        fig.subplots_adjust(hspace=0.35)
    return fig


#: Colour of every Supp 8 series, by the label its sheet carries.
SUPP8_COLORS = {
    "Library barcodes": "black",
    "All in situ barcodes": "#e78ac3",
    "Barcodes in multiple starter cells": "#8da0cb",
    "Multiple starter - Library": "#8da0cb",
    "All in situ - Library": "#e78ac3",
    "Different barcode": "#fc8d62",
    "Same barcode": "#66c2a5",
    "Same barcode, excluding adjacent sections": "#a6d854",
}

#: Height/width ratio of the three Supp 8 panels, all [0.22, 0.75] of a 16 x 5 cm figure.
SUPP8_BOX_ASPECT = 1.07


def _plot_supp8_curves(df, sheet_name, source_name, xcol, ycol, zero_line=False):
    """One line per series, with the confidence band of whichever series carries one."""
    fig, ax = plt.subplots(figsize=_panel_figsize(SUPP8_BOX_ASPECT))
    if zero_line:
        ax.axhline(0, color="grey", ls="--", lw=0.8, zorder=-1)
    for series, sub in _groups(df, "Series"):
        color = SUPP8_COLORS.get(series, "0.5")
        style = "--" if series == "Library barcodes" else "-"
        ax.plot(sub[xcol], sub[ycol], style, color=color, lw=1.2, label=str(series))
        if "CI_Lower" in sub.columns and sub["CI_Lower"].notna().any():
            ax.fill_between(
                sub[xcol],
                sub["CI_Lower"],
                sub["CI_Upper"],
                color=color,
                alpha=0.2,
                lw=0,
                zorder=0,
            )
    _style_panel_axes(ax, xlabel=xcol, ylabel=ycol, box_aspect=SUPP8_BOX_ASPECT)
    ax.legend(fontsize=6, frameon=False, handlelength=1)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_supp8c_pairwise_distances(df, sheet_name, source_name):
    fig, ax = plt.subplots(figsize=_panel_figsize(1.0))
    for label, sub in _groups(df, "Comparison"):
        ax.plot(
            sub["Distance_Between_Starters_mm"],
            sub["Density"],
            lw=1.2,
            label=str(label),
        )
    _style_panel_axes(
        ax,
        xlabel="Distance between starters (mm)",
        ylabel="Density",
        xlim=(0, 2),
        ylim=(0, 1.5),
        box_aspect=1.0,
    )
    ax.legend(fontsize=6, frameon=False)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_supp8c_median_distances(df, sheet_name, source_name):
    fig, ax = plt.subplots(figsize=_panel_figsize(1.0))
    x = range(len(df))
    ax.errorbar(
        x,
        df["Bootstrap_Median_mm"],
        yerr=[
            df["Bootstrap_Median_mm"] - df["CI_Lower_mm"],
            df["CI_Upper_mm"] - df["Bootstrap_Median_mm"],
        ],
        fmt="o",
        color="k",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["Comparison"], rotation=45, fontsize=6)
    _style_panel_axes(ax, ylabel="Bootstrap median (mm)", box_aspect=1.0)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_supp9a_injection_site(df, sheet_name, source_name):
    fig, ax = plt.subplots(figsize=_panel_figsize(1.0))
    hue = "is_barcoded" if "is_barcoded" in df.columns else None
    for label, sub in _groups(df, hue):
        ax.scatter(sub["ARA_Z"], sub["ARA_Y"], s=1, alpha=0.3, label=str(label))
    _style_panel_axes(ax, xlabel="ARA Z", ylabel="ARA Y", aspect="equal")
    _finish(fig, sheet_name, source_name, len(df))
    return fig


def _plot_supp9b_observed_vs_expected(df, sheet_name, source_name):
    fig, ax = plt.subplots(figsize=_panel_figsize(1.0))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(
        x - width / 2,
        df["Observed_Cells"],
        width,
        label="Observed",
        color="dodgerblue",
    )
    ax.bar(
        x + width / 2,
        df["Expected_Cells_Poisson"],
        width,
        label="Expected (Poisson)",
        color="salmon",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["Barcodes_Per_Cell"])
    _style_panel_axes(
        ax,
        xlabel="Barcodes per cell",
        ylabel="Number of cells",
        yscale="log",
        box_aspect=1.0,
    )
    ax.legend(fontsize=6, frameon=False)
    _finish(fig, sheet_name, source_name, len(df))
    return fig


# ---------------------------------------------------------------------------
# Supplementary Figure 4 -- panels whose geometry cannot be inferred from the
# sheet alone
# ---------------------------------------------------------------------------

#: Height/width ratio of each Supplementary Figure 4 panel's axes box. The figure is
#: laid out at 14.0 x 4.5 cm as an equal 1 x 3 grid of subplots under
#: ``fig.tight_layout()``, which leaves each axes 0.176 of the width and 0.610 of the
#: height, so all three share a box aspect of ``0.610 * 4.5 / (0.176 * 14.0)``. The
#: redrawn panels keep those proportions.
SUPP4_BOX_ASPECT = 1.11

#: Colour of each barcode-length library, by the label the sheet carries. Panels a and b
#: draw the same two libraries in the same colours.
SUPP4_LIBRARY_COLORS = {
    "Viral library - 10 nucleotides": "midnightblue",
    "Viral library - 20 nucleotides": "darkorange",
}

#: Barcode lengths panel c marks with a filled marker, in the colour panels a and b give
#: that library. They are points of the curve the sheet already holds, so the colours
#: are geometry rather than data.
SUPP4C_HIGHLIGHTS = {10: "midnightblue", 20: "darkorange"}


def _plot_supp4_unique_vs_length(df, sheet_name, source_name):
    """Supp 4c -- infections for 95% unique labelling, with its highlighted lengths."""
    fig, ax = plt.subplots(figsize=_panel_figsize(SUPP4_BOX_ASPECT))
    lengths = df["Barcode_Length_Nucleotides"]
    infections = df["Infections_For_95pc_Unique"]
    ax.plot(lengths, infections, "o-", color="k", mfc="w", ms=4, mew=0.5, lw=1)
    for length, color in SUPP4C_HIGHLIGHTS.items():
        marked = infections[lengths == length]
        if not marked.empty:
            ax.plot(length, marked.iloc[0], "o", mfc=color, mec="none", ms=4)
    _style_panel_axes(
        ax,
        xlabel="Number of nucleotides",
        ylabel="Number of infection events\nfor 95% unique labelling rate",
        xlim=(3.5, 20.5),
        ylim=(0, 1400),
        xticks=np.arange(4, 21, 2),
        yticks=np.arange(0, 1410, 300),
        box_aspect=SUPP4_BOX_ASPECT,
    )
    _finish(fig, sheet_name, source_name, len(df))
    return fig


# ---------------------------------------------------------------------------
# Supplementary reviewer figure -- Figure 6 along antero-posterior and elevation
# ---------------------------------------------------------------------------

# The reviewer figure redraws Figure 6 along another axis: the same six panels, the same
# `fig.add_axes` layout at 8.8 x 20.0 cm and therefore the same flatmap windows and
# proportions, with the starter antero-posterior position on `turbo` over 8.5 to 8.9 mm
# in place of the medio-lateral position, and receptive-field elevation in place of
# azimuth in panel f. It needs no drawing code of its own: its `PANEL_SPECS` entries
# below delegate to the long-range family above with ``style="reviewer"``, and the value
# column of each sheet is resolved from the sheet itself. Its retinotopy inset, flatmap
# outlines and the dashed data-coverage contour drawn on that inset are images,
# listed in `supplementary.SUPP_REVIEWER_IMAGE_KEYS`, so no worksheet and no panel
# redraws them.


# ---------------------------------------------------------------------------
# Per-panel drawing specifications
# ---------------------------------------------------------------------------

#: What the numbers of a sheet cannot say about how its panel is drawn, matched on a
#: lower-case substring of the sheet name (first match wins, so specific keys go
#: first). ``draw`` replaces the inferred plot entirely; the other keys refine it:
#: ``xscale``/``yscale``, ``aspect``, ``invert_x``/``invert_y`` and ``frameon``.
PANEL_SPECS = [
    # Figure 1, whose panels are drawn on the axes of the published figure.
    (
        "1d library abundance",
        {"kind": "line", "draw": _fig1_abundance("1d")},
    ),
    ("1e unique fraction", {"kind": "line", "draw": _fig1_unique("1e", 1e6)}),
    (
        "1f rescue scaling abundance",
        {"kind": "line", "draw": _fig1_abundance("1f")},
    ),
    (
        "1g rescue scaling unique",
        {"kind": "line", "draw": _fig1_unique("1g", 1e4)},
    ),
    (
        "1f library comparison abund",
        {"kind": "line", "draw": _fig1_abundance("1f")},
    ),
    (
        "1g library compar unique",
        {"kind": "line", "draw": _fig1_unique("1g", 1e6)},
    ),
    ("1h starter spread", {"kind": "line", "draw": _plot_starter_spread_sim}),
    (
        "1k presynaptic density",
        {
            "kind": "line",
            "draw": partial(
                _plot_panel_curves,
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
    ("1m starter positions", {"kind": "scatter", "draw": _plot_starter_positions}),
    ("1n pairwise distances", {"kind": "line", "draw": _plot_pairwise_distances}),
    (
        "4a_1 coronal cell positions",
        {
            "kind": "scatter",
            "draw": partial(
                _plot_cell_positions,
                xcol="ARA_Z_px",
                ycol="ARA_Y_px",
                xlim=FIG4A_XLIM,
                ylim=FIG4A_YLIM,
            ),
        },
    ),
    (
        "4a_2 flatmap cell positions",
        {
            "kind": "scatter",
            "draw": partial(
                _plot_cell_positions,
                xcol="Flatmap_X",
                ycol="Flatmap_Y",
                xlim=FIG4B_XLIM,
                ylim=FIG4B_YLIM,
                width=4.4,
            ),
        },
    ),
    ("4b cortical depth", {"kind": "strip", "draw": _plot_cortical_depth}),
    (
        "4c starters per presynaptic",
        {"kind": "bar", "draw": _plot_starters_per_presynaptic},
    ),
    (
        "4d multibarcoded starters",
        {"kind": "bar", "draw": _plot_multibarcoded_starters},
    ),
    ("4e example barcodes", {"kind": "scatter", "draw": _plot_example_barcodes}),
    (
        "4f_1 relative coords observed",
        {"kind": "scatter", "draw": _plot_relative_coors},
    ),
    (
        "4f_2 relative coords shuffled",
        {"kind": "scatter", "draw": _plot_relative_coors},
    ),
    ("4g ml kde vs shuffle", {"kind": "band", "draw": _plot_ml_kde}),
    # Figure 5, on the colour scales, layer order and proportions of the published one.
    (
        "5a presyn pos by layer",
        {"kind": "scatter", "draw": _plot_fig5_presyn_positions},
    ),
    (
        "5b counts matrix",
        {
            "kind": "heatmap",
            "draw": partial(_plot_fig5_matrix, panel="5b", value_format="{:.0f}"),
        },
    ),
    (
        "5c mean input fraction",
        {"kind": "heatmap", "draw": partial(_plot_fig5_matrix, panel="5c")},
    ),
    (
        "5d input fraction by layer",
        {"kind": "points", "draw": _plot_fig5_input_fraction_ci},
    ),
    (
        "5e connectivity diagram",
        {"kind": "diagram", "draw": partial(_plot_fig5_diagram, panel="5e")},
    ),
    ("5e input vs shuffle", {"kind": "bubble", "draw": _plot_fig5_bubbles}),
    (
        "5f mean output fraction",
        {"kind": "heatmap", "draw": partial(_plot_fig5_matrix, panel="5g")},
    ),
    (
        "5g output vs shuffle",
        {"kind": "bubble", "draw": partial(_plot_fig5_bubbles, show_legend=False)},
    ),
    (
        "5i_1 interneuron counts",
        {
            "kind": "heatmap",
            "draw": partial(
                _plot_fig5_matrix,
                panel="5i",
                value_format="{:.0f}",
                xlabel="Starter cell type",
            ),
        },
    ),
    (
        "5i_2 interneuron input fract",
        {
            "kind": "heatmap",
            "draw": partial(_plot_fig5_matrix, panel="5j", xlabel="Starter cell type"),
        },
    ),
    (
        "5j interneuron diagram",
        {"kind": "diagram", "draw": partial(_plot_fig5_diagram, panel="5k")},
    ),
    # Figure 6, on the flatmap windows and axes of the published figure.
    (
        "6b starter positions",
        {
            "kind": "scatter",
            "draw": partial(
                _plot_long_range_flatmap_scatter,
                style="fig6",
                panel="b",
                marker_size=8,
            ),
        },
    ),
    (
        "6c presynaptic positions",
        {
            "kind": "scatter",
            "draw": partial(
                _plot_long_range_flatmap_scatter,
                style="fig6",
                panel="c",
                marker_size=1,
            ),
        },
    ),
    (
        "6c smoothed starter map",
        {
            "kind": "heatmap",
            "draw": partial(_plot_long_range_smoothed_map, style="fig6"),
        },
    ),
    (
        "6e starter vs presyn ml",
        {
            "kind": "scatter",
            "draw": partial(_plot_long_range_starter_vs_presyn, style="fig6"),
        },
    ),
    (
        "6e running average",
        {
            "kind": "line",
            "draw": partial(_plot_long_range_running_average, style="fig6"),
        },
    ),
    (
        "6f azimuth running avg",
        {
            "kind": "line",
            "draw": partial(_plot_long_range_retinotopy_average, style="fig6"),
        },
    ),
    # The supplementary reviewer figure, Figure 6 along the antero-posterior axis and
    # elevation: the same panels, on the same windows, with the reviewer style.
    (
        "rev b starter positions",
        {
            "kind": "scatter",
            "draw": partial(
                _plot_long_range_flatmap_scatter,
                style="reviewer",
                panel="b",
                marker_size=8,
            ),
        },
    ),
    (
        "rev c presynaptic positions",
        {
            "kind": "scatter",
            "draw": partial(
                _plot_long_range_flatmap_scatter,
                style="reviewer",
                panel="c",
                marker_size=1,
            ),
        },
    ),
    (
        "rev d smoothed starter map",
        {
            "kind": "heatmap",
            "draw": partial(_plot_long_range_smoothed_map, style="reviewer"),
        },
    ),
    (
        "rev e starter vs presyn",
        {
            "kind": "scatter",
            "draw": partial(_plot_long_range_starter_vs_presyn, style="reviewer"),
        },
    ),
    (
        "rev e running average",
        {
            "kind": "line",
            "draw": partial(_plot_long_range_running_average, style="reviewer"),
        },
    ),
    (
        "rev f elevation running avg",
        {
            "kind": "line",
            "draw": partial(_plot_long_range_retinotopy_average, style="reviewer"),
        },
    ),
    # Figure 3, likewise drawn on the axes of the published figure.
    (
        "3a barcodes per cell",
        {"kind": "bar", "draw": _plot_fig3_barcodes_per_cell},
    ),
    ("3b match to library", {"kind": "bar", "draw": _plot_fig3_match_to_library}),
    (
        "3c starters per barcode",
        {"kind": "bar", "draw": _plot_fig3_starters_per_barcode},
    ),
    (
        "3d presynaptic per barcode",
        {"kind": "bar", "draw": _plot_fig3_presyn_per_barcode},
    ),
    ("3e spots per cell", {"kind": "bar", "draw": _plot_fig3_spots_per_cell}),
    ("3f mcherry vs presynaptic", {"kind": "scatter", "draw": _plot_fig3_mcherry}),
    ("umap by cluster", {"kind": "scatter", "draw": _plot_umap_clusters}),
    ("umap barcoded cells", {"kind": "scatter", "draw": _plot_umap_barcoded}),
    ("gene expression map", {"kind": "mosaic", "draw": _plot_gene_expression_mosaic}),
    # Supplementary Figures
    (
        "supp 2c presynaptic density",
        {
            "kind": "line",
            "draw": partial(
                _plot_panel_curves,
                xcol="Sorted_Isocortex_Voxel_Distances_um",
                ycol="Sorted_Labelled_Cell_Distances_um",
                box_aspect=1.15,
            ),
        },
    ),
    (
        "supp 2d starter dilution",
        {"kind": "points", "draw": _plot_supp1d_starter_dilution},
    ),
    # Supplementary Figure 4, on the axes of Figure 1's library panels.
    (
        "supp 4a library abundance",
        {
            "kind": "line",
            "draw": _abundance_draw(SUPP4_LIBRARY_COLORS, SUPP4_BOX_ASPECT),
        },
    ),
    (
        "supp 4b unique fraction",
        {
            "kind": "line",
            "draw": _unique_draw(SUPP4_LIBRARY_COLORS, SUPP4_BOX_ASPECT, 1e6),
        },
    ),
    (
        "supp 4c unique vs length",
        {"kind": "line", "draw": _plot_supp4_unique_vs_length},
    ),
    (
        "supp 5a cells per section",
        {"kind": "line", "draw": _plot_supp5a_cells_per_section},
    ),
    (
        "supp 5b coronal positions",
        {"kind": "scatter", "draw": _plot_supp5b_coronal_positions},
    ),
    (
        "supp 5c dorsal positions",
        {"kind": "scatter", "draw": _plot_supp5c_dorsal_positions},
    ),
    (
        "supp 5c marker gene expression",
        {"kind": "heatmap", "draw": _plot_supp5d_dotplot},
    ),
    (
        "supp 6 cluster positions",
        {
            "kind": "mosaic",
            "draw": partial(
                _plot_supp6_mosaic, xcol="ARA_Z_px", ycol="ARA_Y_px", kind="scatter"
            ),
        },
    ),
    (
        "supp 6 cluster depth kde",
        {
            "kind": "mosaic",
            "draw": partial(
                _plot_supp6_mosaic,
                xcol="Normalised_Density",
                ycol="Cortical_Depth_um",
                kind="line",
            ),
        },
    ),
    (
        "supp 8a library abundance kde",
        {
            "kind": "line",
            "draw": partial(
                _plot_supp8_curves, xcol="Log10_Library_Reads", ycol="Density"
            ),
        },
    ),
    (
        "supp 8b density difference",
        {
            "kind": "line",
            "draw": partial(
                _plot_supp8_curves,
                xcol="Log10_Library_Reads",
                ycol="Density_Difference",
                zero_line=True,
            ),
        },
    ),
    (
        "supp 8c pairwise distances",
        {"kind": "line", "draw": _plot_supp8c_pairwise_distances},
    ),
    # No "supp 8c median distances" entry: the bootstrap medians and their interval are
    # not drawn by the panel, so that sheet no longer exists.
    (
        "supp 9a injection site cells",
        {"kind": "scatter", "draw": _plot_supp9a_injection_site},
    ),
    (
        "supp 9b observed vs expected",
        {"kind": "bar", "draw": _plot_supp9b_observed_vs_expected},
    ),
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
        list: Paths of the written PNGs. Empty if the workbook could not be opened.
    """
    xlsx_path = Path(xlsx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sheets = read_source_data_workbook(xlsx_path)
    except Exception as e:  # an unreadable workbook must not abort the whole run
        size = xlsx_path.stat().st_size if xlsx_path.exists() else 0
        print(
            f"  [!] could not read {xlsx_path.name} ({size:,} bytes) "
            f"({type(e).__name__}: {e}); re-run its export cell"
        )
        return []
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
