"""Estimate whether double-labeling rates are consistent with a Poisson model.

In a barcoded virus injection experiment, each cell in the injection site can
be infected by 0, 1, 2, ... independent virus particles.  If infection events
are independent and occur at a constant average rate within the injection
site, the number of unique barcodes per cell should follow a Poisson
distribution.

This module provides functions to:

1. Locate the injection site center from a smoothed 3-D density map of
   barcoded cells.
2. Define the injection zone as the region where the smoothed barcoded-cell
   density exceeds a fraction (default 50 %) of its peak value, rather
   than imposing a fixed-radius sphere.
3. Count the observed distribution of barcodes per cell in fixed categories
   (0, 1, 2, 3, 4, 5+).
4. Fit a Poisson model (lambda = sample mean) and compute expected counts.
5. Perform a Chi-squared goodness-of-fit test
6. Perform one-sided parametric bootstrap excess tests for individual k
   categories and for the combined k >= 2 group.
7. Visualisation: bar charts (linear & log), Pearson residuals, and
   injection-site 2-D projections with density contour overlay.
8. Spot-count diagnostics: verify that multi-barcoded cells are not
   inflated by low-confidence barcodes by inspecting per-barcode
   transcript spot counts and re-running the analysis with minimum
   spot-count floors.
"""

from __future__ import annotations

from math import lgamma

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln, log_ndtr
from scipy.stats import chi2, poisson

from .utils import despine

# Fixed barcode-count categories used throughout the module.
# The last bin is "5+" and accumulates all cells with k >= 5.
CATEGORY_LABELS = ["0", "1", "2", "3", "4", "5+"]
N_CATEGORIES = len(CATEGORY_LABELS)  # 6


def _build_density_field(
    coords: np.ndarray,
    bin_size_mm: float = 0.1,
    smooth_sigma: float = 2.0,
):
    """Build a smoothed 3-D density histogram from point coordinates.

    Returns the smoothed array, the bin edges, and the bin size used.
    """
    bins = [
        np.arange(
            coords[:, i].min() - bin_size_mm,
            coords[:, i].max() + 2 * bin_size_mm,
            bin_size_mm,
        )
        for i in range(3)
    ]
    H, edges = np.histogramdd(coords, bins=bins)
    H_smooth = gaussian_filter(H.astype(float), sigma=smooth_sigma)
    return H_smooth, edges, bin_size_mm


def find_injection_center(
    adata,
    barcode_col: str = "n_unique_barcodes",
    coord_cols: tuple = ("ara_x", "ara_y", "ara_z"),
    bin_size_mm: float = 0.1,
    smooth_sigma: float = 2.0,
):
    """Find the 3-D density peak of barcoded cells.

    Constructs a 3-D histogram of all cells with at least one barcode,
    smooths it with a Gaussian kernel, and returns the location of the
    global maximum.

    Args:
        adata: Annotated data matrix.
        barcode_col: Column in ``adata.obs`` with unique-barcode counts
            (NaN for unbarcoded cells).
        coord_cols: (x, y, z) coordinate columns in ``adata.obs`` (mm).
        bin_size_mm: Histogram bin width in mm.
        smooth_sigma: Gaussian kernel sigma in *bins*.

    Returns:
        numpy.ndarray of shape ``(3,)`` -- injection center (mm).
    """
    obs = adata.obs
    mask = obs[barcode_col].notna() & (obs[barcode_col] > 0)
    coords = obs.loc[mask, list(coord_cols)].values

    H_smooth, edges, bs = _build_density_field(
        coords, bin_size_mm=bin_size_mm, smooth_sigma=smooth_sigma
    )
    peak_idx = np.unravel_index(H_smooth.argmax(), H_smooth.shape)
    center = np.array([edges[i][peak_idx[i]] + bs / 2 for i in range(3)])
    return center


def select_cells_in_density_region(
    adata,
    density_threshold: float = 0.5,
    barcode_col: str = "n_unique_barcodes",
    coord_cols: tuple = ("ara_x", "ara_y", "ara_z"),
    bin_size_mm: float = 0.1,
    smooth_sigma: float = 2.0,
):
    """Select all cells inside the region where barcoded-cell density
    exceeds *density_threshold* x peak density.

    This replaces a fixed-radius sphere with a data-driven contour that
    naturally adapts to the shape and extent of the injection site.

    Both barcoded and unbarcoded cells falling within the selected voxels
    are returned, because the zeros are needed to anchor the Poisson fit.

    Args:
        adata: Full annotated data matrix.
        density_threshold: Fraction of peak smoothed density that defines
            the region boundary (default 0.5 = 50 %).
        barcode_col: Barcode-count column.
        coord_cols: Coordinate columns (mm).
        bin_size_mm: Histogram bin width (mm).
        smooth_sigma: Smoothing kernel sigma (in bins).

    Returns:
        tuple:
            - **adata_region** -- AnnData copy restricted to cells in the
              region.
            - **region_info** -- dict with ``center``, ``peak_density``,
              ``threshold_density``, ``n_voxels``,
              ``approx_volume_mm3``.
    """
    obs = adata.obs
    barcoded_mask = obs[barcode_col].notna() & (obs[barcode_col] > 0)
    bc_coords = obs.loc[barcoded_mask, list(coord_cols)].values

    H_smooth, edges, bs = _build_density_field(
        bc_coords, bin_size_mm=bin_size_mm, smooth_sigma=smooth_sigma
    )

    peak_val = H_smooth.max()
    cutoff = density_threshold * peak_val

    # Identify which voxels are above the threshold
    above = H_smooth >= cutoff
    n_voxels = int(above.sum())

    # For every cell (barcoded or not), determine its voxel index
    all_coords = obs[list(coord_cols)].values
    bin_indices = np.empty_like(all_coords, dtype=int)
    for d in range(3):
        bin_indices[:, d] = np.searchsorted(edges[d][1:], all_coords[:, d])
        bin_indices[:, d] = np.clip(bin_indices[:, d], 0, H_smooth.shape[d] - 1)

    cell_in_region = above[bin_indices[:, 0], bin_indices[:, 1], bin_indices[:, 2]]

    adata_region = adata[cell_in_region].copy()

    # Injection center = peak voxel centre
    peak_idx = np.unravel_index(H_smooth.argmax(), H_smooth.shape)
    center = np.array([edges[i][peak_idx[i]] + bs / 2 for i in range(3)])

    region_info = dict(
        center=center,
        peak_density=float(peak_val),
        threshold_density=float(cutoff),
        density_threshold_frac=float(density_threshold),
        n_voxels=n_voxels,
        approx_volume_mm3=float(n_voxels * bs**3),
        bin_size_mm=bs,
        smooth_sigma=smooth_sigma,
    )
    return adata_region, region_info


# Legacy helper kept for backwards compatibility / radius-sweep use.
def select_cells_in_sphere(
    adata,
    center,
    radius_mm: float = 0.4,
    coord_cols: tuple = ("ara_x", "ara_y", "ara_z"),
):
    """Select all cells within a sphere (kept for radius-sweep)."""
    coords = adata.obs[list(coord_cols)].values
    dists = np.linalg.norm(coords - np.asarray(center), axis=1)
    return adata[dists <= radius_mm].copy()


def observed_barcode_counts(
    adata_region,
    barcode_col: str = "n_unique_barcodes",
    starter_col: str = "is_starter",
    assume_neuron_density: float | None = 155_000,
    region_volume_mm3: float | None = None,
):
    """Count cells with 0, 1, 2, 3, 4, and 5+ barcodes.

    Starter cells (identified by *starter_col*) are excluded so that
    only presynaptic cells contribute to the distribution.

    The zero-barcode count is determined by *assume_neuron_density*:
    if provided (together with *region_volume_mm3*), the total number
    of neurons in the region is estimated as
    ``assume_neuron_density * region_volume_mm3``, and the zero count
    is set to ``n_neurons - sum(k>=1 counts)``.  This avoids inflating
    the zero bin with non-neuronal cells (glia, endothelial, etc.)
    that cannot be infected by the virus.  If *assume_neuron_density*
    is ``None``, cells with NaN in *barcode_col* are treated as 0 (the
    original behaviour).

    Args:
        adata_region: AnnData restricted to the injection region.
        barcode_col: Column with per-cell barcode counts.
        starter_col: Boolean column identifying starter cells.
            Starter cells are excluded from counting.
        assume_neuron_density: Neurons per mm3.  ``None`` to fall
            back to counting all cells.
        region_volume_mm3: Volume of the region in mm3 (required
            when *assume_neuron_density* is not ``None``).

    Returns:
        numpy.ndarray of length 6 (indices 0-4 are exact counts;
        index 5 is the >= 5 bin).
    """
    obs = adata_region.obs

    # Exclude starter cells
    if starter_col in obs.columns:
        # is_starter may be categorical with string values
        is_starter = obs[starter_col].astype(str) == "True"
    else:
        is_starter = pd.Series(False, index=obs.index)

    raw = obs.loc[~is_starter, barcode_col].fillna(0)
    raw = raw.astype(int).values
    raw = np.clip(raw, 0, None)

    counts = np.zeros(N_CATEGORIES, dtype=int)
    for k in range(1, 5):
        counts[k] = int((raw == k).sum())
    counts[5] = int((raw >= 5).sum())

    # Zero-barcode count
    if assume_neuron_density is not None and region_volume_mm3 is not None:
        n_neurons = int(round(assume_neuron_density * region_volume_mm3))
        n_barcoded = int(counts[1:].sum())
        counts[0] = max(0, n_neurons - n_barcoded)
    else:
        counts[0] = int((raw == 0).sum())

    return counts


def poisson_expected_counts(observed):
    """Compute Poisson-expected counts for the fixed 0-4, 5+ categories.

    Lambda is estimated as the sample-mean barcodes per cell (MLE).

    Args:
        observed: Length-6 array from :func:`observed_barcode_counts`.

    Returns:
        (expected, lambda_hat) where *expected* has the same shape as
        *observed*.
    """
    n_total = int(observed.sum())
    # Categories 0-4 contribute k*count; the 5+ bin uses a
    # conservative lower bound of 5 per cell (exact value barely
    # matters because the bin is tiny).
    k_vals = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    lambda_hat = float((k_vals * observed).sum() / n_total)

    probs = np.array([poisson.pmf(k, lambda_hat) for k in range(5)])
    prob_5plus = max(0.0, 1.0 - probs.sum())
    all_probs = np.append(probs, prob_5plus)

    expected = all_probs * n_total
    return expected, lambda_hat


def _log10_chi2_sf(x, df):
    """Compute log10 of the chi-squared survival function.

    ``scipy.stats.chi2.sf`` and ``chi2.logsf`` both underflow to 0 / -inf
    for very large test statistics (chi2 > ~1400 with df=2).  This
    helper uses exact analytical formulae for df=1 and df=2, and a
    leading-term asymptotic expansion for higher df, so it returns a
    finite log10(p) even when the p-value itself is far below 1e-308.
    """
    a = df / 2.0
    z = x / 2.0

    if df == 2:
        # chi2(2) is Exp(1/2): sf(x) = exp(-x/2)
        return float(-z / np.log(10))

    if df == 1:
        # chi2(1) sf = 2 * Phi(-sqrt(x));  log_ndtr is stable
        return float((np.log(2) + log_ndtr(-np.sqrt(x))) / np.log(10))

    # General leading asymptotic term of the upper incomplete gamma:
    #   Q(a, z) ~ z^(a-1) * exp(-z) / Gamma(a)   for z >> a
    log_p = -z + (a - 1) * np.log(z) - gammaln(a)
    return float(log_p / np.log(10))


def chi_squared_test(observed, expected, min_expected: float = 5.0):
    """Chi-squared GOF test with bin pooling.

    Bins with expected count < *min_expected* are merged with neighbours
    (from the tail inward).  Degrees of freedom = n_bins - 2 (one for
    the total-count constraint, one for the estimated lambda).

    Returns:
        dict with ``chi2``, ``df``, ``p_value``, ``observed_binned``,
        ``expected_binned``, ``bin_labels``.
    """
    obs_b, exp_b, labels = _merge_bins(observed, expected, min_expected)
    n_bins = len(obs_b)
    df = max(n_bins - 2, 1)
    chi2_stat = float(np.sum((obs_b - exp_b) ** 2 / exp_b))
    log10_p = _log10_chi2_sf(chi2_stat, df)
    p_value = float(chi2.sf(chi2_stat, df))
    return dict(
        chi2=chi2_stat,
        df=df,
        p_value=p_value,
        log10_p_value=log10_p,
        observed_binned=obs_b,
        expected_binned=exp_b,
        bin_labels=labels,
    )


def _merge_bins(observed, expected, min_expected=5.0):
    """Pool bins from the tail inward until every bin has
    expected >= *min_expected*."""
    obs_l, exp_l, lab_l = [], [], []
    obs_acc, exp_acc = 0.0, 0.0
    start_k = None

    for k in range(len(observed)):
        if start_k is None:
            start_k = k
        obs_acc += observed[k]
        exp_acc += expected[k]

        if exp_acc >= min_expected:
            lab = (
                CATEGORY_LABELS[start_k]
                if start_k == k
                else (CATEGORY_LABELS[start_k] + "-" + CATEGORY_LABELS[k])
            )
            obs_l.append(obs_acc)
            exp_l.append(exp_acc)
            lab_l.append(lab)
            obs_acc, exp_acc = 0.0, 0.0
            start_k = None

    # Remaining tail -> merge into the last bin
    if obs_acc > 0 or exp_acc > 0:
        if obs_l:
            obs_l[-1] += obs_acc
            exp_l[-1] += exp_acc
            prev = lab_l[-1].split("-")[0]
            lab_l[-1] = ">=" + prev
        else:
            lab_l.append(">=" + CATEGORY_LABELS[start_k])
            obs_l.append(obs_acc)
            exp_l.append(exp_acc)

    return np.array(obs_l), np.array(exp_l), lab_l


def _log_factorial(k: int) -> float:
    """log(k!) via gammaln -- avoids overflow for large k."""
    return lgamma(k + 1)


def _poisson_prob(lam: float, k: int) -> float:
    """Stable Poisson PMF via log-space."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return float(np.exp(-lam + k * np.log(lam) - _log_factorial(k)))


def bootstrap_excess_test(
    observed,
    lambda_hat: float,
    n_boot: int = 5_000,
    categories: tuple = (2, 3, 4),
    random_state: int | None = 0,
):
    """One-sided parametric bootstrap for *excess* of specific
    barcode-count categories.

    For each target category *k* the test statistic is the *surplus*::

        S_k = observed_k - expected_k(lambda_hat)

    More-positive values indicate excess.  For each bootstrap
    replicate we draw *n* cells from Poisson(lambda_hat), refit lambda,
    and recompute S_k.  The one-sided p-value is the fraction of
    replicates with S_k >= the observed surplus.

    A combined test across all *categories* is also reported.

    Args:
        observed: Length-6 array (0, 1, 2, 3, 4, 5+).
        lambda_hat: Estimated Poisson lambda from the data.
        n_boot: Number of bootstrap replicates.
        categories: Which individual *k* values to test.
        random_state: RNG seed (None for non-deterministic).

    Returns:
        pandas.DataFrame with one row per tested category plus one
        "combined" row.
    """
    rng = np.random.default_rng(random_state)
    n = int(observed.sum())
    cats = tuple(int(k) for k in categories)

    # Observed surplus
    obs_stats = {}
    for k in cats:
        idx = min(k, 5)
        exp_k = n * _poisson_prob(lambda_hat, k)
        obs_stats[k] = float(observed[idx]) - exp_k

    # Combined
    obs_combined = sum(float(observed[min(k, 5)]) for k in cats)
    obs_combined_exp = n * sum(_poisson_prob(lambda_hat, k) for k in cats)
    obs_stats["combined"] = obs_combined - obs_combined_exp

    # Bootstrap replicates
    sim_stats = {k: np.empty(n_boot) for k in list(cats) + ["combined"]}

    for b in range(n_boot):
        sim = rng.poisson(lam=lambda_hat, size=n)
        lam_b = float(sim.mean())

        comb_count_b = 0
        comb_exp_b = 0.0
        for k in cats:
            count_b = int((sim == k).sum())
            exp_b = n * _poisson_prob(lam_b, k)
            sim_stats[k][b] = count_b - exp_b
            comb_count_b += count_b
            comb_exp_b += exp_b
        sim_stats["combined"][b] = comb_count_b - comb_exp_b

    # Assemble results
    rows = []
    for k in cats:
        idx = min(k, 5)
        exp_k = n * _poisson_prob(lambda_hat, k)
        p = float((1 + np.sum(sim_stats[k] >= obs_stats[k])) / (n_boot + 1))
        rows.append(
            dict(
                category=str(k),
                observed=int(observed[idx]),
                expected=round(exp_k, 2),
                surplus=round(obs_stats[k], 2),
                p_value_excess=round(p, 4),
            )
        )

    p_comb = float(
        (1 + np.sum(sim_stats["combined"] >= obs_stats["combined"])) / (n_boot + 1)
    )
    rows.append(
        dict(
            category=("combined(" + "+".join(str(k) for k in cats) + ")"),
            observed=int(obs_combined),
            expected=round(obs_combined_exp, 2),
            surplus=round(obs_stats["combined"], 2),
            p_value_excess=round(p_comb, 4),
        )
    )
    return pd.DataFrame(rows)


def summary_table(observed, expected, lambda_hat):
    """Tidy DataFrame comparing observed and expected per category.

    Returns:
        DataFrame with columns ``category``, ``observed``, ``expected``,
        ``obs_frac``, ``exp_frac``, ``residual``.
    """
    n = observed.sum()
    df = pd.DataFrame(
        {
            "category": CATEGORY_LABELS,
            "observed": observed.astype(int),
            "expected": np.round(expected, 1),
            "obs_frac": observed / n,
            "exp_frac": expected / n,
        }
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        df["residual"] = np.where(
            expected > 0,
            (observed - expected) / np.sqrt(expected),
            0.0,
        )
    return df


def plot_observed_vs_expected(
    observed,
    expected,
    lambda_hat,
    test_result=None,
    ax=None,
    title=None,
    bar_width=0.35,
    fontsize_dict={"title": 8, "label": 7, "tick": 6, "legend": 6},
):
    """Bar chart of observed vs Poisson-expected counts."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    # Resolve font sizes
    if fontsize_dict is None:
        fontsize_dict = {}
    label_fontsize = fontsize_dict.get("label", 7)
    tick_fontsize = fontsize_dict.get("tick", 6)
    legend_fontsize = fontsize_dict.get("legend", 6)
    title_fontsize = fontsize_dict.get("title", 8)

    x = np.arange(N_CATEGORIES)
    ax.bar(
        x - bar_width / 2,
        observed,
        bar_width,
        label="Observed",
        color="steelblue",
    )
    ax.bar(
        x + bar_width / 2,
        expected,
        bar_width,
        label="Poisson expected",
        color="salmon",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS, fontsize=tick_fontsize)
    ax.set_xlabel("Unique barcodes per cell", fontsize=label_fontsize)
    ax.set_ylabel("Number of cells", fontsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    ax.legend(
        # title=f"$\\lambda$ = {lambda_hat:.4f}",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        frameon=False,
    )
    despine(ax)

    if test_result is not None:
        p = test_result["p_value"]
        c = test_result["chi2"]
        d = test_result["df"]
        ax.annotate(
            f"$\\chi^2$ = {c:.1f}, df = {d}, p = {p:.2e}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=tick_fontsize,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec="gray",
                alpha=0.8,
            ),
        )
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    return ax


def plot_observed_vs_expected_log(
    observed,
    expected,
    lambda_hat,
    test_result=None,
    ax=None,
    title=None,
    fontsize_dict={"title": 8, "label": 7, "tick": 6, "legend": 6},
):
    """Same as :func:`plot_observed_vs_expected` with log y-axis."""
    ax = plot_observed_vs_expected(
        observed,
        expected,
        lambda_hat,
        test_result=test_result,
        ax=ax,
        title=title,
        fontsize_dict=fontsize_dict,
    )

    # Resolve font sizes
    if fontsize_dict is None:
        fontsize_dict = {}
    label_fontsize = fontsize_dict.get("label", 7)

    ax.set_yscale("log")
    ax.set_ylabel("Number of cells", fontsize=label_fontsize)
    ax.set_ylim(bottom=0.5)
    return ax


def plot_residuals(
    observed,
    expected,
    ax=None,
    fontsize_dict={"title": 8, "label": 7, "tick": 6, "legend": 6},
):
    """Pearson residuals per category."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    # Resolve font sizes
    if fontsize_dict is None:
        fontsize_dict = {}
    label_fontsize = fontsize_dict.get("label", 7)
    tick_fontsize = fontsize_dict.get("tick", 6)

    x = np.arange(N_CATEGORIES)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.where(
            expected > 0,
            (observed - expected) / np.sqrt(expected),
            0.0,
        )
    colors = ["steelblue" if r >= 0 else "salmon" for r in resid]
    ax.bar(x, resid, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS, fontsize=tick_fontsize)
    ax.set_xlabel("Unique barcodes per cell", fontsize=label_fontsize)
    ax.set_ylabel(
        "Pearson residual\n(obs - exp) / sqrt(exp)",
        fontsize=label_fontsize,
    )
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    despine(ax)
    return ax


def plot_density_field(
    adata,
    coord_cols=("ara_x", "ara_y", "ara_z"),
    barcode_col="n_unique_barcodes",
    projection_axes=(0, 2),
    bin_size_mm=0.1,
    smooth_sigma=2.0,
    ax=None,
    cmap="magma",
    show_center=True,
    label_fontsize=12,
    tick_fontsize=10,
):
    """Plot a 2-D projection of the 3-D smoothed barcoded-cell density field.

    Args:
        adata: Full AnnData.
        coord_cols: Coordinate column names.
        barcode_col: Column with barcode counts.
        projection_axes: Pair of axis indices (0=x, 1=y, 2=z).
        bin_size_mm: Bin width for density.
        smooth_sigma: Smoothing sigma (bins).
        ax: Matplotlib axes.
        cmap: Colormap for the density.
        show_center: Whether to plot the red cross at the peak.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    obs = adata.obs
    barcoded_mask = obs[barcode_col].notna() & (obs[barcode_col] > 0)
    bc_coords = obs.loc[barcoded_mask, list(coord_cols)].values

    H_smooth, edges, bs = _build_density_field(
        bc_coords, bin_size_mm=bin_size_mm, smooth_sigma=smooth_sigma
    )

    i, j = projection_axes
    # Find the axis to project out
    k = [idx for idx in [0, 1, 2] if idx not in projection_axes][0]

    # Max-projection along the depth axis
    H_2d = H_smooth.max(axis=k)

    # Prepare extent for imshow
    extent = [edges[i][0], edges[i][-1], edges[j][-1], edges[j][0]]

    im = ax.imshow(
        H_2d,
        extent=extent,
        cmap=cmap,
        aspect="equal",
        origin="upper",
    )

    if show_center:
        peak_idx = np.unravel_index(H_smooth.argmax(), H_smooth.shape)
        center = np.array([edges[d][peak_idx[d]] + bs / 2 for d in range(3)])
        ax.plot(center[i], center[j], "rx", markersize=8, markeredgewidth=2)

    ax.set_xlabel(coord_cols[i] + " (mm)", fontsize=label_fontsize)
    ax.set_ylabel(coord_cols[j] + " (mm)", fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    despine(ax)
    return im


def plot_injection_site(
    adata,
    region_info,
    coord_cols=("ara_x", "ara_y", "ara_z"),
    barcode_col="n_unique_barcodes",
    projection_axes=(0, 2),
    ax=None,
    coord_range=None,
    adata_region=None,
    region_alpha=0.15,
    barcode_alpha=0.8,
    layer="both",
    show_density=False,
    density_cmap="viridis",
    barcode_inside_color="green",
    barcode_outside_color=None,
    show_center=True,
    scalebar_mm=None,
    scalebar_color="black",
    density_color=None,
    show_all_cells=True,
    density_label="Smoothing Density",
    show_colorbar=False,
    colorbar_label="Density (cells/mm³)",
    fontsize_dict={"title": 8, "label": 7, "tick": 6, "legend": 6},
):
    """2-D projection showing barcoded cells and/or the density region.

    Layers (back to front):
        1. All cells in light grey  (always drawn).
        2. Cells inside the density region in black (*region_alpha*).
        3. Barcoded cells in green (*barcode_alpha*).

    Use *layer* to control which of layers 2 and 3 are drawn:

    * ``"both"`` (default) -- density region **and** barcoded cells.
    * ``"region"`` -- density region only.
    * ``"barcoded"`` -- barcoded cells only.

    If *adata_region* is provided its ``.obs.index`` is used to
    identify region cells; otherwise the approximate equivalent-radius
    circle is drawn instead (only relevant when *layer* is ``"both"``
    or ``"region"``).

    Args:
        adata: Full annotated data matrix.
        region_info: Dict returned by ``select_cells_in_density_region``.
        coord_cols: Coordinate column names (mm).
        barcode_col: Column with unique-barcode counts.
        projection_axes: Pair of axis indices (0=x, 1=y, 2=z).
        ax: Optional Matplotlib axes.
        coord_range: Optional dict mapping axis index to (min, max)
            to exclude outlier points.
        adata_region: Optional AnnData subset for the density region.
        region_alpha: Opacity for density-region cells (0-1).
        barcode_alpha: Opacity for barcoded cells (0-1).
        layer: ``"both"``, ``"region"``, or ``"barcoded"``.
        show_density: Whether to overlay density contours.
        density_cmap: Colormap for contours (if *density_color* is None).
        barcode_inside_color: Color for barcoded cells in the region.
        barcode_outside_color: Color for barcoded cells outside the region.
            If provided, barcoded cells are split into two groups.
        show_center: Whether to plot the red cross at the peak.
        scalebar_mm: If provided, adds a scale bar of this length (mm).
        scalebar_color: Color of the scale bar.
        density_color: Single color for all contours (overrides *density_cmap*).
        show_all_cells: Whether to plot the grey background of all cells.
        density_label: Legend label for the density contours.
        show_colorbar: Whether to add a colorbar for the density.
        colorbar_label: Label for the colorbar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Resolve font sizes
    if fontsize_dict is None:
        fontsize_dict = {}

    label_fontsize = fontsize_dict.get("label", 7)
    tick_fontsize = fontsize_dict.get("tick", 6)
    legend_fontsize = fontsize_dict.get("legend", 6)
    title_fontsize = fontsize_dict.get("title", 8)

    axis_labels = list(coord_cols)
    i, j = projection_axes
    center = region_info["center"]

    all_coords = adata.obs[list(coord_cols)].values
    obs_index = adata.obs.index.values

    barcoded_mask = (
        adata.obs[barcode_col].notna() & (adata.obs[barcode_col] > 0)
    ).values

    # Build region mask from adata_region index
    if adata_region is not None:
        region_idx_set = set(adata_region.obs.index)
        region_mask = np.array(
            [idx in region_idx_set for idx in obs_index],
            dtype=bool,
        )
    else:
        region_mask = None

    # Apply coordinate range filter to remove outliers
    if coord_range is not None:
        keep = np.ones(len(all_coords), dtype=bool)
        for axis_idx, (lo, hi) in coord_range.items():
            if lo is not None:
                keep &= all_coords[:, axis_idx] >= lo
            if hi is not None:
                keep &= all_coords[:, axis_idx] <= hi
        all_coords = all_coords[keep]
        barcoded_mask = barcoded_mask[keep]
        if region_mask is not None:
            region_mask = region_mask[keep]

    # Layer 1: all cells (grey background, subsampled)
    if show_all_cells:
        n_all = len(all_coords)
        if n_all > 50_000:
            idx = np.random.default_rng(42).choice(n_all, 50_000, replace=False)
            ax.scatter(
                all_coords[idx, i],
                all_coords[idx, j],
                s=0.1,
                c="lightgrey",
                alpha=0.3,
                rasterized=True,
                label="All cells",
            )
        else:
            ax.scatter(
                all_coords[:, i],
                all_coords[:, j],
                s=0.1,
                c="lightgrey",
                alpha=0.3,
                rasterized=True,
                label="All cells",
            )

    # Layer 2: density-region cells (black, semi-transparent)
    show_region = layer in ("both", "region")
    if show_region:
        if region_mask is not None:
            reg_coords = all_coords[region_mask]
            ax.scatter(
                reg_coords[:, i],
                reg_coords[:, j],
                s=0.3,
                c="black",
                alpha=region_alpha,
                rasterized=True,
                label="Barcoded cell density",
            )
        else:
            # Fallback: draw equivalent-radius circle
            vol = region_info["approx_volume_mm3"]
            equiv_r = (3 * vol / (4 * np.pi)) ** (1.0 / 3.0)
            circle = plt.Circle(
                (center[i], center[j]),
                equiv_r,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
                linestyle="--",
                label=f"equiv. radius {equiv_r:.2f} mm",
            )
            ax.add_patch(circle)
    # Layer 3: barcoded cells
    show_barcoded = layer in ("both", "barcoded")
    if show_barcoded:
        if barcode_outside_color is not None and region_mask is not None:
            # Split into inside and outside
            in_mask = barcoded_mask & region_mask
            out_mask = barcoded_mask & (~region_mask)

            # Outside cells
            out_coords = all_coords[out_mask]
            ax.scatter(
                out_coords[:, i],
                out_coords[:, j],
                s=0.3,
                c=barcode_outside_color,
                alpha=barcode_alpha,
                rasterized=True,
                label="Barcoded Cells",
            )
            # Inside cells
            in_coords = all_coords[in_mask]
            ax.scatter(
                in_coords[:, i],
                in_coords[:, j],
                s=0.3,
                c=barcode_inside_color,
                alpha=barcode_alpha,
                rasterized=True,
                label="Included Barcoded Cells",
            )
        else:
            # Single color for all barcoded cells
            bc_coords = all_coords[barcoded_mask]
            ax.scatter(
                bc_coords[:, i],
                bc_coords[:, j],
                s=2,
                c=barcode_inside_color,
                alpha=barcode_alpha,
                rasterized=True,
                label="barcoded",
            )

    # Layer 4: Density contours
    if show_density:
        obs = adata.obs
        barcoded_mask = obs[barcode_col].notna() & (obs[barcode_col] > 0)
        bc_coords = obs.loc[barcoded_mask, list(coord_cols)].values

        # Use parameters from region_info if available, else defaults
        bs = region_info.get("bin_size_mm", 0.1)
        sigma = region_info.get("smooth_sigma", 2.0)

        H_smooth, edges, _ = _build_density_field(
            bc_coords, bin_size_mm=bs, smooth_sigma=sigma
        )
        H_smooth /= bs**3  # convert to mm3

        k = [idx for idx in [0, 1, 2] if idx not in projection_axes][0]
        H_2d = H_smooth.max(axis=k)
        peak = H_smooth.max()

        # Generate meshgrid for contour
        xx, yy = np.meshgrid(edges[i][:-1] + bs / 2, edges[j][:-1] + bs / 2)

        contour_kwargs = dict(
            levels=np.array([0.25, 0.5, 0.75, 0.9]) * peak,
            linewidths=1,
        )
        if density_color:
            contour_kwargs["colors"] = density_color
        else:
            contour_kwargs["cmap"] = density_cmap

        cs = ax.contour(xx, yy, H_2d, **contour_kwargs)
        if density_label:
            # Proxy artist for the legend
            ax.plot([], [], color=density_color or "gray", lw=1.5, label=density_label)

        if show_colorbar:
            import matplotlib as mpl

            norm = mpl.colors.Normalize(vmin=0, vmax=H_2d.max())
            sm = plt.cm.ScalarMappable(cmap=density_cmap, norm=norm)
            sm.set_array([])

            fig = ax.get_figure()
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.05)
            cbar.set_label(colorbar_label, fontsize=label_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize)
            despine(cbar.ax)

    # Center marker
    if show_center:
        ax.plot(center[i], center[j], "rx", markersize=8, markeredgewidth=2)

    # Scalebar
    if scalebar_mm:
        # Calculate length in axes fraction
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax_width = abs(xlim[1] - xlim[0])
        ax_height = abs(ylim[1] - ylim[0])

        length_frac = scalebar_mm / ax_width
        height_frac = 0.02  # Fixed relative thickness

        # Position at bottom right with 5% margin
        x0 = 0.95 - length_frac
        y0 = 0.05

        import matplotlib.patches as patches

        rect = patches.Rectangle(
            (x0, y0),
            length_frac,
            height_frac,
            transform=ax.transAxes,
            facecolor=scalebar_color,
            edgecolor=None,
            linewidth=0,
            zorder=15,
            clip_on=False,
        )
        ax.add_patch(rect)

    ax.set_xlabel(axis_labels[i] + " (mm)", fontsize=label_fontsize)
    ax.set_ylabel(axis_labels[j] + " (mm)", fontsize=label_fontsize)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, frameon=False)
    despine(ax)
    return ax


def sweep_density_thresholds(
    adata,
    thresholds=None,
    barcode_col="n_unique_barcodes",
    starter_col="is_starter",
    assume_neuron_density: float | None = 155_000,
    coord_cols=("ara_x", "ara_y", "ara_z"),
    bin_size_mm=0.1,
    smooth_sigma=2.0,
):
    """Repeat the analysis across a range of density-threshold fractions.

    Args:
        thresholds: Iterable of fractions (default 0.1-0.9 in steps
            of 0.1).
        starter_col: Passed to :func:`observed_barcode_counts`.
        assume_neuron_density: Passed to
            :func:`observed_barcode_counts`.

    Returns:
        pandas.DataFrame with one row per threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.1)

    rows = []
    for th in thresholds:
        region, info = select_cells_in_density_region(
            adata,
            density_threshold=th,
            barcode_col=barcode_col,
            coord_cols=coord_cols,
            bin_size_mm=bin_size_mm,
            smooth_sigma=smooth_sigma,
        )
        obs = observed_barcode_counts(
            region,
            barcode_col=barcode_col,
            starter_col=starter_col,
            assume_neuron_density=assume_neuron_density,
            region_volume_mm3=info["approx_volume_mm3"],
        )
        exp, lam = poisson_expected_counts(obs)
        test = chi_squared_test(obs, exp)
        n_barcoded = int(obs[1:].sum())
        rows.append(
            dict(
                threshold=round(float(th), 2),
                n_neurons=int(obs.sum()),
                n_barcoded=n_barcoded,
                approx_vol_mm3=round(info["approx_volume_mm3"], 4),
                lambda_hat=lam,
                chi2=test["chi2"],
                df=test["df"],
                p_value=test["p_value"],
                log10_p_value=test["log10_p_value"],
            )
        )
    return pd.DataFrame(rows)


def plot_threshold_sweep(sweep_df, axes=None, label_fontsize=12, tick_fontsize=10):
    """Two-panel plot: lambda and p-value vs density threshold."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axes

    ax0.plot(
        sweep_df["threshold"],
        sweep_df["lambda_hat"],
        "o-",
        color="steelblue",
    )
    ax0.set_xlabel("Density threshold", fontsize=label_fontsize)
    ax0.set_ylabel("Estimated lambda", fontsize=label_fontsize)
    ax0.tick_params(labelsize=tick_fontsize)
    despine(ax0)

    col = "log10_p_value" if "log10_p_value" in sweep_df.columns else None
    if col is not None:
        ax1.plot(
            sweep_df["threshold"],
            sweep_df[col],
            "o-",
            color="salmon",
        )
        ax1.axhline(
            np.log10(0.05),
            ls="--",
            color="grey",
            lw=0.8,
            label="p = 0.05",
        )
        ax1.set_ylabel("$\\log_{10}$(p-value)", fontsize=label_fontsize)
    else:
        ax1.semilogy(
            sweep_df["threshold"],
            sweep_df["p_value"],
            "o-",
            color="salmon",
        )
        ax1.axhline(0.05, ls="--", color="grey", lw=0.8, label="p = 0.05")
        ax1.set_ylabel("p-value (log)", fontsize=label_fontsize)
    ax1.set_xlabel("Density threshold", fontsize=label_fontsize)
    ax1.tick_params(labelsize=tick_fontsize)
    ax1.legend(fontsize=tick_fontsize, frameon=False)
    despine(ax1)
    return ax0, ax1


def load_spots_per_barcode(cells_df_path):
    """Load the cell-barcode DataFrame and return spot-count info.

    The DataFrame is expected to be a pickle produced by
    ``load_cell_barcode_data`` (or the upstream barcode assignment
    pipeline).  It must contain at least:

    * ``all_barcodes`` -- list of barcode sequences per cell
    * ``n_spots_per_barcode`` -- list of spot counts (same order)
    * ``n_unique_barcodes`` -- number of unique barcodes per cell

    Args:
        cells_df_path: Path to the ``.pkl`` file.

    Returns:
        pandas.DataFrame filtered to barcoded cells (``all_barcodes``
        not null).
    """
    cells_df = pd.read_pickle(cells_df_path)
    cells_df = cells_df[cells_df["all_barcodes"].notna()]
    return cells_df


def spot_count_summary(cells_df):
    """Summarise per-barcode spot counts for single- and multi-barcoded
    cells.

    Returns:
        dict with keys:

        * ``single_spots`` -- 1-D array of spot counts for
          single-barcoded cells (one value per cell).
        * ``multi_spots_all`` -- 1-D array of spot counts for every
          individual barcode in multi-barcoded cells.
        * ``multi_min_spots`` -- 1-D array of the *minimum* spot
          count per multi-barcoded cell.
        * ``summary_df`` -- DataFrame with descriptive statistics.
    """
    single = cells_df[cells_df["n_unique_barcodes"] == 1]
    multi = cells_df[cells_df["n_unique_barcodes"] >= 2]

    single_spots = np.array([s[0] for s in single["n_spots_per_barcode"]])
    multi_all = []
    for sl in multi["n_spots_per_barcode"]:
        multi_all.extend(sl)
    multi_all = np.array(multi_all)
    multi_min = np.array([min(s) for s in multi["n_spots_per_barcode"]])

    def _desc(arr, label):
        return dict(
            group=label,
            n=len(arr),
            min=int(arr.min()) if len(arr) else 0,
            median=float(np.median(arr)) if len(arr) else 0,
            mean=round(float(arr.mean()), 1) if len(arr) else 0,
            max=int(arr.max()) if len(arr) else 0,
        )

    sdf = pd.DataFrame(
        [
            _desc(single_spots, "single-barcoded (all)"),
            _desc(multi_all, "multi-barcoded (all barcodes)"),
            _desc(multi_min, "multi-barcoded (weakest bc)"),
        ]
    )
    return dict(
        single_spots=single_spots,
        multi_spots_all=multi_all,
        multi_min_spots=multi_min,
        summary_df=sdf,
    )


def plot_spot_count_distributions(
    spot_info,
    ax=None,
    label_fontsize=12,
    tick_fontsize=10,
    max_spots=30,
):
    """Histogram comparing spot counts for single- vs multi-barcoded
    cells.

    Shows three overlapping distributions:
    * All barcodes in single-barcoded cells (blue).
    * All individual barcodes in multi-barcoded cells (orange).
    * Minimum (weakest) barcode per multi-barcoded cell (red).

    Args:
        spot_info: dict from :func:`spot_count_summary`.
        max_spots: Clip histogram x-axis at this value.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    bins = np.arange(0.5, max_spots + 1.5, 1)
    kw = dict(bins=bins, density=True, alpha=0.5, edgecolor="white")

    ax.hist(
        np.clip(spot_info["single_spots"], 0, max_spots),
        label="Single-barcoded cells",
        color="steelblue",
        **kw,
    )
    ax.hist(
        np.clip(spot_info["multi_spots_all"], 0, max_spots),
        label="Multi-barcoded (all barcodes)",
        color="orange",
        **kw,
    )
    ax.hist(
        np.clip(spot_info["multi_min_spots"], 0, max_spots),
        label="Multi-barcoded (weakest barcode)",
        color="red",
        **kw,
    )
    ax.set_xlabel("Spots per barcode", fontsize=label_fontsize)
    ax.set_ylabel("Density", fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(fontsize=tick_fontsize - 1, frameon=False)
    despine(ax)
    return ax


def recount_barcodes_with_spot_floor(cells_df, adata, min_spots=5):
    """Recount n_unique_barcodes after discarding barcodes with fewer
    than *min_spots* transcript spots.

    Creates (or overwrites) the column
    ``n_unique_barcodes_minN`` in ``adata.obs`` where N = *min_spots*.

    Args:
        cells_df: DataFrame from :func:`load_spots_per_barcode`.
        adata: The AnnData object (modified in place).
        min_spots: Minimum spots for a barcode to count.

    Returns:
        str -- the name of the new column added to ``adata.obs``.
    """
    col_name = f"n_unique_barcodes_min{min_spots}"

    # Compute filtered count from cells_df
    filtered = {}
    for idx, row in cells_df.iterrows():
        spots = row["n_spots_per_barcode"]
        n_above = sum(1 for s in spots if s >= min_spots)
        filtered[idx] = n_above

    filtered_series = pd.Series(filtered, name=col_name)

    # Map onto adata; unbarcoded cells stay NaN
    adata.obs[col_name] = filtered_series.reindex(adata.obs.index)
    # Cells that were barcoded but now have 0 qualifying barcodes
    # should be treated as 0 (not NaN) for the Poisson analysis
    mask = adata.obs["n_unique_barcodes"].notna()
    adata.obs.loc[mask, col_name] = adata.obs.loc[mask, col_name].fillna(0)
    return col_name


def run_filtered_analysis_comparison(
    adata,
    cells_df,
    spot_floors=(3, 5, 10),
    density_threshold=0.5,
    starter_col="is_starter",
    assume_neuron_density: float | None = 155_000,
    n_boot=5_000,
    random_state=0,
    verbose=True,
):
    """Run the double-labeling analysis at several spot-count floors
    and return a comparison table.

    For each floor in *spot_floors*, barcodes with fewer than that
    many spots are discarded before recounting
    ``n_unique_barcodes``, and the full Poisson pipeline is re-run.

    Starter cells are excluded from barcode counting
    (see :func:`observed_barcode_counts`).

    Args:
        adata: Full AnnData (will be modified in place to add
            filtered barcode-count columns).
        cells_df: DataFrame from :func:`load_spots_per_barcode`.
        spot_floors: Iterable of minimum-spot thresholds to test.
        density_threshold: Passed to ``run_double_labeling_analysis``.
        starter_col: Passed to ``run_double_labeling_analysis``.
        assume_neuron_density: Passed to
            ``run_double_labeling_analysis``.
        n_boot: Bootstrap replicates.
        random_state: RNG seed.
        verbose: Print per-threshold summary.

    Returns:
        dict mapping each floor to its ``run_double_labeling_analysis``
        results dict.  Also adds a ``"comparison_df"`` key with a
        single summary DataFrame.
    """
    all_results = {}
    rows = []

    for floor in spot_floors:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Spot floor = {floor} " "(minimum spots per barcode to count)")
            print(f"{'='*60}")

        if floor <= 3:
            # Original data already has a floor of 3
            bc_col = "n_unique_barcodes"
        else:
            bc_col = recount_barcodes_with_spot_floor(cells_df, adata, min_spots=floor)

        res = run_double_labeling_analysis(
            adata,
            density_threshold=density_threshold,
            barcode_col=bc_col,
            starter_col=starter_col,
            assume_neuron_density=assume_neuron_density,
            n_boot=n_boot,
            random_state=random_state,
            verbose=verbose,
        )
        all_results[floor] = res

        obs = res["observed"]
        rows.append(
            dict(
                min_spots=floor,
                barcode_col=bc_col,
                n_neurons=int(obs.sum()),
                n_barcoded=int(obs[1:].sum()),
                n_multi=int(obs[2:].sum()),
                lambda_hat=res["lambda_hat"],
                chi2=res["chi2_test"]["chi2"],
                df=res["chi2_test"]["df"],
                log10_p=res["chi2_test"]["log10_p_value"],
            )
        )

    all_results["comparison_df"] = pd.DataFrame(rows)
    return all_results


def plot_filtered_comparison(
    all_results,
    spot_floors=(3, 5, 10),
    label_fontsize=12,
    tick_fontsize=10,
):
    """Three-column figure: one column per spot floor, each showing
    observed vs expected (log scale).

    Args:
        all_results: dict from :func:`run_filtered_analysis_comparison`.
        spot_floors: Which floors to plot (must be keys in
            *all_results*).

    Returns:
        (fig, axes)
    """
    n = len(spot_floors)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, floor in zip(axes, spot_floors):
        res = all_results[floor]
        plot_observed_vs_expected_log(
            res["observed"],
            res["expected"],
            res["lambda_hat"],
            test_result=res["chi2_test"],
            ax=ax,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            title=f"Min {floor} spots/barcode",
        )
    fig.tight_layout()
    return fig, axes


def run_double_labeling_analysis(
    adata,
    density_threshold: float = 0.5,
    coord_cols=("ara_x", "ara_y", "ara_z"),
    barcode_col="n_unique_barcodes",
    starter_col="is_starter",
    assume_neuron_density: float | None = 155_000,
    bin_size_mm=0.1,
    smooth_sigma=2.0,
    excess_categories=(2, 3, 4),
    n_boot=5_000,
    random_state=0,
    verbose=True,
):
    """Run the full double-labeling analysis pipeline.

    Steps:
        1. Detect injection center.
        2. Select cells inside the >= *density_threshold* x peak density
           contour.
        3. Count barcodes per cell in categories 0, 1, 2, 3, 4, 5+
           (excluding starter cells; zero count from assumed neuron
           density).
        4. Fit Poisson (lambda = mean) and compute expected counts.
        5. Chi-squared GOF test (with bin pooling, df = n_bins - 2).
        6. Bootstrap excess tests for specified *k* values.
        7. Summary table.

    Args:
        adata: Full AnnData.
        density_threshold: Fraction of peak density for region selection.
        coord_cols: Coordinate column names.
        barcode_col: Barcode-count column name.
        starter_col: Boolean column identifying starter cells to
            exclude.
        assume_neuron_density: Neurons per mm3 for estimating the
            zero-barcode count.  ``None`` falls back to counting all
            non-barcoded cells in the region.
        bin_size_mm: Bin width for density estimation.
        smooth_sigma: Gaussian sigma in bins.
        excess_categories: Which *k* values to bootstrap-test.
        n_boot: Number of bootstrap replicates.
        random_state: RNG seed.
        verbose: Print summary to stdout.

    Returns:
        dict with keys ``center``, ``region_info``, ``adata_region``,
        ``observed``, ``expected``, ``lambda_hat``, ``chi2_test``,
        ``excess_df``, ``summary_df``.
    """
    # 1-2.  Region selection
    adata_region, region_info = select_cells_in_density_region(
        adata,
        density_threshold=density_threshold,
        barcode_col=barcode_col,
        coord_cols=coord_cols,
        bin_size_mm=bin_size_mm,
        smooth_sigma=smooth_sigma,
    )
    center = region_info["center"]

    vol = region_info["approx_volume_mm3"]

    # 3.  Observed (starters excluded, zero count from neuron density)
    obs = observed_barcode_counts(
        adata_region,
        barcode_col=barcode_col,
        starter_col=starter_col,
        assume_neuron_density=assume_neuron_density,
        region_volume_mm3=vol,
    )

    # 4.  Expected
    exp, lambda_hat = poisson_expected_counts(obs)

    # 5.  Chi-squared GOF
    test = chi_squared_test(obs, exp)

    # 6.  Bootstrap excess
    excess = bootstrap_excess_test(
        obs,
        lambda_hat,
        n_boot=n_boot,
        categories=excess_categories,
        random_state=random_state,
    )

    # 7.  Summary
    sdf = summary_table(obs, exp, lambda_hat)

    if verbose:
        # Count presynaptic barcoded cells (exclude starters)
        _obs = adata_region.obs
        if starter_col in _obs.columns:
            _starter = _obs[starter_col].astype(str) == "True"
        else:
            _starter = pd.Series(False, index=_obs.index)
        _presyn = _obs.loc[~_starter, barcode_col].fillna(0)
        n_b = int((_presyn > 0).sum())
        n_starters = int(_starter.sum())
        n_t = int(obs.sum())  # total neurons (from density)
        equiv_r = (3 * vol / (4 * np.pi)) ** (1.0 / 3.0)
        print(
            f"Injection center: "
            f"({center[0]:.2f}, {center[1]:.2f}, "
            f"{center[2]:.2f}) mm"
        )
        print(
            f"Region: >={density_threshold:.0%} of peak density, "
            f"approx {vol:.3f} mm3 "
            f"(equiv. radius {equiv_r:.2f} mm)"
        )
        if assume_neuron_density is not None:
            print(
                f"Assumed neuron density: "
                f"{assume_neuron_density:,.0f} /mm3 "
                f"-> {n_t:,} neurons in region"
            )
        print(f"Presynaptic barcoded: {n_b:,} " f"(excluded {n_starters:,} starters)")
        print(f"lambda = {lambda_hat:.5f}")
        print(
            f"chi2 = {test['chi2']:.2f}, "
            f"df = {test['df']}, "
            f"p = {test['p_value']:.2e} "
            f"(log10 p = {test['log10_p_value']:.1f})"
        )
        print()
        print(sdf.to_string(index=False))
        print()
        print("Bootstrap excess tests:")
        print(excess.to_string(index=False))

    return dict(
        center=center,
        region_info=region_info,
        adata_region=adata_region,
        observed=obs,
        expected=exp,
        lambda_hat=lambda_hat,
        chi2_test=test,
        excess_df=excess,
        summary_df=sdf,
    )
