"""Source Data exporters for the supplementary figures.

As in :mod:`brisc.source_data.figures`, each exporter takes the variables its notebook
plots and writes one worksheet per data-bearing panel. Micrographs, atlas outlines and
the exploratory cells of those notebooks have no worksheet.
"""

import numpy as np
import pandas as pd

from brisc.source_data.figures import (
    ABUNDANCE_SUBSAMPLE_NOTE,
    _library_abundance,
    _presynaptic_density,
    _unique_fraction,
    build_long_range_source_data,
)
from brisc.source_data.io import save_excel_sheets

# ---------------------------------------------------------------------------
# Supplementary Figure 1 / 2 — presynaptic density and starter dilution
# ---------------------------------------------------------------------------

SUPP1_PANELS = [
    "Supp 1c Presynaptic density",
    "Supp 1d Starter dilution",
]


def export_suppfig2_source_data(
    output_path,
    voxel_distances_sorted=None,
    cell_distances_sorted=None,
    dilution_densities=None,
):
    """Supplementary Figure 1.

    Args:
        voxel_distances_sorted (np.ndarray): Sorted isocortex voxel distances.
        cell_distances_sorted (np.ndarray): Sorted labelled-cell distances.
        dilution_densities (pd.DataFrame): The V1 density table returned by
            `starter_cell_counting.plot_starter_dilution_densities`.
    """
    panels = {}
    if voxel_distances_sorted is not None and cell_distances_sorted is not None:
        panels["Supp 1c Presynaptic density"] = _presynaptic_density(
            voxel_distances_sorted, cell_distances_sorted
        )
    if dilution_densities is not None:
        table = pd.DataFrame(dilution_densities)
        keep = [
            c for c in ["mouse", "dilution", "count", "density"] if c in table.columns
        ]
        panels["Supp 1d Starter dilution"] = (
            table.reset_index()[keep] if keep else table.reset_index()
        )
    return save_excel_sheets(panels, output_path, expected=SUPP1_PANELS)


# ---------------------------------------------------------------------------
# Supplementary Figure 4 — barcode length
# ---------------------------------------------------------------------------

SUPP4_PANELS = [
    "Supp 4a Library abundance",
    "Supp 4b Unique fraction",
    "Supp 4c Unique vs length",
]


def export_suppfig4_source_data(
    output_path, lib2plot=None, nunique=None, barcode_lengths=None
):
    """Supplementary Figure 4.

    Args:
        lib2plot (dict): The 10 and 20 nucleotide libraries drawn in panels a and b.
        nunique (list): Infection events for 95% unique labelling, per barcode length.
        barcode_lengths (iterable): The barcode lengths matching ``nunique``.
    """
    panels = {}
    if lib2plot is not None:
        panels["Supp 4a Library abundance"] = _library_abundance(lib2plot)
        panels["Supp 4b Unique fraction"] = _unique_fraction(lib2plot, max_cells=1e6)
    if nunique is not None:
        lengths = (
            list(barcode_lengths)
            if barcode_lengths is not None
            else list(range(4, 4 + len(nunique)))
        )
        panels["Supp 4c Unique vs length"] = pd.DataFrame(
            {
                "Barcode_Length_Nucleotides": lengths,
                "Infections_For_95pc_Unique": np.asarray(nunique),
            }
        )
    notes = {"Supp 4a Library abundance": ABUNDANCE_SUBSAMPLE_NOTE}
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
    "Supp 5d Marker gene expression",
]


def export_suppfig5_source_data(
    output_path,
    mcherry_manual=None,
    mcherry_curated=None,
    starters_positions=None,
    inside_cells_df=None,
    outside_cells_df=None,
    adata=None,
    genes_to_plot=None,
    categories_order=None,
):
    """Supplementary Figure 5.

    Args:
        mcherry_manual (pd.DataFrame): Manually curated mCherry cells (chamber_06).
        mcherry_curated (pd.DataFrame): Automatically curated mCherry cells.
        starters_positions (pd.DataFrame): Starter cell positions per section.
        inside_cells_df (pd.DataFrame): Cells inside the sequenced volume.
        outside_cells_df (pd.DataFrame): Cells outside the sequenced volume.
        adata (AnnData): Object behind the marker-gene dot plot.
        genes_to_plot (list): Genes of the dot plot, in the plotted order.
        categories_order (list): Cluster order of the dot plot.
    """
    panels = {}

    if mcherry_manual is not None and mcherry_curated is not None:
        panels["Supp 5a Cells per section"] = _cells_per_section(
            mcherry_manual, mcherry_curated, starters_positions
        )

    atlas_size = 10
    if outside_cells_df is not None and inside_cells_df is not None:
        frames = []
        for label, df in (
            ("Outside volume", outside_cells_df),
            ("Inside volume", inside_cells_df),
        ):
            frames.append(
                pd.DataFrame(
                    {
                        "Cell_Group": label,
                        "ARA_X_px": df["ara_x"].values * 1000 / atlas_size,
                        "ARA_Y_px": df["ara_y"].values * 1000 / atlas_size,
                        "ARA_Z_px": df["ara_z"].values * 1000 / atlas_size,
                    }
                )
            )
        positions = pd.concat(frames, ignore_index=True)
        panels["Supp 5b Coronal positions"] = positions[
            ["Cell_Group", "ARA_Z_px", "ARA_Y_px", "ARA_X_px"]
        ]
        panels["Supp 5c Dorsal positions"] = positions[
            ["Cell_Group", "ARA_X_px", "ARA_Z_px"]
        ]

    if adata is not None and genes_to_plot is not None:
        panels["Supp 5d Marker gene expression"] = _dotplot_table(
            adata, genes_to_plot, categories_order
        )

    return save_excel_sheets(panels, output_path, expected=SUPP5_PANELS)


def _cells_per_section(mcherry_manual, mcherry_curated, starters_positions):
    """Panel a — mCherry and starter cell counts along the antero-posterior axis."""
    rows = []
    m6 = mcherry_manual.query("chamber == 'chamber_06'").groupby("roi").aggregate(len).x
    for roi, count in m6.items():
        rows.append(
            {
                "Cell_Type": "mCherry cells",
                "Chamber": "chamber_06",
                "Section_Position_um": (int(roi) - 10) * 20,
                "Cell_Count": count,
            }
        )

    offset = 10
    for chamber in ["07", "08", "09", "10"]:
        subset = mcherry_curated.query(f"chamber == 'chamber_{chamber}'")
        counts = subset.groupby("roi").aggregate(len).x
        for roi, count in counts.items():
            rows.append(
                {
                    "Cell_Type": "mCherry cells",
                    "Chamber": f"chamber_{chamber}",
                    "Section_Position_um": (roi + offset) * 20,
                    "Cell_Count": count,
                }
            )
        if starters_positions is not None:
            starters = starters_positions.query(f"chamber == 'chamber_{chamber}'")
            if len(starters):
                starter_counts = starters.groupby("roi").aggregate(len).y
                for roi, count in starter_counts.items():
                    rows.append(
                        {
                            "Cell_Type": "Starter cells",
                            "Chamber": f"chamber_{chamber}",
                            "Section_Position_um": (roi + offset) * 20,
                            "Cell_Count": count,
                        }
                    )
        offset += counts.index.max()
    return pd.DataFrame(rows)


def _dotplot_table(adata, genes, categories_order=None):
    """Panel d — mean expression and fraction of expressing cells per cluster."""
    clusters = adata.obs["custom_leiden"]
    order = (
        list(categories_order)
        if categories_order is not None
        else list(pd.unique(clusters))
    )
    rows = []
    for gene in genes:
        if gene not in adata.var_names:
            continue
        column = adata[:, gene].X
        values = np.asarray(
            column.toarray() if hasattr(column, "toarray") else column
        ).ravel()
        frame = pd.DataFrame({"Cluster": np.asarray(clusters), "Expression": values})
        for cluster, group in frame.groupby("Cluster", observed=True):
            if cluster not in order:
                continue
            rows.append(
                {
                    "Gene": gene,
                    "Cluster": cluster,
                    "Mean_Expression": group["Expression"].mean(),
                    "Fraction_Expressing": (group["Expression"] > 0).mean(),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Supplementary Figure 6 — transcriptomic validation mosaic
# ---------------------------------------------------------------------------

SUPP6_PANELS = [
    "Supp 6 Cluster positions",
    "Supp 6 Cluster depth KDE",
]


def export_suppfig6_source_data(
    output_path,
    adata=None,
    group_key="custom_leiden",
    chambers=("chamber_07",),
    clusters_not_used=("Unassigned", "Zero_correlation", "VLMC"),
    cortex_exclude=("fiber_tract", "non_cortical", "TH", "hippocampal"),
    qc=None,
    layer_tops=None,
    x_min=1970,
    x_max=2260,
    bw_method=0.1,
    atlas_size=10,
):
    """Supplementary Figure 6 — the per-cluster coronal scatter and depth KDE.

    The filtering mirrors `cell_typing.plot_cluster_mosaic` exactly, so the sheets hold
    the cells actually drawn.
    """
    panels = {}
    if adata is None:
        return save_excel_sheets(panels, output_path, expected=SUPP6_PANELS)

    if qc is None:
        qc = dict(best_score=0.3, knn_agree_conf=0.3, raw_gene_counts=2)
    if layer_tops is None:
        layer_tops = {"wm": 957.0592130899}

    obs = adata.obs
    keep = (
        ~obs[group_key].isin(list(clusters_not_used))
        & obs["chamber"].isin(list(chambers))
        & (~obs["cortical_area"].isna())
        & ~obs["cortical_area"].isin(list(cortex_exclude))
        & (obs["best_score"] > qc["best_score"])
        & (obs["knn_agree_conf"] > qc["knn_agree_conf"])
        & (obs["raw_gene_counts"] > qc["raw_gene_counts"])
    )
    plotted = obs[keep]

    panels["Supp 6 Cluster positions"] = pd.DataFrame(
        {
            "Cluster": np.asarray(plotted[group_key]),
            "ARA_Z_px": plotted["ara_z"].values * 1000 / atlas_size,
            "ARA_Y_px": plotted["ara_y"].values * 1000 / atlas_size,
        }
    )

    from scipy.stats import gaussian_kde

    depth = plotted["normalised_depth"] * (layer_tops["wm"] / 2000.0)
    flat_x = plotted["flatmap_dorsal_x"] / 10
    in_region = (
        (flat_x >= x_min)
        & (flat_x <= x_max)
        & (depth <= layer_tops["wm"])
        & (depth >= 0)
    )
    frames = []
    for cluster, group in plotted[in_region].groupby(group_key, observed=True):
        values = (group["normalised_depth"] * (layer_tops["wm"] / 2000.0)).to_numpy()
        if len(values) < 3:
            continue
        grid = np.linspace(values.min(), values.max(), 200)
        density = gaussian_kde(values, bw_method=bw_method)(grid)
        if density.max() > 0:
            density = density / density.max()
        frames.append(
            pd.DataFrame(
                {
                    "Cluster": cluster,
                    "Cortical_Depth_um": grid,
                    "Normalised_Density": density,
                }
            )
        )
    if frames:
        panels["Supp 6 Cluster depth KDE"] = pd.concat(frames, ignore_index=True)

    return save_excel_sheets(panels, output_path, expected=SUPP6_PANELS)


# ---------------------------------------------------------------------------
# Supplementary Figure 8 — barcodes in multiple starter cells
# ---------------------------------------------------------------------------

SUPP8_PANELS = [
    "Supp 8a Library abundance KDE",
    "Supp 8b Density difference",
    "Supp 8c Pairwise distances",
    "Supp 8c Median distances",
]


def export_suppfig8_source_data(
    output_path,
    x_grid_kde=None,
    dens_lib=None,
    dens_all=None,
    dens_multi=None,
    ci_low_kde=None,
    ci_up_kde=None,
    diff_all=None,
    diff_multi=None,
    ci_low_diff=None,
    ci_up_diff=None,
    dist2others=None,
    dist2same=None,
    not_adj=None,
    median_distances=None,
    p_val_same=None,
    p_val_exl=None,
):
    """Supplementary Figure 8 — all arguments are variables of the notebook."""
    panels = {}

    if x_grid_kde is not None and dens_lib is not None:
        table = pd.DataFrame(
            {
                "Log10_Library_Reads": x_grid_kde,
                "Library_Barcodes_Density": dens_lib,
                "All_In_Situ_Barcodes_Density": dens_all,
                "Multi_Starter_Barcodes_Density": dens_multi,
            }
        )
        if ci_low_kde is not None:
            table["Multi_Starter_CI_Lower"] = ci_low_kde
            table["Multi_Starter_CI_Upper"] = ci_up_kde
        panels["Supp 8a Library abundance KDE"] = table

    if x_grid_kde is not None and diff_multi is not None:
        table = pd.DataFrame(
            {
                "Log10_Library_Reads": x_grid_kde,
                "All_In_Situ_Minus_Library": diff_all,
                "Multi_Starter_Minus_Library": diff_multi,
            }
        )
        if ci_low_diff is not None:
            table["Multi_Starter_CI_Lower"] = ci_low_diff
            table["Multi_Starter_CI_Upper"] = ci_up_diff
        panels["Supp 8b Density difference"] = table

    if dist2others is not None and dist2same is not None:
        from scipy.stats import gaussian_kde

        grid = np.arange(0, 2, 0.01)
        series = {
            "Different barcode": np.asarray(dist2others),
            "Same barcode": np.asarray(dist2same),
        }
        if not_adj is not None:
            series["Same barcode, excluding adjacent"] = np.asarray(dist2same)[not_adj]
        frames = []
        for label, values in series.items():
            frames.append(
                pd.DataFrame(
                    {
                        "Comparison": label,
                        "Distance_Between_Starters_mm": grid,
                        "Density": gaussian_kde(values, bw_method=0.2)(grid),
                        "Median_Distance_mm": np.nanmedian(values),
                    }
                )
            )
        panels["Supp 8c Pairwise distances"] = pd.concat(frames, ignore_index=True)

    if median_distances is not None:
        rows = []
        p_values = {
            "med2same": p_val_same,
            "med2same_excluding_adjacent": p_val_exl,
        }
        for name, boot in median_distances.items():
            boot = np.asarray(boot)
            rows.append(
                {
                    "Comparison": name,
                    "Bootstrap_Median_mm": np.nanmedian(boot),
                    "CI_Lower_mm": np.percentile(boot, 2.5),
                    "CI_Upper_mm": np.percentile(boot, 97.5),
                    "P_Value_Vs_Different_Barcode": p_values.get(name, np.nan),
                }
            )
        panels["Supp 8c Median distances"] = pd.DataFrame(rows)

    return save_excel_sheets(panels, output_path, expected=SUPP8_PANELS)


# ---------------------------------------------------------------------------
# Supplementary Figure 9 — double labelling estimation
# ---------------------------------------------------------------------------

SUPP9_PANELS = [
    "Supp 9a Injection site cells",
    "Supp 9b Observed vs expected",
]


def export_suppfig9_source_data(output_path, results=None):
    """Supplementary Figure 9.

    Args:
        results (dict): Output of
            `double_labeling_estimation.run_double_labeling_analysis`.
    """
    panels = {}
    if results is None:
        return save_excel_sheets(panels, output_path, expected=SUPP9_PANELS)

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

    for key, sheet in (
        ("summary_df", "Supp 9 Summary"),
        ("excess_df", "Supp 9 Excess"),
    ):
        value = results.get(key)
        if isinstance(value, pd.DataFrame):
            panels[sheet] = value.reset_index()

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
    "Rev f Presynaptic elevation",
    "Rev f Elevation running avg",
]


def export_suppfig_reviewer_source_data(
    output_path,
    v1_starter_cells=None,
    presy_xy=None,
    presy_azel=None,
    starter_pos_ara_x=None,
    center_abs_x=None,
    scale=0.01,
    ctx_img=None,
    ctx_mask=None,
    xlim=(150, 1050),
    ylim=(810, 1330),
    x_calc=None,
    pres_vs_start_kde=None,
    conf_int=None,
    ele_kde=None,
    mean_position=None,
):
    """Supplementary reviewer figure — Figure 6 repeated along elevation / A-P.

    Starter position is the antero-posterior atlas coordinate (``ara_x``, in mm) rather
    than the relative medio-lateral flatmap position, and the running average is over
    receptive-field elevation.
    """

    def rel_pos_x(x):
        return -(np.asarray(x) - center_abs_x)

    starter_values = None
    if v1_starter_cells is not None and "ara_x" in v1_starter_cells.columns:
        starter_values = v1_starter_cells["ara_x"].values

    panels = build_long_range_source_data(
        prefix="Rev",
        v1_starter_cells=v1_starter_cells,
        starter_panel_values=starter_values,
        presy_xy=presy_xy,
        presyn_panel_values=starter_pos_ara_x,
        starter_value_label="Starter_AP_Position_mm",
        presynaptic_axis_values=(
            None
            if presy_xy is None or center_abs_x is None
            else rel_pos_x(presy_xy[0]) * scale
        ),
        ctx_img=ctx_img,
        ctx_mask=ctx_mask,
        xlim=xlim,
        ylim=ylim,
        running_average_x=(
            None
            if x_calc is None or center_abs_x is None
            else rel_pos_x(x_calc) * scale
        ),
        running_average=pres_vs_start_kde,
        conf_int=conf_int,
        mean_position=mean_position,
        retinotopy=None if presy_azel is None else np.asarray(presy_azel)[1],
        retinotopy_label="Receptive_Field_Elevation_deg",
        retinotopy_name="Elevation",
        retinotopy_running_average=ele_kde,
    )
    return save_excel_sheets(panels, output_path, expected=SUPP_REVIEWER_PANELS)
