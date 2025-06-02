import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from anndata import AnnData
from scipy.sparse import csr_matrix

from tqdm import trange
from tqdm import tqdm
import warnings


def df_to_adata(df: pd.DataFrame) -> AnnData:
    gene_cols = [
        c for c in df.columns if c.startswith("gene_") and c != "gene_total_counts"
    ]
    meta_cols = [
        c for c in df.columns if not c.startswith("gene_") or c == "gene_total_counts"
    ]
    X = csr_matrix(df[gene_cols].to_numpy())
    var = pd.DataFrame(index=[c.replace("gene_", "") for c in gene_cols])
    obs = df[meta_cols].copy()
    obs.index = df.index.astype(str)
    adata = AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    return adata


def load_cell_typing_data(
    processed_path,
):
    """
    Load the adata files for BRAC8498.4e and BRYC65.1d.

    Args:
        processed_path : Path
            Path to the processed data directory containing the adata files.

    Returns:
        adata_q : AnnData
            The query dataset (BRAC8498.4e) with cell barcode genes.
        adata_ref : AnnData
            The reference dataset (BRYC65.1d) with annotated leiden clusters.
    """
    # Load BRAC8498.4e data
    cell_genes = pd.read_pickle(
        processed_path / "analysis" / "cell_barcode_genes_df.pkl"
    )
    adata_q = df_to_adata(cell_genes)

    # Load BRYC65.1d reference data
    adata_ref = sc.read_h5ad(
        processed_path.parent / "BRYC65.1d/chamber_13/adata_iso_expanded.h5ad"
    )
    adata_ref.obs.index = adata_ref.obs.barcode_index

    # Annotate leiden clusters with cluster names
    adata_ref.obs["custom_leiden"] = adata_ref.obs.leiden.map(
        lambda x: {
            "0": "L2/3 IT 2",
            "1": "L4 IT",  #
            "2": "L6 CT",  # High Crym
            "3": "L5 IT",
            "4": "L5/6 IT",
            "5": "L2/3 RSP",  #
            "6": "Pvalb",  # High Pvalb, Pcp4l1, Kcnc2 -
            "7": "L5 PT",  # Was "L5 NP",
            "8": "Sst",  # High Sst -
            "9": "L2/3 IT 1",
            "10": "Unassigned",  # High Enpp2 but likely not Sncg
            "11": "Lamp5",
            "12": "Vip",  # High Vip -
            "13": "L4 RSP",  #    "L5 PT", # High Cplx1, Nefl, Pak1, Pde1a, location -
            "14": "L6b",  # High Ctgf, location -
            "15": "Car3",  # High Nr4a2 and Snypr -
            "16": "VLMC",  # High Nnat and Ptn, location -
            "17": "L5 NP",  # High Pcp4, Rgs4, Nrn1 Was L4/5 IT labeled before
            "18": "Unassigned",  # This was a small cluster with no markers
        }.get(x, x)
    ).astype("category")
    # Define a custom color map for Leiden clusters
    cluster_colors = cluster_colors = {
        "L2/3 IT 1": "#d62728",
        "L2/3 IT 2": "#ff7f0e",
        "L2/3 RSP": "#ffbb78",
        "L4 IT": "#dbdb8d",
        "L4 RSP": "#bcbd22",
        "L5 NP": "#98df8a",
        "L5 PT": "#2ca02c",
        "L5 IT": "palegreen",  # aec7e8
        "L5/6 IT": "#17becf",
        "L6 CT": "#c5b0d5",
        "L6b": "#9467bd",
        "Car3": "#e377c2",
        "Lamp5": "sienna",  # 8c564b
        "Pvalb": "#f7b6d2",
        "Sst": "darkslateblue",
        "Vip": "darkviolet",
        "Unassigned": "grey",
        "VLMC": "black",
    }
    adata_ref.obs["custom_colors"] = adata_ref.obs["custom_leiden"].map(cluster_colors)

    # Take only the intersection of observed genes
    shared_genes = adata_ref.var_names.intersection(adata_q.var_names)
    adata_ref = adata_ref[:, shared_genes].copy()
    adata_q = adata_q[:, shared_genes].copy()
    adata_q.raw = adata_q.copy()
    # Normalise
    sc.pp.normalize_total(adata_q, target_sum=10)
    sc.pp.log1p(adata_q)

    return adata_q, adata_ref


def compute_pca_centroids_knn(
    adata_q,
    adata_ref,
):
    """
    Compute PCA, KNN classification and nearest centroid classification
    for the query dataset (adata_q) using the reference dataset (adata_ref).

    Args:
        adata_q : AnnData
            The query dataset with cell barcode genes.
        adata_ref : AnnData
            The reference dataset with annotated leiden clusters.

    Returns:
        adata_q : AnnData
            The query dataset with PCA projections, KNN predictions, and centroid distances.
    """
    # project query cells into reference PCA space
    n_pcs = 30
    pca = PCA(n_components=n_pcs)
    ref_mat = adata_ref.X
    X_ref_pcs = pca.fit_transform(ref_mat)
    q_mat = adata_q.X
    X_q_pcs = pca.transform(q_mat)
    adata_q.obsm["X_pca"] = X_q_pcs

    # KNN classification
    knn = KNeighborsClassifier(n_neighbors=30, weights="distance")
    knn.fit(adata_ref.obsm["X_pca"], adata_ref.obs["custom_leiden"])
    adata_q.obs["knn_pred"] = knn.predict(adata_q.obsm["X_pca"])
    # distance‐based confidence: 1 / (1 + mean(distance_to_knn))
    dists, idxs = knn.kneighbors(
        adata_q.obsm["X_pca"], n_neighbors=30, return_distance=True
    )
    mean_dist = dists.mean(axis=1)
    adata_q.obs["knn_dist_conf"] = 1 / (1 + mean_dist)
    # agreement‐based confidence: fraction of the NN sharing the predicted label
    neighbor_labels = np.array(adata_ref.obs["custom_leiden"])[idxs]
    agree_frac = (neighbor_labels == adata_q.obs["knn_pred"].values[:, None]).sum(
        axis=1
    ) / idxs.shape[1]
    adata_q.obs["knn_agree_conf"] = agree_frac

    # Nearest‐centroid classification + distances to all centroids
    celltypes = adata_ref.obs["custom_leiden"].unique()
    centroids = {
        ct: X_ref_pcs[adata_ref.obs["custom_leiden"] == ct].mean(axis=0)
        for ct in celltypes
    }
    centroid_mat = np.vstack([centroids[ct] for ct in celltypes])

    # compute Euclidean distance from each query cell to each centroid (n_query × n_celltypes)
    d2c = np.sqrt(((X_q_pcs[:, None, :] - centroid_mat[None, :, :]) ** 2).sum(axis=2))
    for i, ct in enumerate(celltypes):
        adata_q.obs[f"dist_to_centroid_{ct}"] = d2c[:, i]

    # nearest centroid
    nearest_idx = d2c.argmin(axis=1)
    adata_q.obs["centroid_pred"] = celltypes[nearest_idx]
    adata_q.obs["centroid_pred_dist"] = d2c[np.arange(d2c.shape[0]), nearest_idx]
    adata_q.obs["dist_to_nearest_centroid"] = d2c.min(axis=1)

    return adata_q


def compute_cluster_mean_correlations(adata_q, cluster_means):
    """
    Compute per-cell correlations with cluster means.

    Args:
        adata_q : AnnData
            The query dataset with cell barcode genes.
        cluster_means : DataFrame (n_clusters x n_genes) Cluster means of reference data

    Returns:
        adata_q : AnnData
            The query dataset with additional columns for best cluster, best score, median score, and delta score.
    """
    expression_matrix = adata_q.X.todense()
    correlation_matrix_q = pd.DataFrame(
        index=adata_q.obs.index, columns=cluster_means.index
    )

    # Iterate over each cell to calculate the correlation with each clusters
    for cell_idx, cell_expression in tqdm(
        enumerate(expression_matrix), total=len(expression_matrix)
    ):
        for cluster in cluster_means.index:
            cluster_mean_expression = cluster_means.loc[cluster].values
            correlation = np.corrcoef(cell_expression, cluster_mean_expression)[0, 1]
            correlation_matrix_q.loc[adata_q.obs.index[cell_idx], cluster] = correlation

    correlation_matrix_q = correlation_matrix_q.apply(pd.to_numeric, errors="coerce")
    correlation_matrix_q.fillna(0, inplace=True)

    raw_correlation_matrix_q = correlation_matrix_q.copy()

    # Assign cells a best cluster based on the highest correlation
    correlation_matrix_q["best_cluster"] = raw_correlation_matrix_q.idxmax(axis=1)
    correlation_matrix_q["best_score"] = raw_correlation_matrix_q.max(axis=1)
    correlation_matrix_q["median_score"] = raw_correlation_matrix_q.median(axis=1)
    correlation_matrix_q["delta_score"] = (
        correlation_matrix_q["best_score"] - correlation_matrix_q["median_score"]
    )

    if pd.api.types.is_categorical_dtype(correlation_matrix_q["best_cluster"]):
        correlation_matrix_q["best_cluster"] = correlation_matrix_q[
            "best_cluster"
        ].cat.add_categories("Zero_correlation")
    correlation_matrix_q.loc[
        correlation_matrix_q["best_score"] == 0, "best_cluster"
    ] = "Zero_correlation"

    adata_q.obs["best_cluster"] = correlation_matrix_q["best_cluster"]
    adata_q.obs["best_score"] = correlation_matrix_q["best_score"]
    adata_q.obs["median_score"] = correlation_matrix_q["median_score"]
    adata_q.obs["delta_score"] = correlation_matrix_q["delta_score"]

    return adata_q


def bootstrap_cluster_confidence(
    adata,
    cluster_means,
    n_boot=100,
    count_frac=0.9,
    use_pearson=True,
    batch_size=20000,
    seed=None,
):
    """
    Per-cell cluster-assignment stability by resampling UMIs.

    Parameters
    ----------
    adata : AnnData
        • If use_pearson = True  ➜  ``adata.raw.X`` must hold **raw counts**.
        • If use_pearson = False ➜  ``adata.X``      must hold the matrix on
          which cluster_means were computed (your log-normalised CPM-10).
    cluster_means : DataFrame (n_clusters x n_genes)
        Must be computed on the **same scale** as the matrix used for similarity.
    n_boot : int
        Bootstrap iterations.
    count_frac : float in (0,1] or None
        - 0 < count_frac < 1 : binomial thinning, keep-rate = count_frac
        - None               : classic bootstrap, resample the same total
                               UMIs with replacement (multinomial)
        - 1 or ≥1            : skip resampling
        *Only allowed when the working matrix is raw integer counts.*
    use_pearson : bool
        True  ➜  Pearson (z-score both cell and cluster profiles).
        False ➜  cosine similarity on un-scaled vectors.
    batch_size : int
        Cells processed per chunk.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    DataFrame with columns
        stable_cluster   - modal winner across bootstraps
        stability        - vote fraction for the modal winner (0-1)
        mean_best_score  - average similarity of the winner
    """
    rng = np.random.default_rng(seed)
    n_cells = adata.n_obs
    clusters = cluster_means.index.to_numpy()
    n_clusters = len(clusters)

    if use_pearson:
        if adata.raw is None:
            raise ValueError("use_pearson=True requires adata.raw.X with raw counts.")
        X_csr = adata.raw.X.tocsr()
    else:
        X_csr = adata.X.tocsr()

    if (not np.issubdtype(X_csr.dtype, np.integer)) and (
        count_frac != 1 and count_frac is not None
    ):
        raise ValueError(
            "count_frac <1 or None is only valid with integer count matrices."
        )

    C = cluster_means.to_numpy(dtype=np.float32)  # (k × g)

    if use_pearson:
        C = (C - C.mean(1, keepdims=True)) / (C.std(1, ddof=0, keepdims=True) + 1e-8)
    else:
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)

    C = np.ascontiguousarray(C)

    votes = np.zeros((n_cells, n_clusters), dtype=np.uint16)
    scores = np.zeros(n_cells, dtype=np.float32)

    for b in trange(n_boot, desc="bootstraps"):
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            X_b = X_csr[start:end].copy()

            # Resample
            if count_frac is None:
                X_arr = X_b.toarray().astype(np.int32)
                for i in range(X_arr.shape[0]):
                    tot = X_arr[i].sum()
                    if tot == 0:
                        continue
                    p = X_arr[i] / tot
                    X_arr[i] = rng.multinomial(tot, p)
            elif 0 < count_frac < 1:
                X_b.data = rng.binomial(X_b.data.astype(np.int32), count_frac).astype(
                    np.int32
                )
                X_b.eliminate_zeros()
                X_arr = X_b.toarray().astype(np.int32)
            else:  # no resampling
                X_arr = X_b.toarray().astype(np.float32)

            if use_pearson:
                X_arr = (X_arr - X_arr.mean(1, keepdims=True)) / (
                    X_arr.std(1, ddof=0, keepdims=True) + 1e-8
                )
                sim = (X_arr @ C.T) / X_arr.shape[1]  # Pearson
            else:
                X_arr = X_arr / (np.linalg.norm(X_arr, axis=1, keepdims=True) + 1e-8)
                sim = X_arr @ C.T  # cosine

            win_idx = np.argmax(sim, axis=1)
            win_sim = sim[np.arange(sim.shape[0]), win_idx]

            rows = np.arange(start, end)
            votes[rows, win_idx] += 1
            scores[rows] += win_sim

    # summarise results
    modal_idx = votes.argmax(1)
    stability = votes[np.arange(n_cells), modal_idx] / n_boot
    mean_best = scores / n_boot

    return pd.DataFrame(
        {
            "stable_cluster": clusters[modal_idx],
            "stability": stability.astype(np.float32),
            "mean_best_score": mean_best.astype(np.float32),
        },
        index=adata.obs_names,
    )


def plot_umap_barcoded_cells(
    adata,
    ax,
    size_non_barcoded=8,
    size_barcoded=10,
    legend_fontsize=6,
):
    """
    Plots a UMAP of barcoded vs non-barcoded cells onto a given matplotlib Axes.

    Args:
        adata: AnnData object with 'in_filtered_df' column in .obs
        ax: matplotlib Axes to plot on
        size_non_barcoded: point size for non-barcoded (NaN)
        size_barcoded: point size for barcoded (True)
    """
    adata.obs["black_const"] = "black"
    is_true = adata.obs["in_filtered_df"] == True
    is_nan = adata.obs["in_filtered_df"].isna()

    # Plot non-barcoded (grey)
    sc.pl.umap(
        adata[is_nan],
        color=None,
        ax=ax,
        title="",
        size=size_non_barcoded,
        frameon=False,
        show=False,
    )

    # Plot barcoded (black)
    sc.pl.umap(
        adata[is_true],
        color="black_const",
        palette=["black"],
        ax=ax,
        title="",
        size=size_barcoded,
        frameon=False,
        show=False,
    )

    # Title and legend
    legend_elements = [
        Patch(facecolor="lightgrey", edgecolor="lightgrey", label="Non-barcoded cells"),
        Patch(facecolor="black", edgecolor="black", label="Barcoded cells"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(1, -0.2),
        fontsize=legend_fontsize,
        frameon=False,
    )


def plot_cell_clusters(adata_q, ax, spot_size=10, fontsize=6, font_outline=1):
    """
    Plot UMAP with cell type clusters

    Args:
        adata_q : AnnData
            The query dataset with cell barcode genes.
        ax : matplotlib Axes
            The axes on which to plot the UMAP.
        spot_size : int
            Size of the spots in the UMAP.
        fontsize : int
            Font size for the legend.
        font_outline : int
            Font outline size for the legend.
    """
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    sc.set_figure_params(figsize=(9, 9))

    sc.pl.umap(
        adata_q,
        use_raw=True,
        color="custom_leiden",
        ax=ax,
        frameon=False,
        size=spot_size,
        title="",
        alpha=0.5,
        legend_loc="on data",
        legend_fontsize=fontsize,
        legend_fontoutline=font_outline,
        show=False,
    )
