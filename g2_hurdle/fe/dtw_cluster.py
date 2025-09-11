import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from ..utils.logging import get_logger
from .embeddings import create_target_encoding_features


logger = get_logger("DTWCluster")


def compute_dtw_clusters(
    df: pd.DataFrame, schema: dict, n_clusters: int = 2, use_gpu: bool = False
):
    """Cluster demand series using Dynamic Time Warping distances.

    Computation relies on optimized CPU routines; GPU acceleration is not
    available.

    Parameters
    ----------
    df : DataFrame
        Input data containing at least date, target and ``store_menu_id`` columns.
    schema : dict
        Schema mapping with keys ``date`` and ``target``.
    n_clusters : int, default 2
        Number of clusters to form.
    use_gpu : bool, default False
        Retained for compatibility. If ``True``, a warning is logged and CPU
        computation is used.

    Returns
    -------
    dict
        Mapping from ``store_menu_id`` to assigned cluster label.
    """

    date_col = schema["date"]
    target_col = schema["target"]
    if "store_menu_id" not in df.columns:
        raise KeyError("compute_dtw_clusters requires 'store_menu_id' column")

    # Pivot demand by store_menu_id -> time series rows
    pivot = (
        df.pivot_table(
            index="store_menu_id",
            columns=date_col,
            values=target_col,
            fill_value=0,
            observed=False,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    series_ids = pivot.index.tolist()
    data = pivot.values.astype("float64")

    n_clusters = int(min(max(1, n_clusters), len(series_ids)))

    if use_gpu:
        logger.debug(
            "GPU acceleration requested but not available; using CPU routines."
        )

    from dtaidistance import dtw

    distance_matrix = dtw.distance_matrix_fast(
        data, compact=False, parallel=True
    )

    try:
        from sklearn_extra.cluster import KMedoids

        km = KMedoids(
            n_clusters=n_clusters, metric="precomputed", random_state=0
        )
        km.fit(distance_matrix)
        labels = km.labels_
    except Exception:
        rng = np.random.default_rng(0)
        n = distance_matrix.shape[0]
        medoids = rng.choice(n, size=n_clusters, replace=False)
        for _ in range(300):
            labels = np.argmin(distance_matrix[:, medoids], axis=1)
            new_medoids = []
            for k in range(n_clusters):
                cluster_idx = np.where(labels == k)[0]
                if len(cluster_idx) == 0:
                    new_medoids.append(medoids[k])
                    continue
                intra = distance_matrix[np.ix_(cluster_idx, cluster_idx)]
                costs = intra.sum(axis=1)
                new_medoids.append(cluster_idx[np.argmin(costs)])
            new_medoids = np.array(new_medoids)
            if np.all(new_medoids == medoids):
                break
            medoids = new_medoids
        labels = np.argmin(distance_matrix[:, medoids], axis=1)

    labels = [int(x) for x in labels]
    clusters = dict(zip(series_ids, labels))
    return clusters


def create_demand_cluster_features(
    df: pd.DataFrame,
    schema: dict,
    cfg: dict,
    mapping: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, float]]]]:
    """Add features derived from demand clusters.

    Currently this computes ``demand_vs_cluster_mean`` based on the
    ``roll_mean_7`` column and applies target encoding to
    ``demand_cluster``.  The rolling mean ensures that only past demand is
    used when comparing against the cluster level average.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe containing ``demand_cluster`` and ``roll_mean_7``.
    schema : dict
        Schema dictionary with at least ``date`` and ``target`` keys.
    cfg : dict
        Configuration dictionary passed to the target encoding routine.
    mapping : dict, optional
        Pre-fitted target encoding mapping. When provided, the same mapping is
        used instead of recomputing it.

    Returns
    -------
    out : pd.DataFrame
        DataFrame with additional cluster-based features.
    mapping : dict
        Mapping used for target encoding so that it can be persisted for
        inference.
    """

    out = df.copy()
    date_col = schema.get("date")
    target_col = schema.get("target")
    roll_col = "roll_mean_7"
    cluster_col = "demand_cluster"

    if cluster_col in out.columns:
        if roll_col not in out.columns and target_col in out.columns:
            if "store_menu_id" in out.columns:
                out[roll_col] = (
                    out.groupby("store_menu_id")[target_col]
                    .shift(1)
                    .rolling(window=7, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
            else:
                out[roll_col] = (
                    out[target_col].shift(1).rolling(window=7, min_periods=1).mean()
                )
            out[roll_col] = out[roll_col].fillna(0)
        if {roll_col, date_col}.issubset(out.columns):
            cluster_mean = (
                out.groupby([cluster_col, date_col], observed=False)[roll_col]
                .transform("mean")
            )
            out["demand_vs_cluster_mean"] = (
                out[roll_col] - cluster_mean
            ).astype("float32")
        else:
            out["demand_vs_cluster_mean"] = 0.0
    else:
        out["demand_vs_cluster_mean"] = 0.0

    out, mapping = create_target_encoding_features(
        out,
        [cluster_col],
        target_col,
        date_col,
        cfg,
        mapping=mapping,
    )

    return out, mapping or {}
