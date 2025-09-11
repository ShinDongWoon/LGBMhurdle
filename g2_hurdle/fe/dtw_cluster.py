import numpy as np
import pandas as pd

from ..utils.logging import get_logger


logger = get_logger("DTWCluster")


def compute_dtw_clusters(
    df: pd.DataFrame, schema: dict, n_clusters: int = 20, use_gpu: bool = False
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
    n_clusters : int, default 20
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
