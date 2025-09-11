import pandas as pd


def compute_dtw_clusters(df: pd.DataFrame, schema: dict, n_clusters: int = 20, use_gpu: bool = True):
    """Cluster demand series using Dynamic Time Warping distances.

    Parameters
    ----------
    df : DataFrame
        Input data containing at least date, target and ``store_menu_id`` columns.
    schema : dict
        Schema mapping with keys ``date`` and ``target``.
    n_clusters : int, default 20
        Number of clusters to form.
    use_gpu : bool, default True
        Whether to attempt GPU-accelerated computation. Falls back to CPU
        implementations if GPU libraries are unavailable.

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
        )
        .sort_index()
        .sort_index(axis=1)
    )
    series_ids = pivot.index.tolist()
    data = pivot.values.astype("float32")

    n_clusters = int(min(max(1, n_clusters), len(series_ids)))

    distance_matrix = None
    if use_gpu:
        try:  # pragma: no cover - GPU path isn't exercised in tests
            import cupy as cp
            try:
                from cudtw import distance_matrix as cudtw_distance_matrix

                distance_matrix = cudtw_distance_matrix(cp.asarray(data))
                distance_matrix = cp.asnumpy(distance_matrix)
            except Exception:  # fall back to CPU DTW
                distance_matrix = None
        except Exception:
            distance_matrix = None

    if distance_matrix is None:
        from tslearn.metrics import cdist_dtw

        distance_matrix = cdist_dtw(data)

    labels = None
    if use_gpu:
        try:  # pragma: no cover
            from cuml.cluster import KMeans as cuKMeans

            km = cuKMeans(n_clusters=n_clusters, random_state=0)
            labels = km.fit_predict(distance_matrix)
            labels = labels.get() if hasattr(labels, "get") else labels
        except Exception:
            labels = None

    if labels is None:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, random_state=0)
        labels = km.fit_predict(distance_matrix)

    labels = [int(x) for x in labels]
    clusters = dict(zip(series_ids, labels))
    return clusters
