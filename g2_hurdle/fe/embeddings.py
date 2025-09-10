import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional


def create_embedding_features(
    df: pd.DataFrame,
    columns: List[str],
    cfg: dict,
    mapping: Optional[Dict[str, Dict[str, List[float]]]] = None,
):
    """Create dense embedding features for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : List[str]
        Categorical columns to embed.
    cfg : dict
        Configuration dictionary. Reads ``features.embeddings.dim`` for
        dimension and ``runtime.seed`` for reproducibility.
    mapping : dict, optional
        Existing mapping of {column: {value: embedding_list}} used to transform
        identifiers consistently. When ``None`` a new mapping is fitted.

    Returns
    -------
    out : pd.DataFrame
        Dataframe with additional embedding columns appended.
    mapping : dict
        Mapping of identifier to embedding vectors for each column.
    """

    emb_cfg = cfg.get("features", {}).get("embeddings", {})
    dim = int(emb_cfg.get("dim", 8))
    seed = int(cfg.get("runtime", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    mapping = {} if mapping is None else {k: dict(v) for k, v in mapping.items()}

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        series = out[col].astype(str)
        if col in mapping:
            col_map = mapping[col]
            d = len(next(iter(col_map.values()))) if col_map else dim
            arr = np.vstack([col_map.get(val, np.zeros(d)) for val in series])
        else:
            le = LabelEncoder()
            codes = le.fit_transform(series)
            emb_matrix = rng.standard_normal((len(le.classes_), dim))
            col_map = {cls: emb_matrix[i].tolist() for i, cls in enumerate(le.classes_)}
            mapping[col] = col_map
            arr = emb_matrix[codes]
        for j in range(arr.shape[1]):
            out[f"{col}_emb_{j}"] = arr[:, j]
    return out, mapping
