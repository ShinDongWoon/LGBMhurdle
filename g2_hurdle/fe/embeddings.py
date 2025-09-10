import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def create_target_encoding_features(
    df: pd.DataFrame,
    columns: List[str],
    target_col: str,
    date_col: str,
    cfg: dict,
    mapping: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
):
    """Create target-encoding features for categorical columns.

    The encoding is computed in a time-aware manner: for each column a
    per-category expanding mean and standard deviation of ``target_col`` are
    calculated using only past observations.  Smoothing based on category
    frequency is applied to shrink statistics toward the global average.  The
    resulting features are appended to ``df`` and the fitted mapping of
    category to statistics is returned so that the same transformation can be
    reused at inference time.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by date within each series.
    columns : List[str]
        Categorical columns to encode.
    target_col : str
        Target column whose statistics are encoded.
    date_col : str
        Date column used for deterministic ordering.
    cfg : dict
        Configuration dictionary. Reads ``features.target_encoding.smoothing``
        for the smoothing strength.
    mapping : dict, optional
        Existing mapping used to transform identifiers consistently. When
        ``None`` a new mapping is fitted.

    Returns
    -------
    out : pd.DataFrame
        Dataframe with additional target encoding columns appended.
    mapping : dict
        Mapping of identifier to {"mean", "std"} statistics for each column.
    """

    te_cfg = cfg.get("features", {}).get("target_encoding", {})
    smoothing = float(te_cfg.get("smoothing", 10.0))

    out = df.copy()
    mapping = {} if mapping is None else {k: dict(v) for k, v in mapping.items()}

    # Work on sorted copy to ensure time-aware encoding but keep original order
    sorted_out = out.sort_values(date_col).reset_index()

    # Compute global statistics incrementally when fitting.  When an existing
    # mapping is provided we reuse the stored global statistics instead of
    # recomputing them from ``df``.
    if mapping and "__global__" in mapping:
        global_mean = float(mapping["__global__"].get("mean", 0.0))
        global_std = float(mapping["__global__"].get("std", 0.0))
        global_means = [global_mean] * len(sorted_out)
        global_stds = [global_std] * len(sorted_out)
    else:
        global_means, global_stds = [], []
        g_count, g_mean, g_M2 = 0, 0.0, 0.0
        for y in sorted_out[target_col]:
            g_std = np.sqrt(g_M2 / g_count) if g_count > 1 else 0.0
            global_means.append(g_mean)
            global_stds.append(g_std)

            g_count += 1
            delta = y - g_mean
            g_mean += delta / g_count
            g_M2 += delta * (y - g_mean)

        global_mean = g_mean
        global_std = np.sqrt(g_M2 / g_count) if g_count > 1 else 0.0
        mapping["__global__"] = {"mean": float(global_mean), "std": float(global_std)}

    for col in columns:
        if col not in out.columns:
            continue

        if col in mapping:
            col_map = mapping[col]
            default = col_map.get("__default__", {"mean": global_mean, "std": global_std})
            means, stds = [], []
            for val in sorted_out[col].astype(str):
                stats = col_map.get(val, default)
                means.append(stats.get("mean", default["mean"]))
                stds.append(stats.get("std", default["std"]))
        else:
            means, stds = [], []
            stats = {}  # value -> (count, mean, M2)
            for i, (val, y) in enumerate(zip(sorted_out[col].astype(str), sorted_out[target_col])):
                gmean = global_means[i]
                gstd = global_stds[i]
                count, mean, M2 = stats.get(val, (0, 0.0, 0.0))
                if count > 0:
                    cat_mean = mean
                    cat_std = np.sqrt(M2 / count) if count > 1 else 0.0
                    w = count / (count + smoothing)
                    enc_mean = w * cat_mean + (1 - w) * gmean
                    enc_std = w * cat_std + (1 - w) * gstd
                else:
                    enc_mean = gmean
                    enc_std = gstd
                means.append(enc_mean)
                stds.append(enc_std)

                # Update running stats
                count += 1
                delta = y - mean
                mean += delta / count
                M2 += delta * (y - mean)
                stats[val] = (count, mean, M2)

            # Build mapping for inference
            col_map = {}
            for val, (count, mean, M2) in stats.items():
                cat_std = np.sqrt(M2 / count) if count > 1 else 0.0
                w = count / (count + smoothing)
                enc_mean = w * mean + (1 - w) * global_mean
                enc_std = w * cat_std + (1 - w) * global_std
                col_map[val] = {"mean": float(enc_mean), "std": float(enc_std)}
            col_map["__default__"] = {"mean": global_mean, "std": global_std}
            mapping[col] = col_map

        sorted_out[f"{col}_te_mean"] = means
        sorted_out[f"{col}_te_std"] = stds

    # Restore original ordering
    out = sorted_out.set_index("index").sort_index()

    return out, mapping
