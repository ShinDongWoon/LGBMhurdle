
import pandas as pd
import numpy as np
import numba as nb

from ..fe import prepare_features
from ..fe.static import prepare_static_future_features
from ..utils.logging import get_logger

logger = get_logger("Recursion")


def _predict_batch(X, clf, reg, threshold: float):
    """Predict in batch ensuring feature alignment.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Feature matrix. When a DataFrame is provided, feature names are
        validated against the fitted models. For a NumPy array, it is assumed
        that the column order matches the training schema.
    clf, reg : LightGBM models
        Classifier and regressor used for the hurdle model.
    threshold : float
        Probability threshold to decide whether to use the regression output.
    """

    if isinstance(X, pd.DataFrame):
        reg_feats = list(getattr(reg, "feature_names_", []))
        clf_feats = list(getattr(clf, "feature_names_", []))
        x_cols = list(X.columns)

        if x_cols != reg_feats:
            logger.fatal(
                "Regressor feature mismatch: DataFrame has %s but regressor expects %s",
                x_cols,
                reg_feats,
            )
            raise AssertionError("Regressor feature mismatch")
        if x_cols != clf_feats:
            logger.fatal(
                "Classifier feature mismatch: DataFrame has %s but classifier expects %s",
                x_cols,
                clf_feats,
            )
            raise AssertionError("Classifier feature mismatch")

        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    clf_core = getattr(clf, "model", clf)
    reg_core = getattr(reg, "model", reg)

    if hasattr(clf_core, "booster_"):
        p_raw = clf_core.booster_.predict(X_arr)
    else:
        p_raw = clf_core.predict_proba(X_arr)
        if p_raw.ndim == 2:
            p_raw = p_raw[:, 1]

    if hasattr(reg_core, "booster_"):
        q_raw = reg_core.booster_.predict(X_arr)
    else:
        q_raw = reg_core.predict(X_arr)

    p = p_raw.astype(np.float32)
    q = q_raw.astype(np.float32)
    yhat = (p > threshold).astype(np.float32) * np.maximum(np.float32(0.0), q)
    return yhat, p, q


@nb.njit
def _compute_dynamic_features(
    hist_matrix,
    dsls_hist_matrix,
    lag_steps,
    lag_idx,
    roll_steps,
    roll_mean_idx,
    roll_std_idx,
    dsls_idx,
    rzero_idx,
    avg_idi_idx,
    out,
):
    """Populate dynamic features from history buffers."""
    n_series, tail_len = hist_matrix.shape
    for i in range(n_series):
        # lags
        for k in range(lag_steps.shape[0]):
            out[i, lag_idx[k]] = hist_matrix[i, tail_len - lag_steps[k]]

        # rolling statistics
        for k in range(roll_steps.shape[0]):
            w = roll_steps[k]
            start = tail_len - w
            s = 0.0
            ss = 0.0
            for j in range(start, tail_len):
                v = hist_matrix[i, j]
                s += v
                ss += v * v
            mean = s / w
            if roll_mean_idx[k] >= 0 and w not in (7, 14):
                out[i, roll_mean_idx[k]] = mean
            if roll_std_idx[k] >= 0:
                if w > 1:
                    var = (ss - s * s / w) / (w - 1)
                    out[i, roll_std_idx[k]] = np.sqrt(var)
                else:
                    out[i, roll_std_idx[k]] = 0.0

        # intermittency-related features
        if dsls_idx >= 0:
            out[i, dsls_idx] = dsls_hist_matrix[i, dsls_hist_matrix.shape[1] - 1]
        if rzero_idx >= 0:
            cnt = 0.0
            start = tail_len - 7
            for j in range(start, tail_len):
                if hist_matrix[i, j] == 0.0:
                    cnt += 1.0
            out[i, rzero_idx] = cnt
        if avg_idi_idx >= 0:
            s = 0.0
            for j in range(dsls_hist_matrix.shape[1]):
                s += dsls_hist_matrix[i, j]
            out[i, avg_idi_idx] = s / dsls_hist_matrix.shape[1]


@nb.njit
def _update_history_dsls(hist_matrix, dsls_hist_matrix, new_y):
    """Update history buffers with new predictions."""
    n_series, tail_len = hist_matrix.shape
    dsls_window = dsls_hist_matrix.shape[1]
    for i in range(n_series):
        # shift history
        for j in range(tail_len - 1):
            hist_matrix[i, j] = hist_matrix[i, j + 1]
        hist_matrix[i, tail_len - 1] = new_y[i]

        # update dsls
        dsls = dsls_hist_matrix[i, dsls_window - 1]
        if new_y[i] > 0.0:
            dsls = 0.0
        else:
            dsls += 1.0
        for j in range(dsls_window - 1):
            dsls_hist_matrix[i, j] = dsls_hist_matrix[i, j + 1]
        dsls_hist_matrix[i, dsls_window - 1] = dsls


def _init_dsls_history(y: np.ndarray, window: int = 28) -> np.ndarray:
    """Compute days-since-last-sale history."""
    dsls = np.zeros(len(y), dtype=np.float32)
    last = 0
    for i, v in enumerate(y):
        if v > 0:
            last = 0
        else:
            last += 1
        dsls[i] = last
    if len(dsls) < window:
        dsls_hist = np.pad(dsls[-window:], (window - len(dsls), 0), constant_values=0)
    else:
        dsls_hist = dsls[-window:]
    return dsls_hist.astype(np.float32)


def _build_dynamic_row(
    history: np.ndarray,
    dsls_hist: np.ndarray | None,
    lags,
    rolls,
    date_col: str,
    future_date,
):
    """Construct a DataFrame row of dynamic features for feature ordering."""
    row = {}
    for lag in lags:
        if lag in (2, 14):
            continue
        row[f"lag_{lag}"] = float(history[-lag])
    for w in rolls:
        window = history[-w:]
        if w not in (7, 14):
            row[f"roll_mean_{w}"] = float(window.mean())
        row[f"roll_std_{w}"] = float(window.std(ddof=1))
    if dsls_hist is not None:
        dsls = dsls_hist[-1]
        row["days_since_last_sale"] = float(dsls)
        row["rolling_zero_count_7d"] = float(np.sum(history[-7:] == 0))
        row["avg_interdemand_interval"] = float(dsls_hist.mean())
    out = pd.DataFrame([row])
    out[date_col] = future_date
    return out

def recursive_forecast_grouped(
    context_df: pd.DataFrame,
    schema: dict,
    cfg: dict,
    clf,
    reg,
    threshold: float,
    horizon: int = 7,
    feature_cols=None,
    categorical_cols=None,
    target_encoding_map=None,
):
    """Run recursive forecast per series group (identified by schema['series']).
    context_df: must contain at least the last 28 days per series.
    Returns DataFrame with columns: id, D1..Dh and optionally stacks of p,q for analysis.
    """
    date_col = schema["date"]
    target_col = schema["target"]
    series_cols = schema["series"]
    H = int(cfg.get("cv", {}).get("horizon", horizon))

    # Ensure series id
    from ..utils.keys import build_series_id, ensure_wide_columns
    id_series = build_series_id(context_df, series_cols)
    context_df = context_df.copy()
    context_df["_series_id"] = id_series

    drop_cols = [
        date_col,
        target_col,
        *[
            c
            for c in series_cols
            if c not in ("store_id", "menu_id", "store_menu_id")
        ],
    ]
    lags = cfg.get("features", {}).get("lags", [1, 7, 28, 365])
    lags = [l for l in lags if l not in (2, 14)]
    rolls = cfg.get("features", {}).get("rollings", [7, 14, 28])
    tail_len = max(max(lags), max(rolls)) + 1
    use_intermittency = cfg.get("features", {}).get("intermittency", {}).get(
        "enable", True
    )

    # Per-series caches
    series_data = {}
    histories = []
    dsls_histories = []
    static_frames = {}
    static_cache = {}

    for sid, g in context_df.groupby("_series_id"):
        g = g.sort_values(date_col).copy()
        last_date = g[date_col].max()

        base_static = static_cache.get(last_date)
        if base_static is None:
            base_static = prepare_static_future_features(g, schema, cfg, H)
            static_cache[last_date] = base_static
        static_feats = base_static.copy()
        for col in ("store_id", "menu_id", "store_menu_id"):
            if col in g.columns:
                val = str(g[col].iloc[0])
                static_feats[col] = pd.Series(
                    [val] * len(static_feats), dtype="category"
                )
                if target_encoding_map and col in target_encoding_map:
                    stats = target_encoding_map[col].get(
                        val, target_encoding_map[col].get("__default__", {})
                    )
                    static_feats[f"{col}_te_mean"] = float(stats.get("mean", 0.0))
                    static_feats[f"{col}_te_std"] = float(stats.get("std", 0.0))
        static_frames[sid] = static_feats

        y = g[target_col].astype(np.float32).values
        if len(y) < tail_len:
            hist = np.pad(y, (tail_len - len(y), 0), constant_values=0)
        else:
            hist = y[-tail_len:]
        histories.append(hist.astype(np.float32))

        if use_intermittency:
            dsls_hist = _init_dsls_history(y)
        else:
            dsls_hist = np.zeros(28, dtype=np.float32)
        dsls_histories.append(dsls_hist)

        series_data[sid] = {
            "last_date": last_date,
            "preds": [],
            "probs": [],
            "qtys": [],
        }

    series_ids = list(series_data.keys())

    # Determine feature order once
    if feature_cols is None:
        fe_rows = []
        for sid, hist, dsls_hist in zip(series_ids, histories, dsls_histories):
            info = series_data[sid]
            future_date = info["last_date"] + pd.Timedelta(days=1)
            dyn_row = _build_dynamic_row(
                hist, dsls_hist if use_intermittency else None, lags, rolls, date_col, future_date
            )
            stat_row = static_frames[sid].loc[[future_date]].reset_index(drop=True)
            fe_rows.append(pd.concat([dyn_row.reset_index(drop=True), stat_row], axis=1))
        fe_df = pd.concat(fe_rows, axis=0, ignore_index=True)
        _, feature_cols, categorical_cols = prepare_features(
            fe_df,
            drop_cols,
            feature_cols=None,
            categorical_cols=None,
        )
    feat_idx = {c: i for i, c in enumerate(feature_cols)}

    # Convert per-series static frames to arrays
    static_array_frames = {}
    for sid, static_df in static_frames.items():
        tmp = static_df.reset_index(drop=True)
        tmp = tmp[[c for c in feature_cols if c in tmp.columns]].copy()
        for c in categorical_cols or []:
            if c in tmp.columns:
                tmp[c] = tmp[c].astype("category").cat.codes.astype(np.float32)
        tmp = tmp.fillna(0)
        mat = np.zeros((len(tmp), len(feature_cols)), dtype=np.float32)
        for c in tmp.columns:
            if c in feat_idx:
                mat[:, feat_idx[c]] = tmp[c].astype(np.float32).values
        static_array_frames[sid] = mat

    # Build static matrix aligned with series order
    static_matrix = np.stack([static_array_frames[sid] for sid in series_ids], axis=0)

    # Prepare history arrays
    hist_matrix = np.vstack(histories)
    dsls_hist_matrix = np.vstack(dsls_histories)

    # Precompute index arrays for numba routines
    lag_steps = np.array(lags, dtype=np.int64)
    lag_idx = np.array([feat_idx.get(f"lag_{l}", -1) for l in lags], dtype=np.int64)
    roll_steps = np.array(rolls, dtype=np.int64)
    roll_mean_idx = np.array(
        [feat_idx.get(f"roll_mean_{w}", -1) for w in rolls], dtype=np.int64
    )
    roll_std_idx = np.array(
        [feat_idx.get(f"roll_std_{w}", -1) for w in rolls], dtype=np.int64
    )
    dsls_idx = feat_idx.get("days_since_last_sale", -1)
    rzero_idx = feat_idx.get("rolling_zero_count_7d", -1)
    avg_idi_idx = feat_idx.get("avg_interdemand_interval", -1)

    n_series = len(series_ids)
    n_feat = len(feature_cols)
    dyn_batch = np.zeros((n_series, n_feat), dtype=np.float32)

    for h in range(1, H + 1):
        dyn_batch.fill(0)
        _compute_dynamic_features(
            hist_matrix,
            dsls_hist_matrix,
            lag_steps,
            lag_idx,
            roll_steps,
            roll_mean_idx,
            roll_std_idx,
            dsls_idx,
            rzero_idx,
            avg_idi_idx,
            dyn_batch,
        )
        X_batch = dyn_batch + static_matrix[:, h - 1, :]
        yhat_batch, p_batch, q_batch = _predict_batch(X_batch, clf, reg, threshold)

        for i, sid in enumerate(series_ids):
            info = series_data[sid]
            info["preds"].append(float(yhat_batch[i]))
            info["probs"].append(float(p_batch[i]))
            info["qtys"].append(float(q_batch[i]))

        if h < H:
            _update_history_dsls(hist_matrix, dsls_hist_matrix, yhat_batch)

    out_rows = []
    for sid in series_ids:
        info = series_data[sid]
        row = {"id": sid}
        for i, v in enumerate(info["preds"], start=1):
            row[f"D{i}"] = float(v)
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    need_cols = ensure_wide_columns(H)
    for c in need_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[["id", *need_cols]]
