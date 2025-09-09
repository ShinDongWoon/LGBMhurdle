
import pandas as pd
import numpy as np
from ..fe import prepare_features
from ..fe.lags_rolling import (
    create_lags_and_rolling_features,
    update_lags_and_rollings,
)
from ..fe.intermittency import (
    create_intermittency_features,
    update_intermittency_features,
)
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

    p_raw = clf.predict_proba(X)
    p = (p_raw if p_raw.ndim == 1 else p_raw[:, 1]).astype(np.float32, copy=False)
    q = reg.predict(X).astype(np.float32, copy=False)
    yhat = (p > threshold).astype(np.float32) * np.maximum(np.float32(0.0), q)
    return yhat, p, q

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

    drop_cols = [date_col, target_col, *series_cols]
    series_data = {}
    static_cache = {}
    for sid, g in context_df.groupby("_series_id"):
        g = g.sort_values(date_col).copy()
        last_date = g[date_col].max()
        static_feats = static_cache.get(last_date)
        if static_feats is None:
            static_feats = prepare_static_future_features(g, schema, cfg, H)
            static_cache[last_date] = static_feats

        fe_g = create_lags_and_rolling_features(g, target_col, series_cols, cfg)
        if cfg.get("features", {}).get("intermittency", {}).get("enable", True):
            fe_g = create_intermittency_features(fe_g, target_col, series_cols)

        lags = cfg.get("features", {}).get("lags", [1, 2, 7, 14, 28, 365])
        rolls = cfg.get("features", {}).get("rollings", [7, 14, 28])
        tail_len = max(max(lags), max(rolls)) + 1
        ctx = fe_g.tail(tail_len).reset_index(drop=True)

        last_y = g[target_col].iloc[-1]
        ctx = update_lags_and_rollings(ctx, last_y, cfg)
        if cfg.get("features", {}).get("intermittency", {}).get("enable", True):
            ctx = update_intermittency_features(ctx, last_y)

        series_data[sid] = {
            "ctx": ctx,
            "static": static_feats,
            "last_date": last_date,
            "preds": [],
            "probs": [],
            "qtys": [],
        }

    series_ids = list(series_data.keys())

    # Determine feature order once
    if feature_cols is None:
        fe_rows = []
        for sid in series_ids:
            info = series_data[sid]
            future_date = info["last_date"] + pd.Timedelta(days=1)
            dyn_row = info["ctx"].iloc[[-1]].copy()
            dyn_row[date_col] = future_date
            stat_row = info["static"].loc[[future_date]].reset_index(drop=True)
            fe_rows.append(pd.concat([dyn_row.reset_index(drop=True), stat_row], axis=1))
        fe_df = pd.concat(fe_rows, axis=0, ignore_index=True)
        _, feature_cols, categorical_cols = prepare_features(
            fe_df,
            drop_cols,
            feature_cols=None,
            categorical_cols=None,
        )
    feat_idx = {c: i for i, c in enumerate(feature_cols)}

    # Convert cached static features to arrays once
    static_array_cache = {}
    for last_date, static_df in static_cache.items():
        tmp = static_df.reset_index(drop=True).copy()
        for c in categorical_cols or []:
            if c in tmp.columns:
                tmp[c] = tmp[c].astype("category").cat.codes.astype(np.float32)
        tmp = tmp.fillna(0)
        mat = np.zeros((len(tmp), len(feature_cols)), dtype=np.float32)
        for c in tmp.columns:
            if c in feat_idx:
                mat[:, feat_idx[c]] = tmp[c].astype(np.float32).values
        static_array_cache[last_date] = mat

    # Prepare per-series arrays
    for sid, info in series_data.items():
        ctx_row = info["ctx"].iloc[[-1]].drop(columns=drop_cols, errors="ignore")
        dyn = np.zeros(len(feature_cols), dtype=np.float32)
        row = ctx_row.iloc[0]
        for c, v in row.items():
            if c in feat_idx:
                if pd.isna(v) or np.isinf(v):
                    v = 0.0
                dyn[feat_idx[c]] = float(v)
        info["dyn_array"] = dyn
        info["static_arrays"] = static_array_cache[info["last_date"]]
        del info["static"]

    for h in range(1, H + 1):
        batch = []
        sid_order = []
        for sid in series_ids:
            info = series_data[sid]
            X = info["dyn_array"] + info["static_arrays"][h - 1]
            batch.append(X)
            sid_order.append(sid)
        X_batch = np.vstack(batch)
        yhat_batch, p_batch, q_batch = _predict_batch(X_batch, clf, reg, threshold)

        for sid, yhat, p, q in zip(sid_order, yhat_batch, p_batch, q_batch):
            info = series_data[sid]
            info["preds"].append(np.float32(yhat))
            info["probs"].append(np.float32(p))
            info["qtys"].append(np.float32(q))
            info["ctx"] = update_lags_and_rollings(info["ctx"], float(yhat), cfg)
            if cfg.get("features", {}).get("intermittency", {}).get("enable", True):
                info["ctx"] = update_intermittency_features(info["ctx"], float(yhat))
            if h < H:
                ctx_row = info["ctx"].iloc[[-1]].drop(columns=drop_cols, errors="ignore")
                info["dyn_array"].fill(0)
                row = ctx_row.iloc[0]
                for c, v in row.items():
                    if c in feat_idx:
                        if pd.isna(v) or np.isinf(v):
                            v = 0.0
                        info["dyn_array"][feat_idx[c]] = float(v)

    out_rows = []
    for sid, info in series_data.items():
        row = {"id": sid}
        for i, v in enumerate(info["preds"], start=1):
            row[f"D{i}"] = float(v)
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    # Ensure D1..Dh exist
    need_cols = ensure_wide_columns(H)
    for c in need_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[["id", *need_cols]]
