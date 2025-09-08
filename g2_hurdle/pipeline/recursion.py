
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


def _predict_one_step(X, clf, reg, threshold):
    """Predict a single step ensuring feature alignment."""
    reg_feats = len(getattr(reg, "feature_names_", []))
    clf_feats = len(getattr(clf, "feature_names_", []))
    if clf_feats != reg_feats:
        logger.fatal(
            "Feature count mismatch: classifier expects %d vs regressor %d",
            clf_feats,
            reg_feats,
        )
        raise AssertionError("Classifier/regressor feature count mismatch")
    if X.shape[1] != reg_feats:
        logger.fatal(
            "Feature shape mismatch: X has %d columns while models expect %d",
            X.shape[1],
            reg_feats,
        )
        raise AssertionError("Regressor feature mismatch")
    p = clf.predict_proba(X)
    q = reg.predict(X)
    yhat = (p > threshold).astype(float) * np.maximum(0.0, q)
    return float(yhat[0]), float(p[0]), float(q[0])

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

    out_rows = []
    for sid, g in context_df.groupby("_series_id"):
        g = g.sort_values(date_col).copy()
        preds, probs, qtys = [], [], []
        last_date = g[date_col].max()
        static_feats = prepare_static_future_features(g, schema, cfg, H)

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

        for h in range(1, H + 1):
            future_date = last_date + pd.Timedelta(days=h)

            dyn_row = ctx.iloc[[-1]].copy()
            dyn_row[date_col] = future_date
            stat_row = static_feats.loc[[future_date]].reset_index(drop=True)
            fe_row = pd.concat([dyn_row.reset_index(drop=True), stat_row], axis=1)
            drop_cols = [date_col, target_col, *series_cols]
            X_row, _, _ = prepare_features(
                fe_row,
                drop_cols,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
            )
            yhat, p, q = _predict_one_step(X_row, clf, reg, threshold)
            preds.append(yhat)
            probs.append(p)
            qtys.append(q)

            ctx = update_lags_and_rollings(ctx, yhat, cfg)
            if cfg.get("features", {}).get("intermittency", {}).get("enable", True):
                ctx = update_intermittency_features(ctx, yhat)

        row = {"id": sid}
        for i, v in enumerate(preds, start=1):
            row[f"D{i}"] = v
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    # Ensure D1..Dh exist
    need_cols = ensure_wide_columns(H)
    for c in need_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[["id", *need_cols]]
