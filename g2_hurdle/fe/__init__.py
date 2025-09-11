import pandas as pd
from typing import Optional
from .calendar import create_calendar_features
from .fourier import create_fourier_features
from .holiday import create_holiday_features
from .lags_rolling import create_lags_and_rolling_features
from .intermittency import create_intermittency_features
from .preprocess import prepare_features
from .dtw_cluster import compute_dtw_clusters, create_demand_cluster_features
from .relational import create_relational_features


def run_feature_engineering(
    df: pd.DataFrame,
    cfg: dict,
    schema: dict,
    mapping: Optional[dict] = None,
):
    date_col = schema["date"]
    target_col = schema["target"]
    series_cols = [c for c in schema.get("series", []) if c != "demand_cluster"]
    out = df.copy()
    extras = {}

    dtw_cfg = cfg.get("features", {}).get("dtw", {})
    if dtw_cfg.get("enable"):
        if "demand_cluster" not in out.columns:
            n_clusters = int(dtw_cfg.get("n_clusters", 20))
            use_gpu = bool(dtw_cfg.get("use_gpu", True))
            clusters = compute_dtw_clusters(out, schema, n_clusters, use_gpu)
        else:
            clusters = (
                out[["store_menu_id", "demand_cluster"]]
                .drop_duplicates()
                .set_index("store_menu_id")["demand_cluster"]
                .to_dict()
            )
        out["demand_cluster"] = out["store_menu_id"].map(clusters).astype("category")
        extras["dtw_clusters"] = clusters

    out = create_calendar_features(out, date_col)
    if cfg.get("features", {}).get("use_holidays"):
        out = create_holiday_features(out, date_col)
    else:
        out["is_holiday"] = 0
        out["holiday_name"] = pd.Series(["None"] * len(out), dtype="category")
    out = create_fourier_features(out, date_col, cfg)
    # Group aggregate features are computed inside each CV fold using only
    # the training portion of the data to avoid leakage.
    out = create_lags_and_rolling_features(out, target_col, series_cols, cfg)
    if cfg.get("features", {}).get("intermittency", {}).get("enable", True):
        out = create_intermittency_features(out, target_col, series_cols)
    if "demand_cluster" in out.columns:
        out, te_map = create_demand_cluster_features(out, schema, cfg, mapping)
        extras["target_encoding"] = te_map
    out = create_relational_features(out, schema)
    # store_id and menu_id are only needed for feature construction
    out = out.drop(columns=[c for c in ["store_id", "menu_id"] if c in out.columns])
    return out, extras
