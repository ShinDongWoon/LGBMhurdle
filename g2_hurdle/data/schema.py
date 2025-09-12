
from typing import Dict, List


def resolve_schema(df_columns: List[str], cfg: dict) -> Dict[str, object]:
    cols = set(df_columns)
    data_cfg = cfg.get("data", {})

    def pick_first(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    # Prefer direct keys if provided; fall back to candidate lists
    date_col = data_cfg.get("date_col")
    if date_col not in cols:
        date_col = pick_first(data_cfg.get("date_col_candidates", []))

    target_col = data_cfg.get("target_col")
    if target_col not in cols:
        target_col = pick_first(data_cfg.get("target_col_candidates", []))

    if "id_cols" in data_cfg:
        series_cols = [c for c in data_cfg.get("id_cols", []) if c in cols]
    else:
        id_candidates = data_cfg.get("id_col_candidates", [])
        series_cols = [c for c in id_candidates if c in cols]

    missing = []
    if date_col is None:
        missing.append("date column (date_col or date_col_candidates)")
    if target_col is None:
        missing.append("target column (target_col or target_col_candidates)")
    if missing:
        raise ValueError(
            "Schema resolution failed. Missing: {}. Available columns: {}".format(
                missing, sorted(list(cols))
            )
        )

    return {"date": date_col, "target": target_col, "series": series_cols}
