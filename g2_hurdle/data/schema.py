
from typing import Dict, List

def resolve_schema(df_columns: List[str], cfg: dict) -> Dict[str, object]:
    cols = set(df_columns)
    data_cfg = cfg.get("data", {})

    def pick_first(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    date_col = pick_first(data_cfg.get("date_col_candidates", []))
    target_col = pick_first(data_cfg.get("target_col_candidates", []))
    id_candidates = data_cfg.get("id_col_candidates", [])
    series_cols = [c for c in id_candidates if c in cols]

    missing = []
    if date_col is None:
        missing.append("date column from candidates")
    if target_col is None:
        missing.append("target column from candidates")
    if missing:
        raise ValueError(f"Schema resolution failed. Missing: {missing}. Available columns: {sorted(list(cols))}")

    return {"date": date_col, "target": target_col, "series": series_cols}
