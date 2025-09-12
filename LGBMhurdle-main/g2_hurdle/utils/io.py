
import os, json, joblib, pandas as pd
from .logging import get_logger
from ..data.schema import resolve_schema

logger = get_logger("IO")

def load_data(path: str, cfg: dict):
    df = pd.read_csv(path)
    schema = resolve_schema(df.columns.tolist(), cfg)
    date_col = schema["date"]
    target_col = schema["target"]
    series_cols = schema["series"]
    df[date_col] = pd.to_datetime(df[date_col])
    for c in series_cols:
        df[c] = df[c].astype("category")
    df = df.sort_values([*series_cols, date_col]).drop_duplicates([*series_cols, date_col])
    return df, schema

def save_artifacts(artifacts: dict, path: str):
    os.makedirs(path, exist_ok=True)
    for k, v in artifacts.items():
        if k.endswith(".json"):
            with open(os.path.join(path, k), "w", encoding="utf-8") as f:
                json.dump(v, f, ensure_ascii=False, indent=2)
        elif k.endswith(".pkl"):
            joblib.dump(v, os.path.join(path, k))
        else:
            # auto pkl
            joblib.dump(v, os.path.join(path, f"{k}.pkl"))
    logger.info(f"Saved artifacts to {path}")

def load_artifacts(path: str):
    # Load known filenames if exist
    import glob, json
    out = {}
    for p in glob.glob(os.path.join(path, "*.pkl")):
        name = os.path.basename(p)
        out[name] = joblib.load(p)
    for p in glob.glob(os.path.join(path, "*.json")):
        name = os.path.basename(p)
        with open(p, "r", encoding="utf-8") as f:
            out[name] = json.load(f)
    return out
