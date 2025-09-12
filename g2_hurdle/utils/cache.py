
import os, hashlib, pandas as pd

def _hash_key(obj: str) -> str:
    return hashlib.md5(obj.encode("utf-8")).hexdigest()

def cache_path(base_dir: str, key: str) -> str:
    h = _hash_key(key)
    return os.path.join(base_dir, f"fe_cache_{h}.parquet")

def load_cache(path: str):
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

def save_cache(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
