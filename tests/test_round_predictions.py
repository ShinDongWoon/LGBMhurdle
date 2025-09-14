import pandas as pd
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

def postprocess(df):
    pred_cols = [c for c in df.columns if c.startswith("D")]
    t = torch.from_numpy(df[pred_cols].values)
    df[pred_cols] = torch.round(torch.clamp(t, min=0)).to(torch.int64).numpy()
    return df

def test_round_predictions():
    df = pd.DataFrame({"id": [1, 2], "D1": [-0.5, 3.7], "D2": [1.2, -4.6]})
    df = postprocess(df)
    pred_cols = [c for c in df.columns if c.startswith("D")]
    assert (df[pred_cols] >= 0).all().all()
    assert (df[pred_cols] % 1 == 0).all().all()
