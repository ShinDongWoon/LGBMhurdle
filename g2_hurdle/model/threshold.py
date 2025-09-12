
import numpy as np
from ..metrics.wsMAPE import calculate_wsmape

def find_optimal_threshold(y_true, y_proba, y_qty, cfg):
    g0 = cfg.get("threshold", {}).get("grid_start", 0.01)
    g1 = cfg.get("threshold", {}).get("grid_end", 0.99)
    gs = cfg.get("threshold", {}).get("grid_step", 0.01)
    grid = np.arange(g0, g1 + 1e-12, gs)
    scores = []
    for t in grid:
        y_hat = (y_proba > t) * np.maximum(0.0, y_qty)
        s = calculate_wsmape(y_true, y_hat)
        scores.append(s)
    best_idx = int(np.argmin(scores))
    return float(grid[best_idx]), float(scores[best_idx])
