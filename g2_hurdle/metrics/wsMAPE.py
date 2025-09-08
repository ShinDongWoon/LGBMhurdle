
import numpy as np

def calculate_wsmape(y_true, y_pred, weights=None, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return 100.0 * np.sum(weights * np.abs(y_true - y_pred) / denom) / np.sum(weights)
