import numpy as np
import pandas as pd
from typing import Tuple


def clip_negative_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Clip negative values in specified columns to zero.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the columns to clip.
    columns : list[str]
        Column names where negative values should be clipped to zero.

    Returns
    -------
    pd.DataFrame
        The dataframe with negative values in ``columns`` replaced by zero.
    """
    df[columns] = df[columns].clip(lower=0)
    return df


def ensure_min_positive_ratio(
    X: pd.DataFrame, y: np.ndarray, min_ratio: float, seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Ensure that the fraction of positive targets is at least ``min_ratio``.

    This function operates on a feature matrix ``X`` and corresponding target
    vector ``y``.  If the proportion of positive targets (``y > 0``) is below
    the requested ratio, positive samples are resampled with replacement until
    the ratio is satisfied.  Both ``X`` and ``y`` are returned with the
    additional rows appended.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values aligned with ``X``.
    min_ratio : float
        Desired minimum ratio of positive targets (0-1 range).
    seed : int, optional
        Random seed for sampling, by default ``42``.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Augmented ``X`` and ``y`` with additional positive samples if needed.
    """
    if min_ratio <= 0:
        return X, y

    total = len(y)
    if total == 0:
        return X, y

    pos_mask = y > 0
    pos_count = int(pos_mask.sum())
    if pos_count == 0:
        return X, y

    current_ratio = pos_count / total
    if current_ratio >= min_ratio:
        return X, y

    required_pos = int(np.ceil(min_ratio * total))
    additional = required_pos - pos_count
    rng = np.random.default_rng(seed)
    pos_indices = np.flatnonzero(pos_mask)
    extra_indices = rng.choice(pos_indices, size=additional, replace=True)

    X_extra = X.iloc[extra_indices].copy()
    y_extra = y[extra_indices]
    X_aug = pd.concat([X, X_extra], ignore_index=True)
    y_aug = np.concatenate([y, y_extra])
    return X_aug, y_aug
