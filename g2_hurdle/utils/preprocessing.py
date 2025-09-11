import numpy as np
import pandas as pd
from typing import Tuple


def ensure_min_positive_ratio(
    X: pd.DataFrame,
    y: np.ndarray,
    min_ratio: float,
    seed: int = 42,
    categorical_cols: list[str] | None = None,
    categories_map: dict[str, list[str]] | None = None,
    use_weights: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Ensure that the fraction of positive targets is at least ``min_ratio``.

    This function operates on a feature matrix ``X`` and corresponding target
    vector ``y``.  If the proportion of positive targets (``y > 0``) is below
    the requested ratio one of two strategies is applied depending on
    ``use_weights``:

    * ``use_weights=False`` (default): positive samples are resampled with
      replacement until the required ratio is approximately met, mimicking the
      previous behaviour.
    * ``use_weights=True``: the original data is kept intact and sample weights
      for positive examples are scaled instead of duplicating rows.  The weight
      scaling is equivalent to the resampling strategy so the model receives
      the same effective contribution from positive samples without inflating
      the dataset size.

    Both ``X`` and ``y`` are returned along with a ``sample_weight`` array
    suitable for model training.

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
    categorical_cols : list[str] | None, optional
        Columns to treat as categorical.  If provided, these columns will be
        converted to the ``category`` dtype prior to resampling.
    categories_map : dict[str, list[str]] | None, optional
        Mapping from column name to full list of categories.  When provided,
        category dtypes are created using the specified category order.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, np.ndarray]
        Possibly augmented ``X`` and ``y`` along with a ``sample_weight`` array.
    """
    # Restore categorical dtypes if columns were coerced to object by operations
    if categorical_cols:
        for c in categorical_cols:
            if c in X.columns:
                cats = categories_map.get(c) if categories_map else None
                X[c] = (
                    pd.Categorical(X[c], categories=cats)
                    if cats
                    else X[c].astype("category")
                )

    # Preserve original categorical columns to restore dtypes after augmentation
    cat_cols = X.select_dtypes(include="category").columns

    if min_ratio <= 0:
        return X, y, np.ones(len(y))

    total = len(y)
    if total == 0:
        return X, y, np.ones(len(y))

    pos_mask = y > 0
    pos_count = int(pos_mask.sum())
    if pos_count == 0:
        return X, y, np.ones(len(y))

    current_ratio = pos_count / total
    if current_ratio >= min_ratio:
        return X, y, np.ones(len(y))

    required_pos = int(np.ceil(min_ratio * total))
    sample_weight = np.ones(len(y))

    if use_weights:
        weight_pos = required_pos / pos_count
        sample_weight[pos_mask] = weight_pos
        X_aug, y_aug = X, y
    else:
        additional = required_pos - pos_count
        rng = np.random.default_rng(seed)
        pos_indices = np.flatnonzero(pos_mask)
        extra_indices = rng.choice(pos_indices, size=additional, replace=True)

        X_extra = X.iloc[extra_indices].copy()
        y_extra = y[extra_indices]
        X_aug = pd.concat([X, X_extra], ignore_index=True)
        y_aug = np.concatenate([y, y_extra])

        # Restore categorical dtypes using original category definitions
        for c in cat_cols:
            cats = (
                categories_map.get(c)
                if categories_map and c in categories_map
                else X[c].cat.categories
            )
            X_aug[c] = pd.Categorical(X_aug[c], categories=cats)

        sample_weight = np.ones(len(y_aug))

    return X_aug, y_aug, sample_weight
