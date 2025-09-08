import numpy as np
import pandas as pd


def ensure_min_positive_ratio(df: pd.DataFrame, target_col: str, min_ratio: float, seed: int = 42) -> pd.DataFrame:
    """Ensure that the fraction of rows with positive target values is at least `min_ratio`.

    If the current positive ratio is below the desired threshold, positive rows
    are resampled with replacement until the ratio is satisfied.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the target column.
    target_col : str
        Name of the target column.
    min_ratio : float
        Minimum required ratio of positive target samples (0-1 range).
    seed : int, optional
        Random seed for sampling, by default 42.

    Returns
    -------
    pd.DataFrame
        Dataframe augmented with additional positive samples if needed.
    """
    if min_ratio <= 0:
        return df

    total = len(df)
    if total == 0:
        return df

    pos_mask = df[target_col] > 0
    pos_count = int(pos_mask.sum())
    if pos_count == 0:
        return df

    current_ratio = pos_count / total
    if current_ratio >= min_ratio:
        return df

    required_pos = int(np.ceil(min_ratio * total))
    additional = required_pos - pos_count
    pos_df = df.loc[pos_mask]
    extra = pos_df.sample(n=additional, replace=True, random_state=seed)
    df_aug = pd.concat([df, extra], ignore_index=True)
    return df_aug
