
import numpy as np
import pandas as pd

def rolling_forecast_origin_split(df: pd.DataFrame, date_col: str, horizon: int, init_train_ratio: float):
    # Split by unique sorted dates
    udates = np.array(sorted(df[date_col].unique()))
    n = len(udates)
    start = max(1, int(n * init_train_ratio))
    while start + horizon <= n:
        train_end_date = udates[start-1]
        val_start_date = udates[start]
        val_end_date = udates[start + horizon - 1]
        train_mask = df[date_col] <= train_end_date
        val_mask = (df[date_col] >= val_start_date) & (df[date_col] <= val_end_date)
        yield train_mask.values, val_mask.values, (train_end_date, val_start_date, val_end_date)
        start += horizon
