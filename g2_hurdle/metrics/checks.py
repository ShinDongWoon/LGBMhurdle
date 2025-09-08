
import pandas as pd
import numpy as np

def check_no_future_leakage(train_dates: pd.Series, val_dates: pd.Series):
    # The latest train date must be strictly < earliest val date
    if train_dates.max() >= val_dates.min():
        raise AssertionError("Temporal leakage: train max date >= val min date")

def sanity_pred_values(y_pred):
    import numpy as np
    if np.isnan(y_pred).any():
        raise AssertionError("NaN predictions detected")
    if (y_pred < 0).any():
        raise AssertionError("Negative predictions detected before clipping")
