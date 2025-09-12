
def calibrate_proba(p_train, y_train_bin, p_valid=None, method="isotonic"):
    \"\"\"Optional probability calibration; returns a callable transform.
    If scikit-learn not available, return identity.
    \"\"\"
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        import numpy as np
    except Exception:
        return lambda x: x

    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p_train, y_train_bin)
        return lambda x: ir.predict(x)
    elif method == "platt":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(p_train.reshape(-1,1), y_train_bin)
        return lambda x: lr.predict_proba(x.reshape(-1,1))[:,1]
    else:
        return lambda x: x
