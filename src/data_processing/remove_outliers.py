import numpy as np


def remove_outliers(X):
    """Replace outliers by NaN using global IQR criterion: abs(x - median) > 10 * IQR"""
    median_X = np.nanmedian(X, axis=0)
    Q1 = np.nanpercentile(X, 25, axis=0)
    Q3 = np.nanpercentile(X, 75, axis=0)
    IQR = Q3 - Q1

    limit = 10 * IQR
    outlier_mask = np.abs(X - median_X) > limit

    Y = X.copy()
    Y[outlier_mask] = np.nan

    n = np.sum(outlier_mask, axis=0)
    return Y, n


def remove_outliers_expanding(X, min_obs=24):
    """Point-in-time outlier detection using expanding-window IQR. Outliers replaced with NaN.

    For each time t, computes median/IQR using only rows 0..t-1 (no look-ahead).
    The first min_obs rows are never flagged — there is not yet enough history to
    compute a stable IQR.
    """
    T, N = X.shape
    Y = X.copy()
    n = np.zeros(N, dtype=int)
    for t in range(min_obs, T):
        window = X[:t, :]
        median_t = np.nanmedian(window, axis=0)
        Q1_t = np.nanpercentile(window, 25, axis=0)
        Q3_t = np.nanpercentile(window, 75, axis=0)
        IQR_t = Q3_t - Q1_t
        limit_t = 10 * IQR_t
        mask_t = np.abs(X[t, :] - median_t) > limit_t
        Y[t, mask_t] = np.nan
        n += mask_t.astype(int)
    return Y, n


def remove_outliers_locf(X, min_obs=24):
    """Point-in-time outlier detection using expanding-window IQR. Outliers replaced with last valid observation.

    Same detection logic as remove_outliers_expanding, but instead of setting outliers
    to NaN they are replaced with the most recent non-NaN value in that column.
    Falls back to NaN only when no prior valid observation exists.
    """
    T, N = X.shape
    Y = X.copy()
    n = np.zeros(N, dtype=int)
    for t in range(min_obs, T):
        window = X[:t, :]
        median_t = np.nanmedian(window, axis=0)
        Q1_t = np.nanpercentile(window, 25, axis=0)
        Q3_t = np.nanpercentile(window, 75, axis=0)
        IQR_t = Q3_t - Q1_t
        limit_t = 10 * IQR_t
        mask_t = np.abs(X[t, :] - median_t) > limit_t
        for j in np.where(mask_t)[0]:
            prev = Y[:t, j]
            valid = prev[~np.isnan(prev)]
            Y[t, j] = valid[-1] if len(valid) > 0 else np.nan
        n += mask_t.astype(int)
    return Y, n


OUTLIER_METHODS = ("global", "expanding", "locf")


def remove_outliers_by_method(X, method="global", min_obs=24):
    """Dispatch to the selected outlier-removal strategy.

    global    — Full-dataset IQR (original behaviour, introduces look-ahead leakage).
    expanding — Point-in-time expanding-window IQR; detected outliers become NaN.
    locf      — Point-in-time expanding-window IQR; detected outliers replaced with
                last valid observation (no NaN introduced by outlier detection).
    """
    if method == "global":
        return remove_outliers(X)
    elif method == "expanding":
        return remove_outliers_expanding(X, min_obs=min_obs)
    elif method == "locf":
        return remove_outliers_locf(X, min_obs=min_obs)
    else:
        raise ValueError(f"Unknown outlier_method {method!r}. Choose from {OUTLIER_METHODS}.")
