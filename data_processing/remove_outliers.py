import numpy as np

def remove_outliers(X):
    """
    Replace outliers by NaN. 
    Criterion : abs(x - median) > 10 * IQR
    """
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