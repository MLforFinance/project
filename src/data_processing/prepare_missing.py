import numpy as np
import pandas as pd

def transxf(x, tcode):
    n = len(x)
    y = np.full(n, np.nan)
    small = 1e-6

    if tcode == 1: # Level
        y = x
    elif tcode == 2: # First difference
        y[1:] = np.diff(x)
    elif tcode == 3: # Second difference
        y[2:] = x[2:] - 2*x[1:-1] + x[:-2]
    elif tcode == 4: # Log
        if np.nanmin(x) > small: y = np.log(x)
    elif tcode == 5: # First diff of log
        if np.nanmin(x) > small:
            lx = np.log(x)
            y[1:] = np.diff(lx)
    elif tcode == 6: # Second diff of log
        if np.nanmin(x) > small:
            lx = np.log(x)
            y[2:] = lx[2:] - 2*lx[1:-1] + lx[:-2]
    elif tcode == 7: # First diff of % change
        y1 = np.zeros(n)
        y1[1:] = (x[1:] - x[:-1]) / x[:-1]
        y[2:] = y1[2:] - y1[1:-1]
        
    return y

def prepare_missing(rawdata, tcode):
    yt = []
    for i in range(rawdata.shape[1]):
        yt.append(transxf(rawdata[:, i], tcode[i]))
    return np.array(yt).T


def compute_NA(data : pd.DataFrame):
    return (data.isna().sum(axis = 0) / data.shape[0]).sort_values(ascending=False)