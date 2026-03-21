import numpy as np


def mrsq(Fhat, lamhat, ve2, series_names):
    series_names = np.asarray(series_names)

    N, ic = lamhat.shape
    R2 = np.zeros((N, ic))
    mR2 = np.zeros((N, ic))

    for i in range(ic):
        R2[:, i] = np.var(Fhat[:, : i + 1] @ lamhat[:, : i + 1].T, axis=0)
        mR2[:, i] = np.var(Fhat[:, i: i + 1] @ lamhat[:, i: i + 1].T, axis=0)

    mR2_F = ve2[:ic] / np.sum(ve2)

    t10_s = []
    t10_mR2 = []
    for i in range(ic):
        idx = np.argsort(mR2[:, i])[::-1][:10]
        t10_s.append(series_names[idx])
        t10_mR2.append(mR2[idx, i])

    return R2, mR2, mR2_F, np.sum(mR2_F), t10_s, t10_mR2
