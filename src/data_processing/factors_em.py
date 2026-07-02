import numpy as np
from scipy.linalg import svd


def transform_data(x, demean_type):
    T, N = x.shape
    if demean_type == 0:
        mut, sdt = np.zeros((T, N)), np.ones((T, N))
    elif demean_type == 1:
        m = np.mean(x, axis=0)
        mut, sdt = np.tile(m, (T, 1)), np.ones((T, N))
    elif demean_type == 2:
        m, s = np.mean(x, axis=0), np.std(x, axis=0)
        mut, sdt = np.tile(m, (T, 1)), np.tile(s, (T, 1))
    elif demean_type == 3:
        mut = np.array([np.mean(x[:t+1, :], axis=0) for t in range(T)])
        sdt = np.tile(np.std(x, axis=0), (T, 1))

    x_trans = (x - mut) / sdt
    return x_trans, mut, sdt


def baing(X, kmax, jj):
    T, N = X.shape
    NT = N * T
    NT1 = N + T
    GCT = min(N, T)

    ii = np.arange(1, kmax + 1)
    if jj == 1: CT = np.log(NT/NT1) * ii * (NT1/NT)
    elif jj == 2: CT = (NT1/NT) * np.log(GCT) * ii
    elif jj == 3: CT = ii * np.log(GCT) / GCT

    U, S, Vh = svd(X, full_matrices=False)

    IC = np.zeros(kmax + 1)
    for k in range(1, kmax + 1):
        Fk = U[:, :k] * np.sqrt(T)
        Lk = Vh[:k, :].T * np.sqrt(N) # Note: approx
        chat = Fk @ (X.T @ Fk / T).T
        ehat = X - chat
        sigma = np.mean(np.sum(ehat**2, axis=0) / T)
        IC[k-1] = np.log(sigma) + CT[k-1]

    IC[kmax] = np.log(np.mean(np.sum(X**2, axis=0) / T))

    ic_star = np.argmin(IC) + 1
    if ic_star > kmax: ic_star = 0
    return ic_star


def factors_em(x, kmax, jj, demean_type, verbose=True):
    T, N = x.shape
    maxit = 50
    err = 999
    it = 0

    nan_mask = np.isnan(x)
    x2 = np.where(nan_mask, np.nanmean(x, axis=0), x)

    chat0 = np.zeros((T, N))

    while err > 1e-6 and it < maxit:
        it += 1
        x3, mut, sdt = transform_data(x2, demean_type)

        icstar = 8 if kmax == 99 else baing(x3, kmax, jj)

        U, S, Vh = svd(x3, full_matrices=False)
        Fhat = U[:, :icstar] * np.sqrt(N)
        lamhat = Vh[:icstar, :].T
        chat = Fhat @ lamhat.T

        x2 = np.where(nan_mask, chat * sdt + mut, x)

        diff = chat - chat0
        err = np.sum(diff**2) / np.sum(chat0**2) if it > 1 else 999
        chat0 = chat.copy()
        if verbose:
            print(f"Iteration {it}: error {err:.6f}, factors {icstar}")

    ehat = x - (chat * sdt + mut)
    return ehat, Fhat, lamhat, S**2, x2


# ---------------------------------------------------------------------------
# Leakage-free imputation alternatives
# ---------------------------------------------------------------------------

def _locf_fill(x):
    """Forward-fill NaNs column by column without using future observations."""
    T, N = x.shape
    result = x.copy()
    for j in range(N):
        last_valid = 0.0
        for t in range(T):
            if np.isnan(result[t, j]):
                result[t, j] = last_valid
            else:
                last_valid = result[t, j]
    return result


def impute_locf(x, kmax, jj, demean_type):
    """Impute missing values by Last Observation Carried Forward — no EM, zero look-ahead.

    A post-hoc SVD on the imputed data provides factors/loadings for mrsq reporting
    only; it does not affect which values are imputed.

    Returns the same 5-tuple as factors_em for a uniform interface:
        (ehat, Fhat, lamhat, S**2, x2)
    """
    T, N = x.shape
    x2 = _locf_fill(x)

    x3, _, _ = transform_data(x2, demean_type)
    icstar = max(1, baing(x3, kmax, jj))
    U, S, Vh = svd(x3, full_matrices=False)
    Fhat = U[:, :icstar] * np.sqrt(N)
    lamhat = Vh[:icstar, :].T

    ehat = np.where(np.isnan(x), 0.0, x - x2)
    return ehat, Fhat, lamhat, S**2, x2


def factors_em_burnin(x, kmax, jj, demean_type, burn_in=60):
    """EM imputation with burn-in period: fit on the first burn_in rows, then project forward.

    Strategy:
      1. Run the full EM on rows 0..burn_in-1 to obtain loadings and
         standardisation statistics that are free of look-ahead for t < burn_in.
      2. For every row t >= burn_in, estimate the latent factor from the *observed*
         entries at time t using the frozen loadings, then reconstruct missing entries.
         Falls back to LOCF when fewer than k observations are available.

    The returned Fhat/lamhat come from a post-hoc SVD on the full imputed matrix,
    so mrsq metrics reflect the complete time series.

    Returns the same 5-tuple as factors_em:
        (ehat, Fhat, lamhat, S**2, x2)
    """
    T, N = x.shape
    burn_in = min(burn_in, T)

    # --- Step 1: EM on burn-in window only ---
    _, _, lamhat_burn, _, x2_burn = factors_em(x[:burn_in, :], kmax, jj, demean_type, verbose=False)
    k = lamhat_burn.shape[1]

    # Frozen standardisation params derived from the burn-in imputed data
    mean_burn = np.nanmean(x2_burn, axis=0)
    std_burn = np.nanstd(x2_burn, axis=0)
    std_burn = np.where(std_burn < 1e-8, 1.0, std_burn)

    result = np.empty((T, N))
    result[:burn_in, :] = x2_burn

    # --- Step 2: causal projection for t >= burn_in ---
    for t in range(burn_in, T):
        row = x[t, :].copy()
        nan_mask_t = np.isnan(row)
        if not nan_mask_t.any():
            result[t, :] = row
            continue

        obs_mask = ~nan_mask_t
        if obs_mask.sum() > k:
            # Standardise with burn-in stats, estimate factor from observed entries,
            # impute missing entries, denormalise.
            z_t = (row - mean_burn) / std_burn
            f_t, _, _, _ = np.linalg.lstsq(lamhat_burn[obs_mask, :], z_t[obs_mask], rcond=None)
            z_t[nan_mask_t] = lamhat_burn[nan_mask_t, :] @ f_t
            row = z_t * std_burn + mean_burn
        else:
            # Not enough observed entries: fall back to LOCF
            for j in np.where(nan_mask_t)[0]:
                prev = result[:t, j]
                valid = prev[~np.isnan(prev)]
                row[j] = valid[-1] if len(valid) > 0 else mean_burn[j]
        result[t, :] = row

    # --- Post-hoc factors on full imputed data for mrsq reporting ---
    x3_full, _, _ = transform_data(result, demean_type)
    U_full, S_full, Vh_full = svd(x3_full, full_matrices=False)
    Fhat_full = U_full[:, :k] * np.sqrt(N)
    lamhat_full = Vh_full[:k, :].T

    ehat = np.where(np.isnan(x), 0.0, x - result)
    return ehat, Fhat_full, lamhat_full, S_full**2, result


IMPUTATION_METHODS = ("em", "locf", "em_burnin")


def impute_by_method(x, method="em", kmax=8, jj=2, demean_type=2, burn_in=60):
    """Dispatch to the selected imputation strategy.

    em        — Original EM algorithm (has look-ahead leakage, fastest).
    locf      — Last Observation Carried Forward (no leakage, simplest).
    em_burnin — EM fit on the first burn_in months; remaining rows projected forward
                using frozen loadings (minimal leakage, recommended).
    """
    if method == "em":
        return factors_em(x, kmax, jj, demean_type)
    elif method == "locf":
        return impute_locf(x, kmax, jj, demean_type)
    elif method == "em_burnin":
        return factors_em_burnin(x, kmax, jj, demean_type, burn_in=burn_in)
    else:
        raise ValueError(f"Unknown imputation_method {method!r}. Choose from {IMPUTATION_METHODS}.")
