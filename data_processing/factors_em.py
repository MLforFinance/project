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

def factors_em(x, kmax, jj, demean_type):
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
        print(f"Iteration {it}: error {err:.6f}, factors {icstar}")

    ehat = x - (chat * sdt + mut)
    return ehat, Fhat, lamhat, S**2, x2