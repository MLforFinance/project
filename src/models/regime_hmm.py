import warnings
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt



transmat_prior = np.array([
    [20,10,1,1],
    [1,50,10,1],
    [1,1,100,5],
    [10,1,1,30]
])

# => solve thrashing, especially relevant if we implement transaction costs

def fit_gmm_hmm(X: pd.DataFrame,
                n_components: int = 5,
                covariance_type: str = "full",
                n_iter: int = 1000,
                random_state: int = 42,
                tol: float = 1e-3,
                plot_convergence: bool = True):

    # Reduce n_iter if window is too small
    n_samples = len(X)
    effective_n_iter = min(n_iter, max(10, n_samples // 2))

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=effective_n_iter,
        random_state=random_state,
        tol=tol
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit(X.values)
    except (Exception, ConvergenceWarning):
        return None, None, None

    if plot_convergence:
        try:
            history = model.monitor_.history
            plt.figure(figsize = (8,5))
            plt.plot(history, marker = "o", linestyle = '-')
            plt.title(f"EM Algo convergence")
            plt.xlabel("Iteration")
            plt.ylabel("Log Likelihood")
            plt.show()
        except Exception:
            pass

    states = model.predict(X.values)
    state_probs = model.predict_proba(X.values)

    states_series = pd.Series(
        states,
        index=X.index,
        name="regime"
    )

    state_probs_df = pd.DataFrame(
        state_probs,
        index=X.index,
        columns=[f"prob_regime_{i+1}" for i in range(n_components)]
    )

    return states_series, state_probs_df, model