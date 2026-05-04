import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
from pathlib import Path

def get_transition_matrix(states):
    df = pd.DataFrame({'current': states, 'next': states.shift(-1)}).dropna()
    contingency = pd.crosstab(df['current'], df['next'])
    transition = contingency.div(contingency.sum(axis=1), axis=0)
    return contingency, transition

def test_order_0_vs_1(states):
    contingency, _ = get_transition_matrix(states)
    stat, p_value, dof, expected = chi2_contingency(contingency)
    
    return {
        "chi2_stat": stat,
        "p_value": p_value,
        "is_order_1_better": p_value < 0.05 
    }

def test_order_1_vs_2(states):
    df = pd.DataFrame({
        'prev': states.shift(2), 
        'current': states.shift(1), 
        'next': states
    }).dropna()
    
    total_chi2 = 0
    total_dof = 0
    
    for state in df['current'].unique():
        subset = df[df['current'] == state]
        if len(subset['prev'].unique()) > 1 and len(subset['next'].unique()) > 1:
            contingency = pd.crosstab(subset['prev'], subset['next'])
            stat, _, dof, _ = chi2_contingency(contingency)
            total_chi2 += stat
            total_dof += dof
            
    p_value = chi2.sf(total_chi2, total_dof) if total_dof > 0 else 1.0
    
    return {
        "chi2_stat": total_chi2,
        "dof": total_dof,
        "p_value": p_value,
        "is_order_2_better": p_value < 0.05
    }

def test_aic_bic(states):
    states = states.dropna()
    N = len(states)
    S = len(states.unique())
    
    # Order 0
    counts_0 = states.value_counts()
    probs_0 = counts_0 / N
    ll_0 = sum(counts_0[s] * np.log(probs_0[s]) for s in counts_0.index if probs_0[s] > 0)
    k_0 = S - 1
    
    # Order 1
    df1 = pd.DataFrame({'current': states.shift(1), 'next': states}).dropna()
    counts_1 = pd.crosstab(df1['current'], df1['next'])
    probs_1 = counts_1.div(counts_1.sum(axis=1), axis=0)
    ll_1 = sum(counts_1.loc[c, n] * np.log(probs_1.loc[c, n]) 
               for c in counts_1.index for n in counts_1.columns if counts_1.loc[c, n] > 0)
    k_1 = S * (S - 1)
    
    # Order 2
    df2 = pd.DataFrame({'prev': states.shift(2), 'current': states.shift(1), 'next': states}).dropna()
    counts_2 = df2.groupby(['prev', 'current', 'next']).size()
    sums_2 = counts_2.groupby(level=[0, 1]).sum()
    ll_2 = sum(n * np.log(n / sums_2[idx[:2]]) for idx, n in counts_2.items() if n > 0)
    k_2 = (S**2) * (S - 1)
    
    return {
        "AIC": {0: 2*k_0 - 2*ll_0, 1: 2*k_1 - 2*ll_1, 2: 2*k_2 - 2*ll_2},
        "BIC": {0: k_0*np.log(N) - 2*ll_0, 1: k_1*np.log(N) - 2*ll_1, 2: k_2*np.log(N) - 2*ll_2}
    }

def test_lag_k(states, k=2):
    states = states.dropna()

    _, P_1 = get_transition_matrix(states)
    P_k_theoretical = np.linalg.matrix_power(P_1.values, k)
    P_k_theo_df = pd.DataFrame(P_k_theoretical, index=P_1.index, columns=P_1.columns)
    
    df_k = pd.DataFrame({'current': states, 'future': states.shift(-k)}).dropna()
    observed_counts = pd.crosstab(df_k['current'], df_k['future'])
    
    observed_counts = observed_counts.reindex(index=P_1.index, columns=P_1.columns, fill_value=0)
    
    row_sums = observed_counts.sum(axis=1)
    expected_counts = P_k_theo_df.multiply(row_sums, axis=0)
    
    chi2_stat = 0
    dof = 0
    
    for c in observed_counts.index:
        for f in observed_counts.columns:
            O = observed_counts.loc[c, f]
            E = expected_counts.loc[c, f]
            if E > 0:
                chi2_stat += ((O - E)**2) / E
        dof += len(observed_counts.columns) - 1
        
    p_value = chi2.sf(chi2_stat, dof)
    
    return {
        "chi2_stat": chi2_stat,
        "dof": dof,
        "p_value": p_value,
        "is_markov_violated": p_value < 0.05
    }

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR.parent.parent / "data" / "2026-02-MD_regimes.csv"
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[Warning] File not found at {data_path}. Using dummy data.\n")
        np.random.seed(42)
        df = pd.DataFrame({'regime': np.random.choice(['Bull', 'Bear', 'Stagnant'], size=500)})
        
    states = df['regime']
    
    _, probs = get_transition_matrix(states)
    print("Transition Matrix (Probabilities):\n", probs.round(4), "\n")
    print("-" * 50)
    
    res_0_vs_1 = test_order_0_vs_1(states)
    print(f"Test 1: Independence (Order 0) vs Markovian (Order 1)")
    print(f"P-value: {res_0_vs_1['p_value']:.4f}")
    print(f"Is Order 1 better than Order 0? {res_0_vs_1['is_order_1_better']}\n")
    print("-" * 50)

    res_1_vs_2 = test_order_1_vs_2(states)
    print(f"Test 2: Markovian (Order 1) vs Higher Order (Order 2)")
    print(f"P-value: {res_1_vs_2['p_value']:.4f}")
    print(f"Is Order 2 better than Order 1? {res_1_vs_2['is_order_2_better']}\n")
    print("-" * 50)

    ic_results = test_aic_bic(states)
    print("Test 3: Information Criteria (Lower score is better)")
    for metric, values in ic_results.items():
        best_order = min(values, key=values.get)
        print(f"{metric} - Order 0: {values[0]:.2f} | Order 1: {values[1]:.2f} | Order 2: {values[2]:.2f} -> Best: Order {best_order}")


    print("-" * 50)
    res_lag = test_lag_k(states, k=2)
    print(f"Test 4: Lag Test (Empirical k=2 vs Theoretical P^2)")
    print(f"P-value: {res_lag['p_value']:.4f}")
    print(f"Is the Markov assumption violated at lag 2? {res_lag['is_markov_violated']}\n")