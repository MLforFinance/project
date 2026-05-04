import pandas as pd


def EMA_feature(X: pd.DataFrame,
                n:int)->pd.DataFrame:
    return X.ewm(n)


def feature_engineering(
        X:pd.DataFrame,
        increments:bool = True,
        EMA:bool = True,
        n_EMA:int = 3
    )-> pd.DataFrame:


    if increments and EMA: return pd.concat([EMA_feature(X, n = 1), EMA_feature(X, n = n_EMA)])
    elif increments: return EMA_feature(X,n = 1)
    elif EMA: return EMA_feature(X, n = n_EMA)
