import umap
import pandas as pd
import numpy as np
from hmmlearn import hmm

def fit_umap(X: pd.DataFrame,
             n_components: int = 4,
             n_neighbors: int = 15,
             min_dist: float = 0.0,
             random_state: int = 42,
             metric:str = "cosine",
             epochs:int = 200):
    
    mapper = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric=metric,
        n_epochs = epochs
    )

    X_reduced = mapper.fit_transform(X.values)

    X_reduced_df = pd.DataFrame(
        X_reduced, 
        index=X.index,
        columns=[f"UMAP_{i+1}" for i in range(n_components)]
    )

    return X_reduced_df, mapper
