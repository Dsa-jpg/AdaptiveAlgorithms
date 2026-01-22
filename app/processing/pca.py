import numpy as np
import pandas as pd
from pandas import DataFrame

def pca_manual(
    X: DataFrame,
    n_components: int
) -> DataFrame:
    """
    PCA based on eigen-decomposition of correlation matrix.
    Used mainly to remove collinearity in embeddings.
    """
    C = np.corrcoef(X.values.T)
    d, v = np.linalg.eig(C)
    idx = np.argsort(d)[::-1]
    V = v[:, idx[:n_components]]

    X_pca = X.values @ V

    return pd.DataFrame(
        X_pca,
        index=X.index,
        columns=[f"PCA{i+1}" for i in range(n_components)]
    )
