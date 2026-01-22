import pandas as pd
from pandas import DataFrame

def build_lag_embedding(
    df: DataFrame,
    signals: list[str],
    lags: dict[str, int]
) -> DataFrame:
    """
    Constructs a lagged embedding matrix for selected signals.

    X = [y1(k), y1(k-1), ..., y2(k), y2(k-1), ...]
    """
    embedded = pd.DataFrame(index=df.index)

    for signal in signals:
        for lag in range(lags[signal] + 1):
            embedded[f"{signal}_t-{lag}"] = df[signal].shift(lag)

    return embedded.dropna()
