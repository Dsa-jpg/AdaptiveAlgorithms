import numpy as np


def zscore(data):
    data = np.asarray(data, dtype=float)
    std = data.std()
    if std == 0:
        return np.zeros_like(data)
    return (data - data.mean()) / std

def minmax(data):
    data = np.asarray(data, dtype=float)
    d_min = np.min(data)
    d_max = np.max(data)

    if d_max == d_min:
        return np.zeros_like(data)

    return (data - d_min) / (d_max - d_min)