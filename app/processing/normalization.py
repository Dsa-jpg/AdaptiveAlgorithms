import numpy as np


def zscore(data):
    data = np.asarray(data, dtype=float)
    std = data.std()
    if std == 0:
        return np.zeros_like(data)
    return (data - data.mean()) / std

def minmax(data):
    data = np.asarray(data, dtype=float)
    return (data - np.min(data)) / np.max(data)