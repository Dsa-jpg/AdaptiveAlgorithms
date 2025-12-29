import numpy as np

def build_input_vector(df, y_cols, lags, target_signal, k):
    x_vec = []
    for col in y_cols:
        if col == target_signal:
            continue
        lag = lags[col]
        x_vec.extend(df[col].values[k-lag:k])
    return np.array(x_vec).reshape(1, -1)

def initialize_storage(N, model_type, model):
    n_weights = model.n_weights if model_type != "MLP" else sum(w.size for w in model.weights)
    wall = np.zeros((N, n_weights))
    delta_wall = np.zeros_like(wall)
    y_pred = np.zeros(N)
    error = np.zeros(N)
    return y_pred, error, wall, delta_wall
