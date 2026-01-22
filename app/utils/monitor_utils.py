import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def compute_LE_window(error, window_size=50):
    if len(error) < window_size:
        return np.array([])
    return np.convolve(error**2, np.ones(window_size)/window_size, mode='valid')

def project_weights_2D(wall, method="PCA"):
    if wall.shape[0] < 2:
        return np.array([])
    if method == "PCA":
        proj = PCA(n_components=2).fit_transform(wall)
    elif method == "t-SNE":
        proj = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(wall)
    else:
        raise ValueError("Unknown projection method")
    return proj
