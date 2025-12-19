import numpy as np
from itertools import combinations_with_replacement

class HONU:
    def __init__(self, degree, n_inputs, mu=0.001, l_method="LMS"):
        self.degree = degree
        self.n_inputs = n_inputs
        self.mu = mu
        self.method = l_method.upper()
        self.n_weights = self._count_weights()
        self.w = np.zeros(self.n_weights)

    def _count_weights(self):
        n = self.n_inputs
        if self.degree==1: return n
        if self.degree==2: return n + (n*(n+1))//2
        if self.degree==3: return n + (n*(n+1))//2 + (n*(n+1)*(n+2))//6

    def _x_vec(self, x):
        x = np.asarray(x).flatten()
        x_vec = list(x)
        if self.degree>=2:
            for i,j in combinations_with_replacement(range(self.n_inputs),2):
                x_vec.append(x[i]*x[j])
        if self.degree>=3:
            for i,j,k in combinations_with_replacement(range(self.n_inputs),3):
                x_vec.append(x[i]*x[j]*x[k])
        return np.array(x_vec)

    def predict(self, x):
        x_vec = self._x_vec(x)
        return float(np.dot(self.w, x_vec))

    def update(self, x, y):
        x_vec = self._x_vec(x)
        y_hat = np.dot(self.w, x_vec)
        error = y - y_hat

        if self.method=="LMS":
            self.w += self.mu * error * x_vec
        elif self.method=="NGD":
            norm = np.dot(x_vec, x_vec) + 1e-8
            self.w += (self.mu/norm) * error * x_vec
        else:
            raise ValueError(f"Unknown method {self.method}")
        return error

class SlidingHONU:
    def __init__(self, degree, n_inputs, window_size=50, lam=0.01):
        self.degree = degree
        self.n_inputs = n_inputs
        self.M = window_size
        self.lam = lam
        self.n_weights = self._count_weights()
        self.w = np.zeros(self.n_weights)
        self.X_window = []
        self.y_window = []

    def _count_weights(self):
        n = self.n_inputs
        if self.degree==1: return n
        if self.degree==2: return n + (n*(n+1))//2
        if self.degree==3: return n + (n*(n+1))//2 + (n*(n+1)*(n+2))//6

    def _x_vec(self, x):
        x = np.asarray(x).flatten()
        x_vec = list(x)
        if self.degree>=2:
            for i,j in combinations_with_replacement(range(self.n_inputs),2):
                x_vec.append(x[i]*x[j])
        if self.degree>=3:
            for i,j,k in combinations_with_replacement(range(self.n_inputs),3):
                x_vec.append(x[i]*x[j]*x[k])
        return np.array(x_vec)

    def predict(self, x):
        x_vec = self._x_vec(x)
        return float(np.dot(self.w, x_vec))

    def add_sample(self, x, y):
        self.X_window.append(self._x_vec(x))
        self.y_window.append(y)
        if len(self.X_window) > self.M:
            self.X_window.pop(0)
            self.y_window.pop(0)

    def update_LM(self):
        if len(self.X_window)<1: return 0.0
        X_mat = np.vstack(self.X_window)
        y_vec = np.array(self.y_window)
        A = X_mat.T @ X_mat + self.lam*np.eye(self.n_weights)
        b = X_mat.T @ y_vec
        w_old = self.w.copy()
        self.w = np.linalg.solve(A,b)
        delta_w = self.w - w_old
        y_hat = X_mat @ self.w
        error = y_vec - y_hat
        return np.mean(error**2), delta_w



