import numpy as np
from itertools import combinations_with_replacement

class BaseHONU:
    def __init__(self, degree, n_inputs):
        self.degree = degree
        self.n_inputs = n_inputs
        self.n_weights = self._count_weights()
        self.w = np.zeros(self.n_weights)

    def _count_weights(self):
        n = self.n_inputs
        if self.degree == 1:
            return n
        if self.degree == 2:
            return n + (n*(n+1))//2
        if self.degree == 3:
            return n + (n*(n+1))//2 + (n*(n+1)*(n+2))//6
        raise ValueError("Unsupported degree")

    def _x_vec(self, x):
        x = np.asarray(x).flatten()
        feats = list(x)

        if self.degree >= 2:
            feats += [x[i]*x[j]
                      for i, j in combinations_with_replacement(range(self.n_inputs), 2)]

        if self.degree >= 3:
            feats += [x[i]*x[j]*x[k]
                      for i, j, k in combinations_with_replacement(range(self.n_inputs), 3)]

        return np.array(feats)

    def predict(self, x):
        return float(self.w @ self._x_vec(x))

class HONU(BaseHONU):
    def __init__(self, degree, n_inputs, mu=0.001, l_method="LMS", delta=1.0):
        super().__init__(degree, n_inputs)
        self.mu = mu
        self.method = l_method.upper()

        # RLS-specific
        if self.method == "RLS":
            self.P = np.eye(self.n_weights) * delta  # inicializace inverzní matice

    def update(self, x, y):
        x_vec = self._x_vec(x)
        y_hat = self.w @ x_vec
        error = y - y_hat

        if self.method == "LMS":
            self.w += self.mu * error * x_vec

        elif self.method == "NGD":
            norm = x_vec @ x_vec + 1e-8
            self.w += (self.mu / norm) * error * x_vec

        elif self.method == "RLS":
            x_vec = x_vec.reshape(-1,1)
            P_x = self.P @ x_vec
            k = P_x / (1 + x_vec.T @ P_x)
            self.w += (k.flatten() * error)
            self.P = self.P - k @ P_x.T

        else:
            raise ValueError(f"Unknown method {self.method}")

        return error

class SlidingHONU(BaseHONU):
    def __init__(self, degree, n_inputs, window_size=50, lam=0.01):
        super().__init__(degree, n_inputs)
        self.M = window_size
        self.lam = lam
        self.X_window = []
        self.y_window = []

    def add_sample(self, x, y):
        self.X_window.append(self._x_vec(x))
        self.y_window.append(y)

        if len(self.X_window) > self.M:
            self.X_window.pop(0)
            self.y_window.pop(0)

    def update_LM(self):
        if not self.X_window:
            return 0.0, np.zeros_like(self.w)

        X = np.vstack(self.X_window)
        y = np.array(self.y_window)

        A = X.T @ X + self.lam * np.eye(self.n_weights)
        b = X.T @ y

        w_old = self.w.copy()
        self.w = np.linalg.solve(A, b)

        error = y - X @ self.w
        return np.mean(error**2), self.w - w_old
