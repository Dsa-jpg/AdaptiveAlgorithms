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

class SimpleMLP:
    def __init__(self, layer_sizes, activations=None, lr=0.01):

        self.layer_sizes = layer_sizes
        self.lr = lr
        self.n_layers = len(layer_sizes) - 1

        if activations is None:
            activations = ['relu'] * (self.n_layers - 1) + ['linear']
        self.activations = activations


        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
                        for i in range(self.n_layers)]
        self.biases = [np.zeros(layer_sizes[i + 1]) for i in range(self.n_layers)]

    def _activate(self, x, func):
        if func == 'relu':
            return np.maximum(0, x)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func == 'tanh':
            return np.tanh(x)
        elif func == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation {func}")

    def _activate_derivative(self, x, func):
        if func == 'relu':
            return (x > 0).astype(float)
        elif func == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif func == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif func == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation {func}")

    def forward(self, x):
        self.zs = []
        self.ys = [x]
        for w, b, act in zip(self.weights, self.biases, self.activations):
            z = self.ys[-1] @ w + b
            y = self._activate(z, act)
            self.zs.append(z)
            self.ys.append(y)
        return self.ys[-1]

    def backward(self, y_true):
        grad_w = [None] * self.n_layers
        grad_b = [None] * self.n_layers


        delta = (self.ys[-1] - y_true) * self._activate_derivative(self.zs[-1], self.activations[-1])

        for l in reversed(range(self.n_layers)):
            grad_w[l] = self.ys[l].T @ delta
            grad_b[l] = np.sum(delta, axis=0)
            if l > 0:
                delta = (delta @ self.weights[l].T) * self._activate_derivative(self.zs[l - 1], self.activations[l - 1])


        for l in range(self.n_layers):
            self.weights[l] -= self.lr * grad_w[l]
            self.biases[l] -= self.lr * grad_b[l]

        return np.mean((self.ys[-1] - y_true) ** 2)

    def predict(self, x):
        return self.forward(x)

