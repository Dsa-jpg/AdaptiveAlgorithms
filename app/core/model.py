from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def predict(self, x):
        pass


    @abstractmethod
    def preprocess(self, x):
        pass



class HONU(BaseModel):

    def __init__(self, degree: int):
        if degree not in [1,2,3]:
            raise ValueError('Degree must be between 1 to 3')
        self.degree = degree
        self.weights = None # TODO by 2. and 3. person

    def preprocess(self, x):
        x = np.asarray(x).flatten()
        features = []
        for d in range(1, self.degree + 1):
            idx = self._generate_exponents(len(x), d)
            for exps in idx:
                features.append(np.prod(x ** exps))
        return np.array(features)

    def _generate_exponents(self, dim, degree):

        if degree == 0:
            return [np.zeros(dim, dtype=int)]
        if dim == 1:
            return [np.array([degree])]

        result = []
        for i in range(degree + 1):
            for tail in self._generate_exponents(dim - 1, degree - i):
                result.append(np.concatenate(([i], tail)))
        return result

    def predict(self, x):
        if self.weights is None:
            return 0.0

        phi = self.preprocess(x)
        return float(np.dot(self.weights, phi))




class MLP(BaseModel):

    def __init__(self, layers: int, neurons: int, activation: str):
        if layers < 1:
            raise ValueError("Must be at least 1 hidden layer.")
        if neurons < 1:
            raise ValueError("Neuron count must be positive.")
        if activation not in ['relu', 'sigmoid','tanh']:
            raise ValueError("Activation must be 'relu', 'sigmoid' or 'tanh'.")
        self.layers = layers
        self.neurons = neurons
        self.activation_fun = activation
        self.weights = None  # TODO by 2. and 3. person
        self.biases = None

        self.activation = {
            "tanh": lambda x: np.tanh(x),
            "relu": lambda x: np.maximum(0, x),
            "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        }[activation]

    def preprocess(self, x):
        return np.asarray(x).flatten()

    def _forward(self, x):
        if self.weights is None:
            return 0.0

        a = x
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self.activation(np.dot(W, a) + b)

        a = np.dot(self.weights[-1], a) + self.biases[-1]
        return float(a)

    def predict(self, x):
        return self._forward(self.preprocess(x))