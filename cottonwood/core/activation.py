import numpy as np

# All of these need to be able to handle 2D numpy arrays as inputs.


class Logistic(object):
    def __init__(self):
        # Including this class attribute lets Logistic re-use its results.
        # Caching the result in this way speeds up the derivative
        # calculation on the backward pass.
        self.calc_fwd = None

    def __str__(self):
        return "logistic"

    def calc(self, v):
        self.calc_fwd = 1 / (1 + np.exp(-v))
        return self.calc_fwd

    def calc_d(self, v):
        return self.calc_fwd * (1 - self.calc_fwd)


class Sigmoid(Logistic):
    def __str__(self):
        return "sigmoid"


class ReLU(object):
    @staticmethod
    def __str__():
        return "ReLU"

    @staticmethod
    def calc(v):
        return np.maximum(0, v)

    @staticmethod
    def calc_d(v):
        derivative = np.zeros(v.shape)
        derivative[np.where(v > 0)] = 1
        return derivative


class Tanh(object):
    def __init__(self):
        # Including this class attribute lets Tanh re-use its results.
        # Caching the result in this way speeds up the derivative
        # calculation on the backward pass.
        self.calc_fwd = None

    def __str__(self):
        return "hyperbolic tangent"

    def calc(self, v):
        self.calc_fwd = np.tanh(v)
        return self.calc_fwd

    def calc_d(self, v):
        return 1 - self.calc_fwd ** 2
