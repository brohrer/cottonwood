import numpy as np

# All of these need to be able to handle 2D numpy arrays as inputs.


class Tanh(object):
    @staticmethod
    def __str__():
        return "hyperbolic tangent"

    @staticmethod
    def calc(v):
        return np.tanh(v)

    @staticmethod
    def calc_d(v):
        return 1 - np.tanh(v) ** 2


class Logistic(object):
    @staticmethod
    def __str__():
        return "logistic"

    def calc(self, v):
        return 1 / (1 + np.exp(-v))

    def calc_d(self, v):
        return self.calc(v) * (1 - self.calc(v))


class ReLU(object):
    @staticmethod
    def __str__():
        return "ReLU"

    @staticmethod
    def calc(v):
        return np.maximum(0, v)

    @staticmethod
    def calc_d(v):
        derivative = 0
        if v > 0:
            derivative = 1
        return derivative
