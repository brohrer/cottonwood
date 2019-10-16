import numpy as np

# All of these need to be able to handle 2D numpy arrays as inputs.


class tanh(object):
    @staticmethod
    def calc(v):
        return np.tanh(v)

    @staticmethod
    def calc_d(v):
        return 1 - np.tanh(v) ** 2


class logistic(object):
    @staticmethod
    def calc(v):
        return 1 / (1 + np.exp(-v))

    @staticmethod
    def calc_d(v):
        return calc(v) * (1 - calc(v))


class relu(object):
    @staticmethod
    def calc(v):
        return np.maximum(0, v)

    @staticmethod
    def calc_d(v):
        derivative = 0
        if v > 0:
            derivative = 1
        return derivative
