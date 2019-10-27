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
    def calc(self, v):
        return 1 / (1 + np.exp(-v))

    def calc_d(self, v):
        return self.calc(v) * (1 - self.calc(v))


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
