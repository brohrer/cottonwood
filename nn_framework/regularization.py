import numpy as np


class L1(object):
    def __init__(self, regularization_amount=1e-2):
        self.regularization_amount = regularization_amount

    def update(self, layer):
        values = layer.weights
        delta = self.regularization_amount * layer.learning_rate
        values[np.where(values > 0)] -= delta
        values[np.where(values < 0)] += delta
        values[np.where(np.abs(values) < delta)] = 0
        return values


class L2(object):
    def __init__(self, regularization_amount=1e-2):
        self.regularization_amount = regularization_amount

    def update(self, layer):
        adjustment = (
            2 * self.regularization_amount
            * layer.learning_rate
            * layer.weights
        )
        return layer.weights - adjustment


class Limit(object):
    def __init__(self, weight_limit=1):
        self.weight_limit = weight_limit

    def update(self, layer):
        values = layer.weights
        values[np.where(values > self.weight_limit)] = self.weight_limit
        values[np.where(values < -self.weight_limit)] = -self.weight_limit
        return values
