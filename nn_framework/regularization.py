import numpy as np


class L1(object):
    def __init__(self, regularization_amount=1e-5):
        self.regularization_amount = regularization_amount

    def update(self, values):
        adjustment = self.regularization_amount * values
        return values - adjustment


class L2(object):
    def __init__(self, regularization_amount=1e-4):
        self.regularization_amount = regularization_amount

    def update(self, values):
        adjustment = (
            self.regularization_amount
            * np.sign(values) * values * values)
        return values - adjustment


class Limit(object):
    def __init__(self, weight_limit=1):
        self.weight_limit = weight_limit

    def update(self, values):
        values[np.where(values > self.weight_limit)] = self.weight_limit
        values[np.where(values < -self.weight_limit)] = -self.weight_limit
        return values
