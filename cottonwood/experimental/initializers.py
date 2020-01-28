import numpy as np


class Uniform(object):
    """
    Make only a fraction of weights nonzero.
    """
    def __init__(self, scale=.2):
        self.scale = scale

    def __str__(self):
        return f"Uniform distribution on [{-self.scale}, {self.scale}]"

    def initialize(self, n_rows, n_cols):
        return np.random.uniform(
            low=-self.scale,
            high=self.scale,
            size=(n_rows, n_cols),
        )


class Skinny(object):
    """
    Similar to He, but instead of a uniform distribution, it's weighted
    more heavily toward both ends.
    """
    @staticmethod
    def __str__():
        return "Skinny"

    @staticmethod
    def initialize(n_rows, n_cols):
        limit = 6 / n_rows
        weights = np.random.uniform(
            low=-limit,
            high=limit,
            size=(n_rows, n_cols),
        )
        weights = np.sign(weights) * np.sqrt(np.abs(weights))
        return weights


class Trimodal(object):
    """
    Weights approximate a sparse network where some of the connections are +1,
    some are -1, and most are 0.
    """
    @staticmethod
    def __str__():
        return "Trimodal"

    @staticmethod
    def initialize(n_rows, n_cols):
        # What fraction of weights are close to a magnitude of 1
        p_large = np.minimum(4 / n_rows, 1)
        sigma = np.sqrt(2 / (n_rows + n_cols))
        size = (n_rows, n_cols)
        weights = np.random.normal(scale=sigma, size=size)
        i_large = np.where(np.random.sample(size=size) < p_large)
        n_large = i_large[0].size
        mu_sign = np.random.choice([1, -1], size=n_large)
        weights[i_large] += mu_sign

        return weights


