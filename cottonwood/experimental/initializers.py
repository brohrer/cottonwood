import numpy as np


class Uniform(object):
    """
    Make only a fraction of weights nonzero.
    """
    def __init__(self, scale=.2):
        self.scale = scale

    def initialize(self, n_rows, n_cols):
        return np.random.uniform(
            low=-self.scale,
            high=self.scale,
            size=(n_rows, n_cols),
        )
