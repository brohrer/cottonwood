import numpy as np
from core.layers.generic_layer import GenericLayer


class BatchNormalization(GenericLayer):
    """
    Transform the input values so that they tend to
    have a mean of 0 and a variance of 1.

    Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift
    Sergey Ioffe, Christian Szegedy
    https://arxiv.org/abs/1502.03167
    """
    def __init__(self, training_data, previous_layer=None):
        self.previous_layer = previous_layer
        # Estimate the range based on a selection of training data
        n_range_test = 100
        self.range_min = 1e10
        self.range_max = -1e10
        for _ in range(n_range_test):
            sample = next(training_data())
            if self.range_min > np.min(sample):
                self.range_min = np.min(sample)
            if self.range_max < np.max(sample):
                self.range_max = np.max(sample)
        self.scale_factor = self.range_max - self.range_min
        self.offset_factor = self.range_min
        self.size = sample.size
        self.reset()

    def forward_pass(self, **kwargs):
        if self.previous_layer is not None:
            self.x += self.previous_layer.y
        self.y = (self.x - self.offset_factor) / self.scale_factor - .5

    def backward_pass(self):
        self.de_dx = self.de_dy / self.scale_factor
        if self.previous_layer is not None:
            self.previous_layer.de_dy += self.de_dx

    def denormalize(self, transformed_values):
        """
        In case you ever need to reverse the normalization process.
        """
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = 2 / (max_val - min_val)
        offset_factor = min_val - 1
        return transformed_values / scale_factor - offset_factor
