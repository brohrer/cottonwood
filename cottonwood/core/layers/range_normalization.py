import numpy as np
from cottonwood.core.layers.generic_layer import GenericLayer


class RangeNormalization(GenericLayer):
    """
    Transform the input/output values so that they tend to
    fall between -.5 and .5
    """
    def __init__(self, training_data, previous_layer=None):
        self.previous_layer = previous_layer
        # Estimate the range based on a selection of training data
        n_range_test = 100
        self.range_min = 1e10
        self.range_max = -1e10
        for _ in range(n_range_test):
            sample = next(training_data)
            if self.range_min > np.min(sample):
                self.range_min = np.min(sample)
            if self.range_max < np.max(sample):
                self.range_max = np.max(sample)
        self.scale_factor = self.range_max - self.range_min
        self.offset_factor = self.range_min
        self.size = sample.size
        self.reset()

    def __str__(self):
        str_parts = [
            "range normalization",
            f"range maximum: {self.range_max}",
            f"range minimum: {self.range_min}",
        ]
        return "\n".join(str_parts)

    def forward_pass(self, **kwargs):
        if self.previous_layer is not None:
            self.x += self.previous_layer.y
        self.y = (self.x - self.offset_factor) / self.scale_factor - .5

    def backward_pass(self):
        self.de_dx = self.de_dy / self.scale_factor
        if self.previous_layer is not None:
            self.previous_layer.de_dy += self.de_dx

    def denormalize(self, vals):
        """
        In case you ever need to reverse the normalization process.
        """
        return self.scale_factor * (vals + .5) + self.offset_factor
