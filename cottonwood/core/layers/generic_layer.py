import numpy as np


class GenericLayer(object):
    """
    This is a generic layer, but one without any useful function.
    When you go to write a custom layer, you can use it as a starting point.
    """
    def __init__(self, previous_layer):
        self.previous_layer = previous_layer
        self.size = self.previous_layer.y.size
        self.reset()

    def reset(self):
        self.x = np.zeros((1, self.size))
        self.y = np.zeros((1, self.size))
        self.de_dx = np.zeros((1, self.size))
        self.de_dy = np.zeros((1, self.size))

    def forward_pass(self, **kwargs):
        self.x += self.previous_layer.y
        self.y = self.x

    def backward_pass(self):
        self.de_dx = self.de_dy
        self.previous_layer.de_dy += self.de_dx
