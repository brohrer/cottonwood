from cottonwood.core.layers.generic_layer import GenericLayer


class Difference(GenericLayer):
    """
    A difference layer calculates the difference between the
    outputs of two earlier layers, the previous layer
    and another one from further back, the subtract_me layer.
    The output of this layer is
    y = previous_layer.y - subtract_me_layer.y
    """
    def __init__(self, previous_layer, subtract_me_layer):
        self.previous_layer = previous_layer
        self.subtract_me_layer = subtract_me_layer
        assert self.subtract_me_layer.y.size == self.previous_layer.y.size

        self.size = self.previous_layer.y.size

    def __str__(self):
        return "difference"

    def forward_pass(self, **kwargs):
        self.y = self.previous_layer.y - self.subtract_me_layer.y

    def backward_pass(self):
        self.previous_layer.de_dy += self.de_dy
        self.subtract_me_layer.de_dy -= self.de_dy
