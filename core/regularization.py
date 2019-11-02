import numpy as np


class GenericRegularizer:
    def __init__(self):
        pass

    def pre_optim_update(self, layer):
        pass

    def post_optim_update(self, layer):
        pass


class L1(GenericRegularizer):
    def __init__(self, regularization_amount=1e-2):
        self.regularization_amount = regularization_amount

    def __str__(self):
        str_parts = [
            "L1 (lasso)",
            f"regularization amount: {self.regularization_amount}",
        ]
        return "\n".join(str_parts)

    def pre_optim_update(self, layer):
        layer.de_dw += np.sign(layer.weights) * self.regularization_amount


class L2(GenericRegularizer):
    def __init__(self, regularization_amount=1e-2):
        self.regularization_amount = regularization_amount

    def __str__(self):
        str_parts = [
            "L2 (Ridge)",
            f"regularization amount: {self.regularization_amount}",
        ]
        return "\n".join(str_parts)

    def pre_optim_update(self, layer):
        layer.de_dw += 2 * layer.weights * self.regularization_amount


class Limit(GenericRegularizer):
    def __init__(self, weight_limit=1):
        self.weight_limit = weight_limit

    def __str__(self):
        str_parts = [
            "Limit",
            f"weight limit: {self.weight_limit}",
        ]
        return "\n".join(str_parts)

    def post_optim_update(self, layer):
        layer.weights = np.minimum(self.weight_limit, layer.weights)
        layer.weights = np.maximum(-self.weight_limit, layer.weights)
