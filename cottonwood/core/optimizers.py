import numpy as np


class GenericOptimizer(object):
    def __init__(self, **kwargs):
        default_adam_beta_1 = .9
        default_adam_beta_2 = .999
        default_epsilon = 1e-8
        default_learning_rate = 1e-3
        default_minibatch_size = 1
        default_momentum_amount = .9
        default_scaling_factor = 1e-3

        self.adam_beta_1 = kwargs.get(
            "adam_beta_1", default_adam_beta_1)
        self.adam_beta_2 = kwargs.get(
            "adam_beta_2", default_adam_beta_2)
        self.epsilon = kwargs.get(
            "epsilon", default_epsilon)
        self.learning_rate = kwargs.get(
            "learning_rate ", default_learning_rate)
        self.minibatch_size = kwargs.get(
            "minibatch_size ", default_minibatch_size)
        self.momentum_amount = kwargs.get(
            "momentum_amount", default_momentum_amount)
        self.scaling_factor = kwargs.get(
            "scaling_factor", default_scaling_factor)

        self.i_minibatch = 0
        self.de_dw_total = None

    def update_minibatch(self, layer):
        if self.de_dw_total is None:
            self.de_dw_total = np.zeros(layer.de_dw.shape)
        self.de_dw_total += layer.de_dw
        self.i_minibatch += 1

        de_dw_batch = None
        if self.i_minibatch >= self.minibatch_size:
            de_dw_batch = self.de_dw_total / self.minibatch_size
            self.de_dw_total = None
            self.i_minibatch = 0

        return de_dw_batch


class SGD(GenericOptimizer):
    """
    Uses minibatch_size, learning_rate parameters.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        str_parts = [
            "stochastic gradient descent",
            f"learning_rate: {self.learning_rate}",
            f"minibatch size: {self.minibatch_size}",
        ]
        return "\n".join(str_parts)

    def update(self, layer):
        de_dw_batch = self.update_minibatch(layer)
        if de_dw_batch is None:
            return
        layer.weights -= de_dw_batch * self.learning_rate


class Momentum(GenericOptimizer):
    """
    Uses minibatch_size, learning_rate, momentum_amount parameters.

    Rumelhart, David E.; Hinton, Geoffrey E.; Williams, Ronald J.
    (8 October 1986). "Learning representations by back-propagating errors".
    Nature. 323 (6088): 533–536.
    Bibcode:1986Natur.323..533R. doi:10.1038/323533a0.
    http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_adjustment = None

    def __str__(self):
        str_parts = [
            "momentum",
            f"learning_rate: {self.learning_rate}",
            f"momentum amount: {self.momentum_amount}",
            f"minibatch size: {self.minibatch_size}",
        ]
        return "\n".join(str_parts)

    def update(self, layer):
        de_dw_batch = self.update_minibatch(layer)
        if de_dw_batch is None:
            return

        if self.previous_adjustment is None:
            self.previous_adjustment = np.zeros(layer.weights.shape)
        new_adjustment = (
            self.previous_adjustment * self.momentum_amount
            + de_dw_batch * self.learning_rate
        )
        layer.weights -= new_adjustment

        # Update previous_adjustment to get set up for the next iteration.
        self.previous_adjustment = new_adjustment


class Adam(GenericOptimizer):
    """
    Uses parameters minbatch_size, learning_rate, beta_1, beta_2, epsilon

    Adam: A Method for Stochastic Optimization
    Diederik P. Kingma, Jimmy Lei Ba
    https://arxiv.org/abs/1412.6980
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_moment = 0
        self.second_moment = 0
        self.timestep = 0

    def __str__(self):
        str_parts = [
            "adam",
            f"learning_rate: {self.learning_rate}",
            f"beta 1: {self.adam_beta_1}",
            f"beta 2: {self.adam_beta_2}",
            f"minibatch size: {self.minibatch_size}",
        ]
        return "\n".join(str_parts)

    def update(self, layer):
        self.timestep += 1

        de_dw_batch = self.update_minibatch(layer)
        if de_dw_batch is None:
            return

        self.first_moment = (
            self.adam_beta_1 * self.first_moment
            + (1 - self.adam_beta_1) * de_dw_batch
        )
        self.second_moment = (
            self.adam_beta_2 * self.second_moment
            + (1 - self.adam_beta_2) * de_dw_batch ** 2
        )
        corrected_first_moment = self.first_moment / (
            1 - self.adam_beta_1 ** self.timestep)
        corrected_second_moment = self.second_moment / (
            1 - self.adam_beta_2 ** self.timestep)

        adjustment = self.learning_rate * corrected_first_moment / (
            corrected_second_moment ** .5 + self.epsilon)
        layer.weights -= adjustment
