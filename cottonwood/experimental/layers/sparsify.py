import numpy as np
from cottonwood.core.layers.generic_layer import GenericLayer


class Sparsify(GenericLayer):
    """
    Ensure that only a few nodes have a nonzero output.
    """
    def __init__(
        self,
        n_nodes,
        n_active_nodes=None,
        # n_iter_train=None,
        previous_layer=None,
        # settling_time=.25,
    ):
        self.previous_layer = previous_layer
        self.m_inputs = n_nodes
        self.n_outputs = n_nodes
        if n_active_nodes is None:
            self.n_active = int(n_nodes / 2)
        else:
            self.n_active = int(n_active_nodes)
        self.counts = np.zeros(self.m_inputs)

        # n_shrink is the number of training iterations over which
        # nodes are dropped.
        # settling_time is the fraction of the training period over
        # which no shrinkage occurs.
        # self.n_shrink = n_iter_train * (1 - settling_time)
        self.i_iter = 0

        self.weights = np.zeros((self.m_inputs, self.n_outputs))
        self.s_min = 1
        self.s_max = 10
        self.s_time_const = 100
        self.sensitivity = self.s_min * np.ones(self.m_inputs)
        self.reset()

    def __str__(self):
        str_parts = [
            "sparsify",
            f"{self.n_outputs} total nodes",
            f"{self.n_active} active nodes",
        ]
        return "\n".join(str_parts)

    def reset(self):
        self.x = np.zeros((1, self.m_inputs))
        self.y = np.zeros((1, self.n_outputs))
        self.de_dx = np.zeros((1, self.m_inputs))
        self.de_dy = np.zeros((1, self.n_outputs))
        # Reset the active connections
        self.weights[np.diag_indices(self.m_inputs)] = 0

    def find_active_nodes_deterministic(self):
        i_sort = np.argsort(np.abs(self.x.ravel()))
        # i_active = i_sort[-self.get_n_active():]
        i_active = i_sort[-self.n_active:]
        return i_active

    '''
    def find_active_nodes_stochastic(self):
        # choice_weight = np.abs(self.x.ravel()) ** 2
        choice_weight = np.abs(self.x.ravel()) ** self.get_match_exponent()
        choice_p = choice_weight / np.sum(choice_weight)
        i_active = np.random.choice(
            np.arange(self.m_inputs, dtype=np.int),
            size=self.get_n_active(),
            # size=self.n_active,
            replace=False,
            p=choice_p,
        )
        return i_active

    def get_n_active(self):
        time_const = self.n_shrink / np.log2(self.m_inputs)
        n_active = self.m_inputs * 2 ** (-self.i_iter / time_const)
        n_active = np.minimum(self.m_inputs, n_active)
        n_active = np.maximum(self.n_active, n_active)

        n_active = self.n_active
        return int(n_active)

    def get_match_exponent(self):
        initial_exponent = 2
        final_exponent = 10
        exponent = initial_exponent + (
            final_exponent - initial_exponent) * self.i_iter / self.n_shrink
         return exponent
    '''

    def forward_pass(self, evaluating=False, **kwargs):
        self.i_iter += 1
        if self.previous_layer is not None:
            self.x += self.previous_layer.y
        if evaluating:
            self.i_active = self.find_active_nodes_deterministic()
        else:
            self.i_active = self.find_active_nodes_deterministic()
            # self.i_active = self.find_active_nodes_stochastic()
        self.counts[self.i_active] += 1

        self.weights[self.i_active, self.i_active] = 1
        self.y = np.zeros((1, self.n_outputs))
        self.y[:, self.i_active] = self.x[:, self.i_active]
        self.sensitivity += (self.s_max - self.sensitivity) / self.s_time_const
        self.sensitivity[self.i_active] = self.s_min

    def backward_pass(self):
        self.de_dx = np.zeros((1, self.m_inputs))
        self.de_dx[:, self.i_active] = self.de_dy[:, self.i_active]
        self.de_dx *= self.sensitivity
        if self.previous_layer is not None:
            self.previous_layer.de_dy += self.de_dx
