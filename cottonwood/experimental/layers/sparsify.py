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
        previous_layer=None,
    ):
        self.previous_layer = previous_layer
        self.m_inputs = n_nodes
        self.n_outputs = n_nodes
        if n_active_nodes is None:
            self.n_active = int(n_nodes / 2)
        else:
            self.n_active = int(n_active_nodes)

        # The weights array is just for show here, but it's
        # really helpful when it comes time to visualize the while network.
        self.weights = np.zeros((self.m_inputs, self.n_outputs))

        # Sensitivity helps rarely active nodes to be more malleable.
        # It helps transform them into something more useful.
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

    def forward_pass(self, evaluating=False, **kwargs):
        if self.previous_layer is not None:
            self.x += self.previous_layer.y

        # Find which nodes are active on this pass.
        # They will be the ones with the highest activation.
        i_sort = np.argsort(np.abs(self.x.ravel()))
        i_active = i_sort[-self.n_active:]

        # Only propogate the active nodes' activities forward.
        self.weights[self.i_active, self.i_active] = 1
        self.y = np.zeros((1, self.n_outputs))
        self.y[:, self.i_active] = self.x[:, self.i_active]

        # Update the sensitivity for each node.
        # Sensitivity gradually approaches s_max, until a node is active.
        # Then it resets to s_min.
        self.sensitivity += (self.s_max - self.sensitivity) / self.s_time_const
        self.sensitivity[self.i_active] = self.s_min

    def backward_pass(self):
        # Only propogate the active nodes' gradients backward.
        self.de_dx = np.zeros((1, self.m_inputs))
        self.de_dx[:, self.i_active] = self.de_dy[:, self.i_active]

        # Ensure that adjustments to nodes that are rarely active
        # will be amplified. 
        self.de_dx *= self.sensitivity
        if self.previous_layer is not None:
            self.previous_layer.de_dy += self.de_dx
