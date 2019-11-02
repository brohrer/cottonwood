import numpy as np
from cottonwood.core.activation import Tanh
from cottonwood.core.initializers import Glorot
from cottonwood.core.layers.generic_layer import GenericLayer
from cottonwood.core.optimizers import SGD
import cottonwood.core.toolbox as tb


class Dense(GenericLayer):
    def __init__(
        self,
        n_outputs,
        activation_function=None,
        dropout_rate=0,
        initializer=None,
        previous_layer=None,
        optimizer=None,
    ):
        self.previous_layer = previous_layer
        self.m_inputs = self.previous_layer.y.size
        self.n_outputs = int(n_outputs)
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        if activation_function is None:
            self.activation_function = Tanh()
        else:
            self.activation_function = activation_function

        if initializer is None:
            self.initializer = Glorot()
        else:
            self.initializer = initializer

        if optimizer is None:
            self.optimizer = SGD()
        else:
            self.optimizer = optimizer

        # Choose random weights.
        # Inputs match to rows. Outputs match to columns.
        # Add one to m_inputs to account for the bias term.
        self.weights = self.initializer.initialize(
            self.m_inputs + 1, self.n_outputs)

        self.reset()
        self.regularizers = []

    def __str__(self):
        """
        Make a descriptive, human-readable string for this layer.
        """
        str_parts = [
            "fully connected",
            f"number of inputs: {self.m_inputs}",
            f"number of outputs: {self.n_outputs}",
            "activation function:" + tb.indent(
                self.activation_function.__str__()),
            "initialization:" + tb.indent(self.initializer.__str__()),
            "optimizer:" + tb.indent(self.optimizer.__str__()),
        ]
        for regularizer in self.regularizers:
            str_parts.append(
                "regularizer:" + tb.indent(regularizer.__str__()))
        return "\n".join(str_parts)

    def add_regularizer(self, new_regularizer):
        self.regularizers.append(new_regularizer)

    def reset(self):
        self.x = np.zeros((1, self.m_inputs))
        self.y = np.zeros((1, self.n_outputs))
        self.de_dx = np.zeros((1, self.m_inputs))
        self.de_dy = np.zeros((1, self.n_outputs))

    def forward_pass(self, evaluating=False, **kwargs):
        """
        Propagate the inputs forward through the network.

        evaluating: boolean
            Is this part of a training run or an evaluation run?
        """
        if self.previous_layer is not None:
            self.x += self.previous_layer.y
        # Apply dropout only during training runs.
        if evaluating:
            dropout_rate = 0
        else:
            dropout_rate = self.dropout_rate

        self.i_dropout = np.zeros(self.x.size, dtype=bool)
        self.i_dropout[np.where(
            np.random.uniform(size=self.x.size) < dropout_rate)] = True
        self.x[:, self.i_dropout] = 0
        self.x[:, np.logical_not(self.i_dropout)] *= 1 / (1 - dropout_rate)

        bias = np.ones((1, 1))
        x_w_bias = np.concatenate((self.x, bias), axis=1)
        v = x_w_bias @ self.weights
        self.y = self.activation_function.calc(v)

    def backward_pass(self):
        """
        Propagate the outputs back through the layer.
        """
        bias = np.ones((1, 1))
        x_w_bias = np.concatenate((self.x, bias), axis=1)

        dy_dv = self.activation_function.calc_d(self.y)
        # v = self.x @ self.weights
        dv_dw = x_w_bias.transpose()
        dv_dx = self.weights.transpose()

        dy_dw = dv_dw @ dy_dv
        self.de_dw = self.de_dy * dy_dw

        for regularizer in self.regularizers:
            regularizer.pre_optim_update(self)

        self.optimizer.update(self)

        for regularizer in self.regularizers:
            regularizer.post_optim_update(self)

        self.de_dx = (self.de_dy * dy_dv) @ dv_dx
        # Remove the dropped-out inputs from this run.
        de_dx_no_bias = self.de_dx[:, :-1]
        de_dx_no_bias[:, self.i_dropout] = 0

        # Remove the bias node from the gradient vector.
        self.previous_layer.de_dy += de_dx_no_bias
