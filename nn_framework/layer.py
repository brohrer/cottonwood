import numpy as np


class Identity(object):
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
        self.x = self.previous_layer.y
        self.y = self.x

    def backward_pass(self):
        self.de_dx = self.de_dy
        self.previous_layer.de_dy = self.de_dx


class Dense(Identity):
    def __init__(
        self,
        n_outputs,
        activation_function,
        previous_layer=None,
        dropout_rate=0,
    ):
        self.previous_layer = previous_layer
        self.m_inputs = self.previous_layer.y.size
        self.n_outputs = int(n_outputs)
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        self.learning_rate = .001

        # Choose random weights.
        # Inputs match to rows. Outputs match to columns.
        # Add one to m_inputs to account for the bias term.
        self.weights = (np.random.sample(
            size=(self.m_inputs + 1, self.n_outputs)) * 2 - 1)

        self.reset()
        self.regularizers = []

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
            self.x = self.previous_layer.y
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
        de_dw = self.de_dy * dy_dw

        self.weights -= de_dw * self.learning_rate
        for regularizer in self.regularizers:
            self.weights = regularizer.update(self)

        self.de_dx = (self.de_dy * dy_dv) @ dv_dx
        # Remove the dropped-out inputs from this run.
        de_dx_no_bias = self.de_dx[:, :-1]
        de_dx_no_bias[:, self.i_dropout] = 0

        # Remove the bias node from the gradient vector.
        self.previous_layer.de_dy += de_dx_no_bias


class RangeNormalization(Identity):
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
            self.x = self.previous_layer.y
        self.y = (self.x - self.offset_factor) / self.scale_factor - .5

    def backward_pass(self):
        de_dx = self.de_dy / self.scale_factor
        if self.previous_layer is not None:
            self.previous_layer.de_dy += de_dx


class Difference(Identity):
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

    def forward_pass(self, **kwargs):
        self.y = self.previous_layer.y - self.subtract_me_layer.y

    def backward_pass(self):
        self.previous_layer.de_dy += self.de_dy
        self.subtract_me_layer.de_dy -= self.de_dy
