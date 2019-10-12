import numpy as np


class Dense(object):
    def __init__(
        self,
        m_inputs,
        n_outputs,
        activate,
    ):
        self.m_inputs = int(m_inputs)
        self.n_outputs = int(n_outputs)
        self.activate = activate

        self.learning_rate = .001
        self.dropout_rate = 0

        # Choose random weights.
        # Inputs match to rows. Outputs match to columns.
        # Add one to m_inputs to account for the bias term.
        self.weights = (np.random.sample(
            size=(self.m_inputs + 1, self.n_outputs)) * 2 - 1)
        self.x = np.zeros((1, self.m_inputs + 1))
        self.y = np.zeros((1, self.n_outputs))

        self.regularizers = []

    def add_regularizer(self, new_regularizer):
        self.regularizers.append(new_regularizer)

    def set_dropout_rate(self, new_dropout_rate):
        self.dropout_rate = new_dropout_rate

    def forward_prop(self, inputs, evaluating=False):
        """
        Propagate the inputs forward through the network.

        inputs: 2D array
            One column array of input values.
        evaluating: boolean
            Is this part of a training run or an evaluation run?
        """
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
        self.x = np.concatenate((inputs, bias), axis=1)
        v = self.x @ self.weights
        self.y = self.activate.calc(v)
        return self.y

    def back_prop(self, de_dy):
        """
        Propagate the outputs back through the layer.
        """
        dy_dv = self.activate.calc_d(self.y)
        # v = self.x @ self.weights
        dv_dw = self.x.transpose()
        dv_dx = self.weights.transpose()

        dy_dw = dv_dw @ dy_dv
        de_dw = de_dy * dy_dw

        self.weights -= de_dw * self.learning_rate
        for regularizer in self.regularizers:
            self.weights = regularizer.update(self)

        de_dx = (de_dy * dy_dv) @ dv_dx
        # Remove the dropped-out inputs from this run.
        de_dx[:, self.i_dropout] = 0
        return de_dx[:, :-1]
