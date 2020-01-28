import numpy as np


class Glorot(object):
    """
    Understanding the difficulty of training deep feedforward neural networks
    Xavier Glorot, Yoshua Bengio
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def __str__():
        return "Glorot"

    @staticmethod
    def initialize(n_rows, n_cols):
        return np.random.normal(
            scale=np.sqrt(2 / (n_rows + n_cols)),
            size=(n_rows, n_cols),
        )


class He(object):
    """
    Delving Deep into Rectifiers:
    Surpassing Human-Level Performance on ImageNet Classification
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    https://arxiv.org/abs/1502.01852
    """
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def __str__():
        return "He"

    @staticmethod
    def initialize(n_rows, n_cols):
        return np.random.uniform(
            low=-np.sqrt(6 / n_rows),
            high=np.sqrt(6 / n_rows),
            size=(n_rows, n_cols),
        )


class LSUV(object):
    """
    This is taken from

    All you need is a good init
    Dmytro Mishkin, Jiri Matas
    https://arxiv.org/abs/1511.06422

    which builds on

    Exact solutions to the nonlinear dynamics of learning
    in deep linear neural networks
    Andrew M. Saxe, James L. McClelland, Surya Ganguli
    https://arxiv.org/abs/1312.6120
    """
    def __init__(self, scale=1, **kwargs):
        # The scale is the expected standard deviation of the inputs.
        # It's assumed that the inputs are all distributed identically.
        self.input_stddev = scale

    @staticmethod
    def __str__():
        return "LSUV"

    def initialize(self, n_rows, n_cols):
        # Step 1: Generate a weight matrix that is orthonormal.
        # It's guaranteed to generate outputs that are independent
        # of each other. This from Saxe et al.
        u, _, v = np.linalg.svd(
            np.random.normal(size=(n_rows, n_cols)),
            full_matrices=False)
        weights = u if u.shape == (n_rows, n_cols) else v

        # Step 2: Normalize the weights so that the outputs have
        # a variance of 1.
        input_values = np.random.normal(
            scale=self.input_stddev,
            size=(1, n_rows))
        output_values = input_values @ weights
        return weights / np.std(output_values)
