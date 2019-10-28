import numpy as np


class Glorot(object):
    """
    Understanding the difficulty of training deep feedforward neural networks
    Xavier Glorot, Yoshua Bengio
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
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
    @staticmethod
    def initialize(n_rows, n_cols):
        return np.random.uniform(
            low=-np.sqrt(6 / n_rows),
            high=np.sqrt(6 / n_rows),
            size=(n_rows, n_cols),
        )
