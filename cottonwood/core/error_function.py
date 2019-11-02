import numpy as np

# All of these expect two identically sized numpy arrays as inputs
# and return the same size error output.


class Abs(object):
    @staticmethod
    def __str__():
        return "mean absolute error"

    @staticmethod
    def calc(x):
        return np.mean(np.abs(x))

    @staticmethod
    def calc_d(x):
        return np.sign(x)


class Sqr(object):
    @staticmethod
    def __str__():
        return "mean squared error"

    @staticmethod
    def calc(x):
        return np.mean(x**2)

    @staticmethod
    def calc_d(x):
        return 2 * x
