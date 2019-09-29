import numpy as np

# All of these expect two identically sized numpy arrays as inputs
# and return the same size error output.


class abs(object):
    @staticmethod
    def calc(x, y):
        return np.abs(y - x)

    @staticmethod
    def calc_d(x, y):
        return np.sign(y - x)


class sqr(object):
    @staticmethod
    def calc(x, y):
        return (y - x)**2

    @staticmethod
    def calc_d(x, y):
        return 2 * (y - x)
