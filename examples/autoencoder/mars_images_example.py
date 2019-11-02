import numpy as np
from core.activation import Tanh
from core.model import ANN
from core.error_function import Sqr
from core.initializers import Glorot
from core.layers.dense import Dense
from core.layers.range_normalization import RangeNormalization
from core.layers.difference import Difference
from core.optimizers import Momentum
from core.regularization import Limit
import data.data_loader_martian_images as dat
from examples.autoencoder.autoencoder_viz import Printer


def run():
    msg = """

Running autoencoder demo
    on Martian images data set.
    Find performance history plots
    in the 'reports' directory
    and neural network visualizations
    in the 'nn_images' directory.

"""
    print(msg)

    training_set, tuning_set, evaluation_set = dat.get_data_sets()

    sample = next(training_set)
    n_pixels = np.prod(sample.shape)
    printer = Printer(input_shape=sample.shape)

    # N_NODES = [64, 36, 24, 36, 64]
    N_NODES = [64]
    n_nodes = N_NODES + [n_pixels]
    layers = []

    layers.append(RangeNormalization(training_set))

    for i_layer in range(len(n_nodes)):
        new_layer = Dense(
            n_nodes[i_layer],
            activation_function=Tanh,
            initializer=Glorot(),
            previous_layer=layers[-1],
            optimizer=Momentum(),
        )
        # new_layer.add_regularizer(L1())
        new_layer.add_regularizer(Limit(4.0))
        layers.append(new_layer)

    layers.append(Difference(layers[-1], layers[0]))

    autoencoder = ANN(
        layers=layers,
        error_function=Sqr,
        printer=printer,
    )
    autoencoder.train(training_set)
    autoencoder.evaluate(tuning_set)
    autoencoder.evaluate(evaluation_set)
