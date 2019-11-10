import numpy as np
import cottonwood.data.data_loader_nordic_runes as dat
from cottonwood.core.activation import Tanh
from cottonwood.core.model import ANN
from cottonwood.core.error_function import Sqr
# from cottonwood.core.initializers import Glorot
# from cottonwood.core.initializers import He
from cottonwood.core.layers.dense import Dense
from cottonwood.core.layers.range_normalization import RangeNormalization
from cottonwood.core.layers.difference import Difference
from cottonwood.core.optimizers import Momentum
from cottonwood.core.regularization import L1, Limit
from cottonwood.examples.autoencoder.autoencoder_viz import Printer
from cottonwood.experimental.initializers import Uniform


def run():
    training_set, evaluation_set = dat.get_data_sets()

    sample = next(training_set)
    n_pixels = np.prod(sample.shape)
    printer = Printer(input_shape=sample.shape)

    N_NODES = [24]
    n_nodes = N_NODES + [n_pixels]
    layers = []

    layers.append(RangeNormalization(training_set))

    for i_layer in range(len(n_nodes)):
        new_layer = Dense(
            n_nodes[i_layer],
            activation_function=Tanh,
            # initializer=Glorot(),
            # initializer=He(),
            initializer=Uniform(scale=3),
            previous_layer=layers[-1],
            optimizer=Momentum(),
        )
        new_layer.add_regularizer(L1())
        new_layer.add_regularizer(Limit(4.0))
        layers.append(new_layer)

    layers.append(Difference(layers[-1], layers[0]))

    autoencoder = ANN(
        layers=layers,
        error_function=Sqr,
        printer=printer,
    )

    msg = """

Running autoencoder demo
    on Nordic Runes data set.
    Find performance history plots,
    model parameter report,
    and neural network visualizations
    in the {} directory.

""".format(autoencoder.reports_path)

    print(msg)

    autoencoder.train(training_set)
    autoencoder.evaluate(evaluation_set)
