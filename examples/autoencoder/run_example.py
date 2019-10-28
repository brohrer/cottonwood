import numpy as np
import data.data_loader_nordic_runes as dat
import core.activation as activation
from core.model import ANN
import core.error_fun as error_fun
from core.initializers import Glorot, He
from core.layers.dense import Dense
from core.layers.range_normalization import RangeNormalization
from core.layers.difference import Difference
from core.optimizers import SGD, Momentum, Adam
from core.regularization import L1, Limit
from examples.autoencoder.autoencoder_viz import Printer
from experimental.optimizers import NoisyMomentum
from experimental.initializers import Uniform

print("")
print("Running autoencoder demo on Nordic Runes data set.")
print("    Find performance history plots in the 'reports' directory")
print("    and neural network visualizations in the 'nn_images' directory.")
print("")

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
n_pixels = np.prod(sample.shape)
printer = Printer(input_shape=sample.shape)

N_NODES = [24]
n_nodes = N_NODES + [n_pixels]
layers = []

layers.append(RangeNormalization(training_set))

for i_layer in range(len(n_nodes)):
    new_layer = Dense(
        n_nodes[i_layer],
        activation.tanh,
        initializer=Glorot(),
        # initializer=He(),
        # initializer=Uniform(scale=.3),
        previous_layer=layers[-1],
        optimizer=Momentum(),
    )
    new_layer.add_regularizer(L1())
    # new_layer.add_regularizer(L2())
    new_layer.add_regularizer(Limit(4.0))
    layers.append(new_layer)

layers.append(Difference(layers[-1], layers[0]))

autoencoder = ANN(
    layers=layers,
    error_fun=error_fun.sqr,
    printer=printer,
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
