import numpy as np
import data_loader_nordic_runes as dat
import nn_framework.activation as activation
import nn_framework.framework as framework
import nn_framework.error_fun as error_fun
import nn_framework.layer as layer
from nn_framework.regularization import L1, L2, Limit
from autoencoder_viz import Printer

N_NODES = [24]

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
# input_value_range = (0, 1)
n_pixels = np.prod(sample.shape)
printer = Printer(input_shape=sample.shape)

n_nodes = N_NODES + [n_pixels]
dropout_rates = [.2, .5]
model = []

model.append(layer.RangeNormalization(training_set))
for i_layer in range(len(n_nodes)):
    new_layer = layer.Dense(
        n_nodes[i_layer],
        activation.tanh,
        previous_layer=model[-1],
        dropout_rate=dropout_rates[i_layer],
    )
    # new_layer.add_regularizer(L1())
    # new_layer.add_regularizer(L2())
    new_layer.add_regularizer(Limit(4.0))
    model.append(new_layer)
model.append(layer.Difference(model[-1], model[0]))

autoencoder = framework.ANN(
    model=model,
    error_fun=error_fun.sqr,
    printer=printer,
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
