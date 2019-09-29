import data_loader_nordic_runes as dat
import nn_framework.activation as activation
import nn_framework.framework as framework
import nn_framework.error_fun as error_fun
import nn_framework.layer as layer
from autoencoder_viz import Printer

N_NODES = [24]

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
input_value_range = (0, 1)
n_pixels = sample.shape[0] * sample.shape[1]
printer = Printer(input_shape=sample.shape)

n_nodes = [n_pixels] + N_NODES + [n_pixels]
model = []
for i_layer in range(len(n_nodes) - 1):
    model.append(layer.Dense(
        n_nodes[i_layer],
        n_nodes[i_layer + 1],
        activation.tanh
    ))

autoencoder = framework.ANN(
    model=model,
    error_fun=error_fun.abs,
    printer=printer,
    expected_range=input_value_range,
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
