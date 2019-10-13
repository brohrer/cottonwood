import data_loader_nordic_runes as dat
import nn_framework.activation as activation
import nn_framework.framework as framework
import nn_framework.error_fun as error_fun
from nn_framework.layer import Dense
from nn_framework.regularization import L1, L2, Limit
from autoencoder_viz import Printer

N_NODES = [24]

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
input_value_range = (0, 1)
n_pixels = sample.shape[0] * sample.shape[1]
printer = Printer(input_shape=sample.shape)

n_nodes = [n_pixels] + N_NODES + [n_pixels]
dropout_rates = [.2, .5]
model = []
for i_layer in range(len(n_nodes) - 1):
    new_layer = Dense(
        n_nodes[i_layer],
        n_nodes[i_layer + 1],
        activation.tanh,
        dropout_rate=dropout_rates[i_layer],
    )
    # new_layer.add_regularizer(L1())
    # new_layer.add_regularizer(L2())
    new_layer.add_regularizer(Limit(4.0))
    model.append(new_layer)

autoencoder = framework.ANN(
    model=model,
    error_fun=error_fun.abs,
    printer=printer,
    expected_range=input_value_range,
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
