import numpy as np
import data.data_loader_nordic_runes as dat
import core.activation as activation
import core.framework as framework
import core.error_fun as error_fun
from core.layers.dense import Dense
from core.layers.range_normalization import RangeNormalization
from core.layers.difference import Difference
from core.optimizers import SGD, Momentum, Adam, NoisyMomentum
from core.regularization import L1, Limit
from examples.autoencoder.autoencoder_viz import Printer

print("")
print("Running autoencoder demo on Nordic Runes data set.")
print("  Find performance history plots in the 'reports' directory")
print("  and neural network visualizations in the 'nn_images' directory.")
print("")

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
n_pixels = np.prod(sample.shape)
printer = Printer(input_shape=sample.shape)

N_NODES = [24]
n_nodes = N_NODES + [n_pixels]
# dropout_rates = [.2, .5]
model = []

model.append(RangeNormalization(training_set))

for i_layer in range(len(n_nodes)):
    new_layer = Dense(
        n_nodes[i_layer],
        activation.tanh,
        previous_layer=model[-1],
        # dropout_rate=dropout_rates[i_layer],
        # optimizer=SGD(),
        optimizer=Momentum(),
        # optimizer=NoisyMomentum(),
        # optimizer=Adam(),
    )
    new_layer.add_regularizer(L1())
    # new_layer.add_regularizer(L2())
    new_layer.add_regularizer(Limit(4.0))
    model.append(new_layer)

model.append(Difference(model[-1], model[0]))

autoencoder = framework.ANN(
    model=model,
    error_fun=error_fun.sqr,
    printer=printer,
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
