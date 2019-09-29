import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")


class ANN(object):
    def __init__(
        self,
        model=None,
        error_fun=None,
        printer=None,
        expected_range=(-1, 1),
    ):
        self.layers = model
        self.error_fun = error_fun
        self.error_history = []
        self.n_iter_train = int(1e8)
        self.n_iter_evaluate = int(1e6)
        self.viz_interval = int(1e5)
        self.reporting_bin_size = int(1e3)
        self.report_min = -3
        self.report_max = 0
        self.printer = printer
        self.expected_range = expected_range

        self.reports_path = "reports"
        self.report_name = "performance_history.png"
        # Ensure that subdirectories exist.
        try:
            os.mkdir("reports")
        except Exception:
            pass

    def train(self, training_set):
        for i_iter in range(self.n_iter_train):
            x = self.normalize(next(training_set()).ravel())
            y = self.forward_prop(x)
            error = self.error_fun.calc(x, y)
            error_d = self.error_fun.calc_d(x, y)
            self.error_history.append((np.mean(error**2))**.5)
            self.back_prop(error_d)

            if (i_iter + 1) % self.viz_interval == 0:
                self.report()
                self.printer.render(self, x, f"train_{i_iter + 1:08d}")

    def evaluate(self, evaluation_set):
        for i_iter in range(self.n_iter_evaluate):
            x = self.normalize(next(evaluation_set()).ravel())
            y = self.forward_prop(x)
            error = self.error_fun.calc(x, y)
            self.error_history.append((np.mean(error**2))**.5)

            if (i_iter + 1) % self.viz_interval == 0:
                self.report()
                self.printer.render(self, x, f"eval_{i_iter + 1:08d}")

    def forward_prop(self, x):
        # Convert the inputs into a 2D array of the right shape.
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers:
            y = layer.forward_prop(y)
        return y.ravel()

    def back_prop(self, de_dy):
        for i_layer, layer in enumerate(self.layers[::-1]):
            de_dx = layer.back_prop(de_dy)
            de_dy = de_dx

    def forward_prop_to_layer(self, x, i_layer):
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers[:i_layer]:
            y = layer.forward_prop(y)
        return y.ravel()

    def forward_prop_from_layer(self, x, i_layer):
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers[i_layer:]:
            y = layer.forward_prop(y)
        return y.ravel()

    def normalize(self, values):
        """
        Transform the input/output values so that they tend to
        fall between -.5 and .5
        """
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = max_val - min_val
        offset_factor = min_val
        return (values - offset_factor) / scale_factor - .5

    def denormalize(self, transformed_values):
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = 2 / (max_val - min_val)
        offset_factor = min_val - 1
        return transformed_values / scale_factor - offset_factor

    def report(self):
        n_bins = int(len(self.error_history) // self.reporting_bin_size)
        smoothed_history = []
        for i_bin in range(n_bins):
            smoothed_history.append(np.mean(self.error_history[
                i_bin * self.reporting_bin_size:
                (i_bin + 1) * self.reporting_bin_size
            ]))
        error_history = np.log10(np.array(smoothed_history) + 1e-10)
        ymin = np.minimum(self.report_min, np.min(error_history))
        ymax = np.maximum(self.report_max, np.max(error_history))
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(error_history)
        ax.set_xlabel(f"x{self.reporting_bin_size} iterations")
        ax.set_ylabel("log error")
        ax.set_ylim(ymin, ymax)
        ax.grid()
        fig.savefig(os.path.join(self.reports_path, self.report_name))
        plt.close()
