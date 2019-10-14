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
    ):
        self.layers = model
        self.error_fun = error_fun
        self.error_history = []
        self.n_iter_train = int(8e5)
        self.n_iter_evaluate = int(2e5)
        self.viz_interval = int(1e5)
        self.reporting_bin_size = int(1e3)
        self.report_min = -3
        self.report_max = 0
        self.printer = printer

        self.reports_path = "reports"
        self.report_name = "performance_history.png"
        # Ensure that subdirectories exist.
        try:
            os.mkdir("reports")
        except Exception:
            pass

    def train(self, training_set):
        for i_iter in range(self.n_iter_train):
            x = next(training_set()).ravel()
            y = self.forward_pass(x)
            error = self.error_fun.calc(y)
            error_d = self.error_fun.calc_d(y)
            self.error_history.append(error)
            self.backward_pass(error_d)

            if (i_iter + 1) % self.viz_interval == 0:
                self.report()
                self.printer.render(self, x, f"train_{i_iter + 1:08d}")

    def evaluate(self, evaluation_set):
        for i_iter in range(self.n_iter_evaluate):
            x = next(evaluation_set()).ravel()
            y = self.forward_pass(x, evaluating=True)
            error = self.error_fun.calc(y)
            self.error_history.append(error)

            if (i_iter + 1) % self.viz_interval == 0:
                self.report()
                self.printer.render(self, x, f"eval_{i_iter + 1:08d}")

    def forward_pass(
        self,
        x,
        evaluating=False,
        i_start_layer=None,
        i_stop_layer=None,
    ):
        if i_start_layer is None:
            i_start_layer = 0
        if i_stop_layer is None:
            i_stop_layer = len(self.layers)

        for layer in self.layers:
            layer.reset()

        # Convert the inputs into a 2D array of the right shape.
        self.layers[i_start_layer].x += x.ravel()[np.newaxis, :]

        # for i_layer, layer in enumerate(self.layers[i_start_layer: i_stop_layer]):
        for i_layer, layer in enumerate(self.layers):
            layer.forward_pass(evaluating=evaluating)
            print(i_layer, layer.y[:8])

        return layer.y.ravel()

    def backward_pass(self, de_dy):
        self.layers[-1].de_dy += de_dy
        for layer in self.layers[::-1]:
            layer.backward_pass()

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
