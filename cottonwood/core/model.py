import datetime as dt
import os
import numpy as np
import matplotlib.pyplot as plt
from cottonwood.core.error_function import Sqr
import cottonwood.core.toolbox as tb
plt.switch_backend("agg")


class ANN(object):
    def __init__(
        self,
        layers=None,
        error_function=None,
        printer=None,
    ):
        if error_function is None:
            self.error_function = Sqr()
        else:
            self.error_function = error_function

        self.layers = layers
        self.error_history = []
        self.i_iter = 0
        self.n_iter_train = int(8e5)
        self.n_iter_evaluate = int(2e5)
        self.viz_interval = int(1e5)
        self.report_interval = int(1e4)
        self.reporting_bin_size = int(1e3)
        self.report_min = -3
        self.report_max = 0
        self.printer = printer

        time_dir = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.reports_path = os.path.join("reports", time_dir)
        self.performance_report_name = "performance_history.png"
        self.parameter_report_name = "model_parameters.txt"

        # Ensure that subdirectories exist.
        try:
            os.mkdir("reports")
        except Exception:
            pass
        try:
            os.mkdir(self.reports_path)
        except Exception:
            pass

        self.report_parameters()

    def __str__(self):
        str_parts = [
            "artificial neural network",
            "number of training iterations: " + str(self.n_iter_train),
            "number of evaluation iterations: " + str(self.n_iter_evaluate),
            "error_function:" + tb.indent(self.error_function.__str__())
        ]
        for i_layer, layer in enumerate(self.layers):
            str_parts.append(
                f"layer {i_layer}:" + tb.indent(layer.__str__())
            )
        return "\n".join(str_parts)

    def train(self, training_set):
        for _ in range(self.n_iter_train):
            self.i_iter += 1
            x = next(training_set).ravel()
            y = self.forward_pass(x)
            error = self.error_function.calc(y)
            error_d = self.error_function.calc_d(y)
            self.error_history.append(error)
            self.backward_pass(error_d)

            if (self.i_iter) % self.report_interval == 0:
                self.report_performance()

            if (self.i_iter) % self.viz_interval == 0:
                if self.printer is not None:
                    self.printer.render(
                        self,
                        x,
                        self.reports_path,
                        f"train_{self.i_iter:08d}")

    def evaluate(self, evaluation_set):
        for _ in range(self.n_iter_evaluate):
            self.i_iter += 1
            x = next(evaluation_set).ravel()
            y = self.forward_pass(x, evaluating=True)
            error = self.error_function.calc(y)
            self.error_history.append(error)

            if (self.i_iter) % self.report_interval == 0:
                self.report_performance()

            if (self.i_iter) % self.viz_interval == 0:
                if self.printer is not None:
                    self.printer.render(
                        self,
                        x,
                        self.reports_path,
                        f"eval_{self.i_iter:08d}")

    def forward_pass(
        self,
        x,
        evaluating=False,
        i_start_layer=None,
        i_stop_layer=None,
    ):
        """
        evaluating: boolean
            Tells whether this is an evaluation
            (or testing, or validation) run. Some layers behave
            a bit differently during evaluation
        i_start_layer, i_stop_layer: int
            Which layers to include in this forward pass?
            Uses the python indexing convention - layer[i_stop_layer]
            is *not* included in the pass.
            For some purposes, like visualization, it's helpful to
            inject activities into a layer, or pull them out from
            a middle layer.
        """
        if i_start_layer is None:
            i_start_layer = 0
        if i_stop_layer is None:
            i_stop_layer = len(self.layers)
        # Check for the case in which no layers are included in the range.
        if i_start_layer >= i_stop_layer:
            return x

        # Reset all the layers to get them ready for the new iteration.
        for layer in self.layers:
            layer.reset()

        # Convert the inputs into a 2D array of the right shape
        # and increment the inputs of the start layer.
        self.layers[i_start_layer].x += x.ravel()[np.newaxis, :]

        for layer in self.layers[i_start_layer: i_stop_layer]:
            layer.forward_pass(evaluating=evaluating)

        return layer.y.ravel()

    def backward_pass(self, de_dy):
        self.layers[-1].de_dy += de_dy
        for layer in self.layers[::-1]:
            layer.backward_pass()

    def report_parameters(self):
        """
        Create a human-readable summary of the model's parameters.
        """
        param_info = "type: " + self.__str__()
        with open(
            os.path.join(self.reports_path, self.parameter_report_name), "w"
        ) as param_file:
            param_file.write(param_info)

    def report_performance(self):
        """
        Create a plot of the error history.
        """
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
        fig.savefig(os.path.join(
            self.reports_path, self.performance_report_name))
        plt.close()
