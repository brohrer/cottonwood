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

    def report(self):
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
        fig.savefig(os.path.join(self.reports_path, self.report_name))
        plt.close()
