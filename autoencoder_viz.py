"""
Generate an autoencoder neural network visualization
"""
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")


class Printer(object):
    def __init__(self, input_shape=None):

        # Choose a color palette
        self.blue = "#04253a"
        self.green = "#4c837a"
        self.tan = "#e1ddbf"
        self.cmap = "bone"
        self.error_cmap = "RdGy"
        self.im_vmax = .5
        self.im_vmin = -.5
        self.DPI = 300

        # Changing these adjusts the size and layout of the visualization
        self.figure_width = 16
        self.figure_height = 9
        self.right_border = 0.7
        self.left_border = 0.7
        self.top_border = 0.8
        self.bottom_border = 0.6

        self.n_image_rows = input_shape[0]
        self.n_image_cols = input_shape[1]

        self.input_image_bottom = 5
        self.input_image_height = 0.25 * self.figure_height
        self.error_image_scale = 0.7
        self.error_gap_scale = 0.3
        self.between_layer_scale = 0.8
        self.between_node_scale = 0.4

        self.savedir = "nn_images"
        try:
            os.mkdir(self.savedir)
        except Exception:
            pass
        try:
            for filename in os.listdir(self.savedir):
                if filename[-3:] == "png":
                    os.remove(os.path.join(self.savedir, filename))
        except Exception:
            pass

    def render(self, nn, inputs, name=""):
        """
        Build a visualization of an image autoencoder neural network,
        piece by piece.
        """
        fig, ax_boss = self.create_background()
        self.find_nn_size(nn)
        self.find_node_image_size()
        self.find_between_layer_gap()
        self.find_between_node_gap()
        self.find_error_image_position()

        image_axes = []
        self.add_input_image(fig, image_axes, nn, inputs)
        for i_layer in range(self.n_layers):
            self.add_node_images(fig, i_layer, image_axes, nn, inputs)
        self.add_output_image(fig, image_axes, nn, inputs)
        self.add_error_image(fig, image_axes, nn, inputs)
        self.add_layer_connections(ax_boss, image_axes)
        self.save_nn_viz(fig, name)
        plt.close()

    def create_background(self):
        fig = plt.figure(
            edgecolor=self.tan,
            facecolor=self.green,
            figsize=(self.figure_width, self.figure_height),
            linewidth=4,
        )
        ax_boss = fig.add_axes((0, 0, 1, 1), facecolor="none")
        ax_boss.set_xlim(0, 1)
        ax_boss.set_ylim(0, 1)
        return fig, ax_boss

    def find_nn_size(self, nn):
        """
        Find all the parameters that describe the size and location
        of the elements of the visualization.
        """
        # Enforce square pixels.
        # Each pixel will have the same height and width.
        self.aspect_ratio = self.n_image_cols / self.n_image_rows
        self.input_image_width = self.input_image_height * self.aspect_ratio

        # The network as a whole
        self.n_nodes = []
        for layer in nn.layers:
            self.n_nodes.append(layer.m_inputs)
        self.n_nodes.append(layer.n_outputs)
        self.n_layers = len(self.n_nodes)
        self.max_nodes = np.max(self.n_nodes)

    def find_node_image_size(self):
        """
        What should the height and width of each node image be?
        As big as possible, given the constraints.
        There are two possible constraints:
            1. Fill the figure top-to-bottom.
            2. Fill the figure side-to-side.
        To determine which of these limits the size of the node images,
        we'll calculate the image size assuming each constraint separately,
        then respect the one that results in the smaller node image.
        """
        # First assume height is the limiting factor.
        total_space_to_fill = (
            self.figure_height
            - self.bottom_border
            - self.top_border
        )
        # Use the layer with the largest number of nodes (n_max).
        # Pack the images and the gaps as tight as possible.
        # In that case, if the image height is h,
        # the gaps will each be h * self.between_node_scale.
        # There will be n_max nodes and (n_max - 1) gaps.
        # After a wee bit of algebra:
        height_constrained_by_height = (
            total_space_to_fill / (
               self.max_nodes
               + (self.max_nodes - 1)
               * self.between_node_scale
            )
        )

        # Second assume width is the limiting factor.
        total_space_to_fill = (
            self.figure_width
            - self.left_border
            - self.right_border
            - 2 * self.input_image_width
        )
        # Again, pack the images as tightly as possible side-to-side.
        # In this case, if the image width is w,
        # the gaps will each be w * self.between_layer_scale.
        # There will be n_layer nodes and (n_layer + 1) gaps.
        # After another tidbit of algebra:
        width_constrained_by_width = (
            total_space_to_fill / (
               self.n_layers + (self.n_layers + 1) * self.between_layer_scale
            )
        )

        # Figure out what the height would be for this width.
        height_constrained_by_width = (
            width_constrained_by_width
            / self.aspect_ratio
        )

        # See which constraint is more restrictive, and go with that one.
        self.node_image_height = np.minimum(
            height_constrained_by_width,
            height_constrained_by_height)
        self.node_image_width = self.node_image_height * self.aspect_ratio

    def find_between_layer_gap(self):
        """
        How big is the horizontal spacing between_layers?
        This is also the spacing between the input image and the first layer
        and between the last layer and the output image.
        """
        horizontal_gap_total = (
            self.figure_width
            - 2 * self.input_image_width
            - self.n_layers * self.node_image_width
            - self.left_border
            - self.right_border
        )
        n_horizontal_gaps = self.n_layers + 1
        self.between_layer_gap = horizontal_gap_total / n_horizontal_gaps

    def find_between_node_gap(self):
        """
        How big is the vertical gap between_node images?
        """
        vertical_gap_total = (
            self.figure_height
            - self.top_border
            - self.bottom_border
            - self.max_nodes
            * self.node_image_height
        )
        n_vertical_gaps = self.max_nodes - 1
        self.between_node_gap = vertical_gap_total / n_vertical_gaps

    def find_error_image_position(self):
        """
        Where exactly should the error image be positioned?
        """
        self.error_image_width = (
            self.input_image_width
            * self.error_image_scale
        )
        self.error_image_height = (
            self.input_image_height
            * self.error_image_scale
        )
        self.error_image_bottom = (
            self.input_image_bottom
            - self.input_image_height
            * self.error_gap_scale
            - self.error_image_height
        )
        error_image_center = (
            self.figure_width
            - self.right_border
            - self.input_image_width / 2
        )
        self.error_image_left = (
            error_image_center
            - self.error_image_width / 2
        )

    def add_input_image(self, fig, image_axes, nn, inputs):
        """
        All Axes to be added use the rectangle specification
            (left, bottom, width, height)
        """
        input_image = inputs.reshape(self.n_image_rows, self.n_image_cols)
        absolute_pos = (
            self.left_border,
            self.input_image_bottom,
            self.input_image_width,
            self.input_image_height)
        ax_input = self.add_image_axes(fig, image_axes, absolute_pos)
        ax_input.imshow(
            input_image,
            vmin=self.im_vmin,
            vmax=self.im_vmax,
            cmap=self.cmap,
            zorder=6,
        )
        image_axes.append([ax_input])

    def add_node_images(self, fig, i_layer, image_axes, nn, inputs):
        """
        Add in all the node images for a single layer
        """
        node_activities = nn.forward_prop_to_layer(inputs, i_layer)
        node_image_left = (
            self.left_border
            + self.input_image_width
            + i_layer * self.node_image_width
            + (i_layer + 1) * self.between_layer_gap
        )
        n_nodes = self.n_nodes[i_layer]
        total_layer_height = (
            n_nodes * self.node_image_height
            + (n_nodes - 1) * self.between_node_gap
        )
        layer_bottom = (self.figure_height - total_layer_height) / 2
        layer_axes = []
        for i_node in range(n_nodes):
            node_signal = np.zeros(n_nodes)
            node_signal[i_node] = 1
            node_signature = nn.forward_prop_from_layer(node_signal, i_layer)
            node_image = node_signature.reshape(
                self.n_image_rows, self.n_image_cols)
            node_image *= node_activities[i_node]

            node_image_bottom = (
                layer_bottom + i_node * (
                    self.node_image_height + self.between_node_gap))

            absolute_pos = (
                node_image_left,
                node_image_bottom,
                self.node_image_width,
                self.node_image_height)
            ax = self.add_image_axes(fig, image_axes, absolute_pos)
            ax.imshow(
                node_image,
                vmin=self.im_vmin,
                vmax=self.im_vmax,
                cmap=self.cmap,
                zorder=6,
            )
            layer_axes.append(ax)
        image_axes.append(layer_axes)

    def add_output_image(self, fig, image_axes, nn, inputs):
        outputs = nn.forward_prop(inputs)
        output_image = outputs.reshape(self.n_image_rows, self.n_image_cols)
        output_image_left = (
            self.figure_width
            - self.input_image_width
            - self.right_border
        )
        absolute_pos = (
            output_image_left,
            self.input_image_bottom,
            self.input_image_width,
            self.input_image_height)
        ax_output = self.add_image_axes(fig, image_axes, absolute_pos)
        ax_output.imshow(
            output_image,
            vmin=self.im_vmin,
            vmax=self.im_vmax,
            cmap=self.cmap,
            zorder=6,
        )
        image_axes.append([ax_output])

    def add_error_image(self, fig, image_axes, nn, inputs):
        outputs = nn.forward_prop(inputs)
        errors = inputs - outputs
        error_image = errors.reshape(self.n_image_rows, self.n_image_cols)
        absolute_pos = (
            self.error_image_left,
            self.error_image_bottom,
            self.error_image_width,
            self.error_image_height)
        ax_error = self.add_image_axes(fig, image_axes, absolute_pos)
        ax_error.imshow(
            error_image,
            vmin=self.im_vmin,
            vmax=self.im_vmax,
            cmap=self.error_cmap,
            zorder=6,
        )

    def add_image_axes(self, fig, image_axes, absolute_pos):
        """
        Locate the Axes for the image corresponding to this node
        within the Figure.

        absolute_pos: Tuple of
            (left_position, bottom_position, width, height)
        in inches on the Figure.
        """
        scaled_pos = (
            absolute_pos[0] / self.figure_width,
            absolute_pos[1] / self.figure_height,
            absolute_pos[2] / self.figure_width,
            absolute_pos[3] / self.figure_height)
        ax = fig.add_axes(scaled_pos)
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.tick_params(
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False)
        ax.spines["top"].set_color(self.tan)
        ax.spines["bottom"].set_color(self.tan)
        ax.spines["left"].set_color(self.tan)
        ax.spines["right"].set_color(self.tan)
        return ax

    def add_layer_connections(self, ax_boss, image_axes):
        """
        Add in the connectors between all the layers
        Treat the input image as the first layer and
        the output layer as the last.
        """
        for i_start_layer in range(len(image_axes) - 1):
            n_start_nodes = len(image_axes[i_start_layer])
            n_end_nodes = len(image_axes[i_start_layer + 1])
            x_start = image_axes[i_start_layer][0].get_position().x1
            x_end = image_axes[i_start_layer + 1][0].get_position().x0

            for i_start_ax, ax_start in enumerate(image_axes[i_start_layer]):
                ax_start_pos = ax_start.get_position()
                y_start_min = ax_start_pos.y0
                y_start_max = ax_start_pos.y1
                start_spacing = (y_start_max - y_start_min) / (n_end_nodes + 1)

                for i_end_ax, ax_end in enumerate(
                    image_axes[i_start_layer + 1]
                ):
                    ax_end_pos = ax_end.get_position()
                    y_end_min = ax_end_pos.y0
                    y_end_max = ax_end_pos.y1
                    end_spacing = (y_end_max - y_end_min) / (n_start_nodes + 1)

                    # Spread out y_start and y_end a bit
                    y_start = y_start_min + start_spacing * (i_end_ax + 1)
                    y_end = y_end_min + end_spacing * (i_start_ax + 1)
                    self.plot_connection(
                        ax_boss, x_start, x_end, y_start, y_end)

    def plot_connection(self, ax_boss, x0, x1, y0, y1):
        """
        Represent the weights connecting nodes in one layer
        to nodes in the next.
        """
        weight = np.random.sample() * 2 - 1
        x = np.linspace(x0, x1, num=50)
        y = y0 + (y1 - y0) * (
            -np.cos(
                np.pi * (x - x0) / (x1 - x0)
            ) + 1) / 2
        if weight > 0:
            conn_color = self.tan
        else:
            conn_color = self.blue
        ax_boss.plot(x, y, color=conn_color, linewidth=weight)

    def save_nn_viz(self, fig, postfix="0"):
        """
        Generate a new filename for each step of the process.
        """
        base_name = "nn_viz_"
        filename = base_name + postfix + ".png"
        filepath = os.path.join(self.savedir, filename)
        fig.savefig(
            filepath,
            edgecolor=fig.get_edgecolor(),
            facecolor=fig.get_facecolor(),
            dpi=self.DPI,
        )
