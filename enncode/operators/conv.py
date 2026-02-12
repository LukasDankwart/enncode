from gurobipy import GRB
from itertools import product
import numpy as np
from .base_operator import BaseOperator
from ..utils import _node_to_string

class ConvOperator(BaseOperator):
    """
    Implements the 2D Convolution operator.

    Attributes:
        input (str): The name of the input tensor, with shape (channels, height_in, width_in).
        weights (str): The name of the weights tensor, with shape (feature_maps, channels/group, kernel_height, kernel_width).
        bias (str): The name of the bias tensor, if present.
        output (str): The name of the output tensor, with shape (feature_maps, height_out, width_out).
        input_shape (list): The shape of the input tensor.
        output_shape (list): The shape of the output tensor.
        pads (list): Padding applied [top, left, bottom, right].
        strides (list): The horizontal and vertical strides.
        dilations (list): The horizontal and vertical dilation factors.
        group (int): Number of groups dividing input and output channels.
        initializers (dict): A dictionary containing constant values for any node inputs.

    """

    def __init__(self, node, initializers):
        """
        Initializes the convolution operator with node and initializer information.


        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).

        """
        super().__init__(node, initializers)
        self.input = node["input"][0]["name"]
        self.weights = node["input"][1]["name"]
        self.bias = node["input"][2]["name"] if len(node["input"]) > 2 else None
        self.output = node["output"][0]["name"]
        self.input_shape = node["input"][0]["shape"]   # shape = [channels, height_in, width_in]
        self.output_shape = node["output"][0]["shape"] # shape = [feature_maps, height_out, width_out]
        self.initializers = initializers

        # Attributes
        self.pads = node["attributes"].get('pads', [0, 0, 0, 0])
        self.strides = node["attributes"].get('strides', [1, 1])
        self.dilations = node["attributes"].get('dilations', [1, 1])
        self.group = node["attributes"].get('group', 1)

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints to encode the convolution operation.

        Iterates over each element in the output tensor, computing the convolution
        sum from the corresponding part of the input tensor.
        This part is determined by stride, dilation, and padding attributes.
        If a bias tensor is present, it is added to each feature map.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to either Gurobi variables or constant values.


        Raises:
            ValueError: If the input or output variables are not found.
        """
        var_input = variables.get(self.input)
        var_output = variables.get(self.output)
        weights = self.initializers.get(self.weights)  # shape: [feature_maps, channels/group, kernel_height, kernel_width]
        bias = self.initializers.get(self.bias, np.zeros(weights.shape[0])) if weights is not None else None

        if var_input is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for input '{self.input}' not found."
                )
        if var_output is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for output '{self.output}' not found."
                )

        gurobi_model.update()
        channels, height_in, width_in = self.input_shape
        feature_maps, C_group, kernel_height, kernel_width = weights.shape
        height_out, width_out = self.output_shape[1], self.output_shape[2]
        pad_top, pad_left, pad_bottom, pad_right = self.pads
        stride_h, stride_w = self.strides
        dilation_h, dilation_w = self.dilations
        group = self.group

        channels_per_group = channels // group
        feature_maps_per_group = feature_maps // group

        # Iterate over the output tensor dimensions
        for m in range(feature_maps):
            group_idx = m // feature_maps_per_group
            for h_out in range(height_out):
                for w_out in range(width_out):
                    # Compute the receptive field in the input tensor
                    h_start = h_out * stride_h - pad_top
                    w_start = w_out * stride_w - pad_left

                    conv_sum = 0
                    for c in range(channels_per_group):
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                h_in = h_start + kh * dilation_h
                                w_in = w_start + kw * dilation_w

                                if 0 <= h_in < height_in and 0 <= w_in < width_in:
                                    input_idx = (group_idx * channels_per_group + c, h_in, w_in)
                                    conv_sum += weights[m, c, kh, kw] * var_input[input_idx]

                    if bias is not None:
                        conv_sum += bias[m]

                    output_idx = (m, h_out, w_out)
                    gurobi_model.addConstr(
                        var_output[output_idx] == conv_sum,
                        name=f"Conv_{self.output}_1_{m}_{h_out}_{w_out}"
                    )
