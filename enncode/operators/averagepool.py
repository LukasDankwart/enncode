
from gurobipy import GRB, quicksum
from itertools import product
import numpy as np
from .base_operator import BaseOperator
from ..utils import _node_to_string


class AveragePoolOperator(BaseOperator):
    """
    Implements the 2D average pool operator.

    Attributes:
        input (str): The name of the input tensor.
        output (str): The name of the output tensor.
        input_shape (list): The shape of the input tensor.
        output_shape (list): The shape of the output tensor.
        pads (list): Padding applied [top, left, bottom, right].
        strides (list): The horizontal and vertical strides.
        dilations (list): The horizontal and vertical dilation factors.
        ceil_mode (int): The value indicating whether to use ceil or floor (default) to compute the output shape.
        count_include_pad (int):The value indicating whether to include pad pixels when calculating values for the edges.

    """

    def __init__(self, node, initializers):
        """
        Initializes the average pool operator with node and initializer information.


        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).

        """
        super().__init__(node, initializers)
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]
        self.input_shape = node["input"][0]["shape"]
        self.output_shape = node["output"][0]["shape"]

        self.kernel_shape = node["attributes"].get('kernel_shape', [1, 1])
        self.pads = node["attributes"].get('pads', [0, 0, 0, 0])
        self.strides = node["attributes"].get('strides', [1, 1])
        self.dilations = node["attributes"].get('dilations', [1, 1])
        self.ceil_mode = node["attributes"].get('ceil_mode', 0)
        self.count_include_pad = node["attributes"].get('count_include_pad', 0)

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints to encode the average pool operation.

        Iterates over each element in the output tensor, computing the average
        from the corresponding part of the input tensor.
        This part is determined by stride, dilation, and padding attributes.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to either Gurobi variables or constant values.


        Raises:
            ValueError: If the input or output variables are not found.
        """
        var_input = variables.get(self.input)
        var_output = variables.get(self.output)

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
        channels, height_out, width_out = self.output_shape
        kernel_height, kernel_width = self.kernel_shape
        stride_h, stride_w = self.strides
        pad_top, pad_left, pad_bottom, pad_right = self.pads
        dilation_h, dilation_w = self.dilations

        pool_size = kernel_height * kernel_width

        for c in range(channels):
            for h in range(height_out):
                for w in range(width_out):
                    h_start = h * stride_h - pad_top
                    w_start = w * stride_w - pad_left

                    pooling_elements = []

                    if self.count_include_pad:
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                h_in = h_start + kh * dilation_h
                                w_in = w_start + kw * dilation_w

                                if 0 <= h_in < height_in and 0 <= w_in < width_in:
                                    pooling_elements.append(var_input[c, h_in, w_in])
                                else:
                                    pooling_elements.append(0)
                        divisor = pool_size
                    else:
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                h_in = h_start + kh * dilation_h
                                w_in = w_start + kw * dilation_w

                                if 0 <= h_in < height_in and 0 <= w_in < width_in:
                                    pooling_elements.append(var_input[c, h_in, w_in])
                        divisor = len(pooling_elements) if pooling_elements else 1  # Avoid division by zero


                    if self.count_include_pad:
                        # Fixed divisor
                        average = quicksum(pooling_elements) / pool_size
                        gurobi_model.addConstr(
                            var_output[c, h, w] == average,
                            name=f"AveragePool_{self.output}_1_{c}_{h}_{w}_constraint"
                        )
                    else:
                        # Dynamic divisor
                        average = quicksum(pooling_elements) / divisor
                        gurobi_model.addConstr(
                            var_output[c, h, w] == average,
                            name=f"AveragePool_{self.output}_1_{c}_{h}_{w}_constraint"
                        )
