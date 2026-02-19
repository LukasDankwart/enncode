import numpy as np
from gurobipy import GRB
from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string

class ReLUOperator(BaseOperator):
    """
    Implements the ReLU (Rectified Linear Unit) operator.

    Attributes:
        input (str): The name of the input tensor.
        output (str): The name of the output tensor.
        input_shape (list): Shape of the input tensor.
        output_shape (list): Shape of the output tensor.
    """
    def __init__(self, node, initializers):
        """
        Initializes the ReLUOperator with the given node and initializers.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).
            It's unused here, but included for consistency with the base operator.
        """
        super().__init__(node, initializers)
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]
        self.input_shape = node["input"][0]["shape"]
        self.output_shape = node["output"][0]["shape"]

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints to encode the ReLU operation.

        For encoding, the Gurobi specific "addGenContrMax" function is used. Please see
        Documentation of Gurobi Machine Learning for more details.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to Gurobi variables or constant values.
              Expected to include a binary variable named "relu_binary_<output>" for this operator.

        Raises:
            ValueError: If input, output, or the required binary variable is missing from `variables`.
        """
        var_input = variables[self.input]
        var_output = variables[self.output]
        var_output_shape = self.output_shape

        if var_input is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for input '{self.input}' not found."
            )
        if var_output is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for input '{self.output}' not found."
            )
        gurobi_model.update()

        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:
            constraint_name = f"ReLU_{self.output}_{'_'.join(map(str, idx))}"

            gurobi_model.addGenConstrMax(
                var_output[idx],
                [var_input[idx]],
                constant=0.0,
                name=constraint_name
            )