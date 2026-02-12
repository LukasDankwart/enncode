from gurobipy import GRB
from itertools import product
import numpy as np
from .base_operator import BaseOperator
from ..utils import _node_to_string

class FlattenOperator(BaseOperator):
    """
    Implements the Flatten operator.

    Attributes:
        input (str): The name of the input tensor to be flattened.
        output (str): The name of the output tensor after flattening.
        input_shape (list): The shape of the input tensor.
        output_shape (list): The shape of the output tensor.
    """

    def __init__(self, node, initializers):
        """
        Initializes the Flatten operator with node and initializer information.

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
        Applies the Gurobi constraints to encode the Flatten operation.

        This method verifies that the total number of elements in the input and output shapes match,
        and then maps each multidimensional index of the input to its corresponding flat index in the output.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to Gurobi variables or constant values.

        Raises:
            ValueError: If the input or output variables are not found,
            or if the total number of elements do not match between input and output.
        """
        var_input = variables.get(self.input)
        var_output = variables.get(self.output)
        var_input_shape = self.input_shape
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

        # Calculate total number of elements for input and output
        input_total = np.prod(var_input_shape)
        output_total = np.prod(var_output_shape)

        if input_total != output_total:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Total elements mismatch: input has {input_total}, output has {output_total}."
            )

        # Generate all multi-dimensional indices for the input tensor
        input_indices = list(product(*[range(dim) for dim in var_input_shape]))

        # Map each input index to a flat output index
        for flat_idx, multi_idx in enumerate(input_indices):
            constraint_name = f"Flatten_{self.output}_{flat_idx}"
            gurobi_model.addConstr(
                var_output[flat_idx,] == var_input[multi_idx],
                name=constraint_name
            )
