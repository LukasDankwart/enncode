from gurobipy import GRB
from itertools import product
import numpy as np
from .base_operator import BaseOperator
from ..utils import _node_to_string

class ReshapeOperator(BaseOperator):
    """
    Implements the reshape operator.

    Attributes:
        input (str): The name of the input tensor.
        output (str): The name of the output tensor.
        input_shape (list): The shape of the input tensor.
        output_shape (list): The shape of the output tensor.
    """

    def __init__(self, node, initializers):
        """
        Initializes the ReshapeOperator with the given node and initializers.

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
        Applies the Gurobi constraints for the Reshape operation.

        This method enforces a one-to-one mapping between each element of the input
        tensor and the corresponding element of the output tensor.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to Gurobi variables or constant values.

        Raises:
            ValueError: If the input or output variable is missing,
            or if the total number of elements in input and output shapes do not match.
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
                f"Variable for output '{self.output}' not found."
                )

        gurobi_model.update()

        input_total = np.prod(var_input_shape)
        output_total = np.prod(var_output_shape)

        if input_total != output_total:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Total elements mismatch: input has {input_total}, output has {output_total}."
            )

        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:
            # Compute the linear index for the current output index
            linear_idx = np.ravel_multi_index(idx, var_output_shape)

            # Convert the linear index to the corresponding input index
            input_idx = np.unravel_index(linear_idx, var_input_shape)

            constraint_name = f"Reshape_{self.output}_{'_'.join(map(str, idx))}"

            gurobi_model.addConstr(
                var_output[idx] == var_input[input_idx],
                name=constraint_name
            )
