import numpy as np
from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string

class AddOperator(BaseOperator):
    """
    Implements the element-wise addition operator.

    Attributes:
        node (dict): A dictionary representing the ONNX node.
        input1 (str): The name of the first input tensor or scalar.
        input2 (str): The name of the second input tensor or scalar.
        output (str): The name of the output tensor.
        input1_shape (list): The shape of the first input.
        input2_shape (list): The shape of the second input.
        output_shape (list): The shape of the output.
        initializers (dict): A dictionary containing constant values for any node inputs.
    """
    def __init__(self, node, initializers):
        """
        Initializes the AddOperator with the node and initializers information.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).
        """
        super().__init__(node, initializers)
        self.node = node
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]      # Scalar or tensor
        self.output = node["output"][0]["name"]
        self.input1_shape = node["input"][0]["shape"]
        self.input2_shape = node["input"][1]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = node["initializers"]

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints for the Add operation.

        This method encodes the element-wise addition of two inputs, which may be
        scalars or tensors.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to either Gurobi variables or constant values.

        Raises:
            ValueError: If any required input or output variable is missing,
            or if tensor shapes do not match (for tensor-tensor addition).
        """
        var_input1 = self.initializers.get(self.input1)
        if var_input1 is None:
            var_input1 = variables.get(self.input1)

        var_input2 = self.initializers.get(self.input2)
        if var_input2 is None:
            var_input2 = variables.get(self.input2)

        var_output = variables.get(self.output)
        var_input1_shape = self.input1_shape
        var_input2_shape = self.input2_shape
        var_output_shape = self.output_shape

        gurobi_model.update()

        if var_input1 is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for input '{self.input1}' not found."
                )
        if var_input2 is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable or constant for input '{self.input2}' not found."
                )
        if var_output is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for output '{self.output}' not found."
                )

        # Generate all indices for the output tensor
        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:
            # Check if input2 is a tensor or a scalar
            if isinstance(var_input2, dict) or isinstance(var_input2, np.ndarray):
                if var_input1_shape != var_input2_shape:
                    raise ValueError(
                        f"Error in {_node_to_string(self.node)}:"
                        f"Shape mismatch: input1 shape {var_input1_shape} != input2 shape {var_input2_shape}"
                        )

                expression = var_input1[idx] + var_input2[idx]
            else:
                # input2 is a scalar
                expression = var_input1[idx] + var_input2

            if isinstance(idx, tuple):
                constraint_name = f"Add_{self.output}_{'_'.join(map(str, idx))}"
            else:
                constraint_name = f"Add_{self.output}_{idx}"

            gurobi_model.addConstr(var_output[idx] == expression, name=constraint_name)
