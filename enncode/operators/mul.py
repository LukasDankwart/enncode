import numpy as np
from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string

class MulOperator(BaseOperator):
    """
    Implements the element-wise multiplication operator.

    Attributes:
        node (dict): A dictionary representing the ONNX node.
        input1 (str): The name of the first input tensor or scalar.
        output (str): The name of the output tensor.
        input1_shape (list): The shape of the first input.
        output_shape (list): The shape of the output.
        initializers (dict): A dictionary containing constant values for any node inputs.
    """
    def __init__(self, node, initializers):
        """
        Initializes the Mul-Operator with the node and initializers information.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).
        """
        super().__init__(node, initializers)
        self.node = node
        self.input1 = node["input"][0]["name"]
        self.multiplier = node["input"][1]["name"]
        self.output = node["output"][0]["name"]
        self.input1_shape = node["input"][0]["shape"]
        self.multiplier_shape = node["input"][1]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = node["initializers"]

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints for the Mul operation.

        This method encodes the element-wise multiplication of two inputs, which may be
        scalars or tensors.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to either Gurobi variables or constant values.

        Raises:
            ValueError: If any required input or output variable is missing,
            or if tensor shapes do not match (for elementwise tensor with tensor multiplication).
        """
        # Since both inputs can be a tensor or scalar, both have to be fetched and converted.
        # Variable inputs are stored as a dictionary, while constant inputs are stored as np.ndarrays
        var_input1 = self.initializers.get(self.input1)
        if var_input1 is None:
            var_input1 = variables.get(self.input1)
            if isinstance(var_input1, list):
                var_input1 = np.array(var_input1, dtype=np.float32)

        multiplier = self.initializers.get(self.multiplier)
        if multiplier is None:
            multiplier = variables.get(self.multiplier)
            if isinstance(multiplier, list):
                multiplier = np.array(multiplier, dtype=np.float32)

        # If shape of both inputs are different and none of them is a scalar (shape=1), elementwise mul. is not possible
        if self.multiplier_shape != self.input1_shape:
            if not self.multiplier_shape == 1 or self.input1_shape == 1:
                raise ValueError(f"A tensor cant be multiplied by another tensor, "
                                 f"if their shapes are different. Input has {self.input1_shape} but multiplier has {self.multiplier_shape}.")

        var_output = variables.get(self.output)
        var_output_shape = self.output_shape

        gurobi_model.update()

        if var_input1 is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for input '{self.input1}' not found."
                )
        if var_output is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for output '{self.output}' not found."
                )

        # Generate all indices for the output tensor
        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:

            # In both upper cases, a multi. from tensor with scalar is interpreted
            if self.input1_shape == 1:
                expression = var_input1 * multiplier[idx]
            elif self.multiplier_shape == 1:
                expression = var_input1[idx] * multiplier
            # Otherwise an elementwise multiplication is interpreted
            else:
                expression = var_input1[idx] * multiplier[idx]

            if isinstance(idx, tuple):
                constraint_name = f"Div_{self.output}_{'_'.join(map(str, idx))}"
            else:
                constraint_name = f"Div_{self.output}_{idx}"

            gurobi_model.addConstr(var_output[idx] == expression, name=constraint_name)


