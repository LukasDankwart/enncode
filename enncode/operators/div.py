import numpy as np
from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string

class DivOperator(BaseOperator):
    """
    Implements the element-wise division operator.

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
        Initializes the Div-Operator with the node and initializers information.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).
        """
        super().__init__(node, initializers)
        self.node = node
        self.input1 = node["input"][0]["name"]
        self.divisor = node["input"][1]["name"]
        self.output = node["output"][0]["name"]
        self.input1_shape = node["input"][0]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = node["initializers"]

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints for the Div operation.

        This method encodes the element-wise division of two inputs, which may be
        scalars or tensors.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to either Gurobi variables or constant values.

        Raises:
            ValueError: If any required input or output variable is missing,
            or if tensor shapes do not match (for tensor-divided by tensor).
        """
        var_input1 = self.initializers.get(self.input1)
        if var_input1 is None:
            var_input1 = variables.get(self.input1)

        divisor = self.initializers.get(self.divisor)
        if divisor is None:
            divisor = variables.get(self.divisor)

        if isinstance(divisor, np.ndarray):
            if list(divisor.shape) != self.input1_shape:
                raise ValueError(f"A tensor cant be divided by another tensor, "
                                 f"if their shapes are different. Input has {var_input1.shape} but divisor has {divisor.shape}.")
            if np.any(divisor == 0):
                raise ValueError(f"Divisor tensor has entry 0 which leads to division by zero.")

        if isinstance(divisor, np.float32) or isinstance(divisor, np.int32):
            if divisor == 0:
                raise ValueError(f"Divisor is 0 which leads to division by zero.")


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

            if isinstance(divisor, np.ndarray):
                expression = var_input1[idx] / divisor[idx]
            # Otherwise it is interpreted as tensor divided by a scalar (single divisor)
            else:
                expression = var_input1[idx] / divisor

            if isinstance(idx, tuple):
                constraint_name = f"Div_{self.output}_{'_'.join(map(str, idx))}"
            else:
                constraint_name = f"Div_{self.output}_{idx}"

            gurobi_model.addConstr(var_output[idx] == expression, name=constraint_name)

