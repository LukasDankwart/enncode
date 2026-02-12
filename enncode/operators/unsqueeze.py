from itertools import product
from gurobipy import GRB
from .base_operator import BaseOperator
from ..utils import _node_to_string


class UnsqueezeOperator(BaseOperator):
    """
    Implements the Unsqueeze operator.

    Attributes:
        node (dict): A dictionary representing the ONNX node.
        input (str): The name of the input tensor.
        output (str): The name of the output tensor.
        input_shape (list): The shape of the input tensor.
        output_shape (list): The shape of the output tensor.
        axes (list of int): A list of axes at which to insert new dimensions.
    """
    def __init__(self, node, initializers):
        """
        Initializes the UnsqueezeOperator with the node and initializer information.

        Args:
            node (dict): A dictionary describing the ONNX node. It is expected to contain keys like "input",
                         "output", and "attributes".
            initializers (dict): A dictionary of initial values for any constant tensors.

        Raises:
            ValueError: If the 'axes' attribute is missing from the node.
        """
        super().__init__(node, initializers)
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]
        self.input_shape = node["input"][0]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.axes = node.get("attributes").get("axes")

        if not self.axes:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Unsqueeze node is missing the 'axes' attribute."
            )

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints to model the Unsqueeze operation.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to either Gurobi variables or constant values.

        Raises:
            ValueError: If any required input or output variable is missing.
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

        sorted_axes = sorted(self.axes)

        # Generate all indices for the output tensor
        output_indices = list(product(*[range(dim) for dim in self.output_shape]))

        for out_idx in output_indices:

            input_idx = list(out_idx)

            for axis in sorted_axes:

                if axis < 0:
                    axis += len(self.output_shape)

                input_idx.pop(axis)

            input_idx = tuple(input_idx)

            gurobi_model.addConstr(
                var_output[out_idx] == var_input[input_idx],
                name=f"Unsqueeze_{self.output}_{'_'.join(map(str, out_idx))}_eq_{'_'.join(map(str, input_idx))}"
            )
