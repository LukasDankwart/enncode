import numpy as np
from gurobipy import GRB
from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string
class ConcatOperator(BaseOperator):
    """
    Implements the concatenation operator.

    Attributes:
        inputs (list): A list of input tensor names to be concatenated.
        output (str): The name of the output tensor.
        inputs_shapes (list): A list of shapes for each input tensor.
        output_shape (list): The shape of the output tensor.
        axis (int): The axis along which the inputs are concatenated.
        Defaults to 0 if no axis attribute is specified in the node.
    """

    def __init__(self, node, initializers):
        """
        Initializes the ConcatOperator with the given node and initializer information.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).

        """
        super().__init__(node, initializers)
        self.inputs = [input['name'] for input in node["input"]]  # inputs to concatenate
        self.output = node["output"][0]['name']
        self.inputs_shapes = [input['shape'] for input in node["input"]]
        self.output_shape = node["output"][0]['shape']
        self.axis = 0

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints to encode the Concat operation.

        This method copies values from each input tensor into the correct portions
        of the output tensor along the specified concatenation axis.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to either Gurobi variables or constant values.


        Raises:
            ValueError: If any of the required input variables or the output variable is missing.

        """
        input_vars = [variables[input_name] for input_name in self.inputs]
        output_vars = variables[self.output]

        for name, var in zip(self.inputs, input_vars):
            if var is None:
                raise ValueError(
                    f"Error in {_node_to_string(self.node)}:"
                    f"Variable for input '{name}' not found."
                )
        if output_vars is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for output '{self.output}' not found."
            )

        current_offset = 0

        for input_var, input_shape in zip(input_vars, self.inputs_shapes):
            dim = input_shape[0]  # Since we're concatenating on the first dimension

            for i in range(dim):

                for other_indices in product(*[range(s) for s in input_shape[1:]]):
                    full_output_index = (current_offset + i,) + other_indices
                    full_input_index = (i,) + other_indices

                    gurobi_model.addConstr(
                        output_vars[full_output_index] == input_var[full_input_index],
                        name=f"Concat_{self.output}_{'_'.join(map(str, full_output_index))}"
                    )

            current_offset += dim