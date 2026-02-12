import numpy as np
from gurobipy import quicksum
from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string

class MatMul(BaseOperator):
    """
    Implements the MatMul (matrix multiplication) operator.

    Attributes:
        input1 (str): The name of the first (left-hand side) input tensor.
        input2 (str): The name of the second (right-hand side) input tensor.
        output (str): The name of the output tensor.
        input1_shape (list): The shape of the first input.
        input2_shape (list): The shape of the second input.
        output_shape (list): The shape of the output.
        initializers (dict): A dictionary of initial values for any constant tensors.
        constants (dict): A dictionary with additional constant values if not found in `initializers`.
    """

    def __init__(self, node, initializers):
        """
        Initializes the MatMul operator with the given node and initializers.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).

        """
        super().__init__(node, initializers)
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]
        self.output = node["output"][0]["name"]
        self.input1_shape = node["input"][0]["shape"]
        self.input2_shape = node["input"][1]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = initializers
        self.constants = node["constants"]
        self.attributes = node["attributes"]

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints to represent the matrix multiplication operation.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model in which constraints are created.
            variables (dict): A dictionary mapping tensor names to Gurobi variables or constant values.

        Raises:
            ValueError: If the second input's initializer or constants data is missing,
                or if the operator's internal shapes are unexpected.
            IndexError: If any dimension in the resulting weights array is out of
                bounds for the required operation.
        """
        var_input1 = variables.get(self.input1)
        if var_input1 is None:
            var_input1 = self.initializers.get(self.input1)
        if var_input1 is None:
            var_input1 = np.array(self.constants[self.input1])
        
        var_input2 = self.initializers.get(self.input2)
        if var_input2 is None:
            var_input2 = np.array(self.constants[self.input2])
        if var_input2 is None:
            var_input2 = variables.get(self.input2)

        var_output = variables[self.output]
        var_input1_shape = self.input1_shape
        var_input2_shape = self.input2_shape 
        var_output_shape = self.output_shape
        transB = self.attributes.get('transB', 0)
        transA = self.attributes.get('transA', 0)

        if var_input1 is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for input '{self.input}' not found."
            )
        if var_output is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for input '{self.output}' not found."
            )
        if var_input2 is None:
            raise ValueError(f"Initializer for '{self.input2}' not found or is None.")

        gurobi_model.update()

        if isinstance(var_input1_shape, int):
            var_input1_shape = [var_input1_shape]
        if isinstance(var_output_shape, int):
            var_output_shape = [var_output_shape]

        if transB == 1:
            var_input2 = var_input2.T
        if transA == 1:
            var_input1 = var_input1.T

        sum_dim = var_input1_shape[-1]

        # Generate all multi-dimensional indices for the input tensor
        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:

            if idx[-1] >= var_input2_shape[-1]:
                raise IndexError(
                    f"Error in {_node_to_string(self.node)}:"
                    f"Index {idx[-1]} out of bounds for var_input2 with shape {var_input2_shape[-1]} "
                )

            expression = quicksum(
                var_input1[(k,)] * float(var_input2[(k, idx[-1])])
                for k in range(sum_dim)
            )

            gurobi_model.addConstr(
                var_output[idx] == expression,
                name=f"MatMul_{self.output}_{'_'.join(map(str, idx))}"
            )
