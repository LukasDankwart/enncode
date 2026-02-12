import numpy as np
from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string

class BatchNormalization(BaseOperator):
    """
    Implements the BatchNormalization operator.

    Attributes:
        name (str): The node's name in the ONNX graph.
        input (str): Name of the input tensor.
        weights (str): Name of the scaling factor tensor.
        bias (str): Name of the bias tensor.
        mean (str): Name of the mean tensor.
        variance (str): Name of the variance tensor.
        output (str): Name of the output tensor.
        shape_input_output (list): Shape of the input and output.
        initializers (dict): A dictionary of initial values for known constants.
        epsilon (float): A small constant to avoid division by zero.
    """
    def __init__(self, node, initializers):
        """
        Initializes the BatchNormalization operator with the given node and initializer information.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).

        """
        super().__init__(node, initializers)
        self.name = node["name"]
        self.input = node["input"][0]["name"]
        self.weights = node["input"][1]["name"]
        self.bias = node["input"][2]["name"]
        self.mean = node["input"][3]["name"]
        self.variance = node["input"][4]["name"]
        self.output = node["output"][0]["name"]
        self.shape_input_output = node["input"][0]["shape"]
        self.initializers = initializers
        self.epsilon = node["attributes"].get("epsilon", 1e-5)

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies Gurobi constraints for the BatchNormalization operation.

        It computes per-channel scaling and offset terms based on the formula:
            a[channel] = weights[channel] / sqrt(variance[channel] + epsilon)
            b[channel] = bias[channel] - (weights[channel] * mean[channel]) / sqrt(variance[channel] + epsilon)
        Then, for each element of the input, it creates a constraint ensuring:
            output[idx] = a[channel] * input[idx] + b[channel]

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints are added.
            variables (dict): A mapping of tensor names to Gurobi variables or constants.

        Raises:
            ValueError: If the input or output variables are not found.
        """
        weights = self.initializers.get(self.weights)
        bias = self.initializers.get(self.bias)
        mean = self.initializers.get(self.mean)
        variance = self.initializers.get(self.variance)
        var_input = variables[self.input]
        var_output = variables[self.output]

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
        
        if mean is None:
            mean = bias
        
        if variance is None:
            variance = weights
        

        gurobi_model.update()

        a = weights / np.sqrt(variance + self.epsilon)
        b = bias - (weights * mean) / np.sqrt(variance + self.epsilon)

        gurobi_model.update()

        output_indices = list(product(*[range(dim) for dim in self.shape_input_output]))

        for idx in output_indices:
            channel = idx[0]
            gurobi_model.addConstr(
                var_output[idx] == a[channel] * var_input[idx] + b[channel],
                name=f"BatchNorm_{self.name}_{'_'.join(map(str, idx))}"
            )


