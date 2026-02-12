from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string

class Identity(BaseOperator):
    """
    Implements the Identity operator.

    """
    def __init__(self, node, initializers):
        """
        Initializes the Identity operator with the node and initializer information.

        Args:
            node (dict): A dictionary describing the ONNX node.
            initializers (dict): A dictionary of initial values for any constant tensors.
        """
        super().__init__(node, initializers)
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]
        self.input_shape = node["input"][0]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = node["initializers"]


    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints for the Identity operation.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model to which constraints should be added.
            variables (dict): A dictionary mapping tensor names to Gurobi variables or constant values.
        """
        var_input = variables.get(self.input, self.initializers.get(self.input))
        var_output = variables.get(self.output, self.initializers.get(self.output))

        if var_input is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}: Variable for input '{self.input}' not found."
            )
        if var_output is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}: Variable for output '{self.output}' not found."
            )

        gurobi_model.update()

        indices = list(product(*[range(dim) for dim in self.input_shape]))

        for idx in indices:
            gurobi_model.addConstr(
                var_output[idx] == var_input[idx],
                name=f"Identity_{self.output}_{'_'.join(map(str, idx))}"
            )
