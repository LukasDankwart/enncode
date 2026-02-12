from gurobipy import GRB
from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string

class DropoutOperator(BaseOperator):
    """
    Implements the dropout operator in inference mode.

    In inference mode, dropout is disabled so that the input is passed through unchanged.

    Attributes:
        node (dict): A dictionary representing the ONNX node.
        input (str): The name of the input tensor.
        output (str): The name of the output tensor.
        input_shape (list): The shape of the input tensor.
        output_shape (list): The shape of the output tensor.
        ratio (float): The dropout ratio (provided for reference, not used in inference mode).
        training_mode (bool): Forced to False, indicating that the operator functions only in inference mode.
    """
    def __init__(self, node, initializers):
        """
        Initializes the DropoutOperator with the node and initializer information.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to contain keys such as
                "input", "output", and "attributes".
            initializers (dict): A dictionary of initial values for any constant tensors.
        """
        super().__init__(node, initializers)
        self.node = node
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]
        self.input_shape = node["input"][0]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.ratio = node["attributes"].get('ratio', 0.5)
        self.training_mode = False

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints for the dropout operation in inference mode.

        In inference mode, dropout is disabled and the operator functions as an identity.

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
                f"Error in {_node_to_string(self.node)}: "
                f"Variable for input '{self.input}' not found."
            )
        if var_output is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}: "
                f"Variable for output '{self.output}' not found."
            )

        gurobi_model.update()

        output_indices = list(product(*[range(dim) for dim in self.output_shape]))

        # Inference mode only, so the input passes through unchanged
        for idx in output_indices:
            gurobi_model.addConstr(
                var_output[idx] == var_input[idx],
                name=f"Dropout_Inference_{self.output}_{'_'.join(map(str, idx))}"
            )
