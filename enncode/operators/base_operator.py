from abc import ABC, abstractmethod

class BaseOperator(ABC):
    """
    Serves as the abstract base class for all operator implementations.

    Each operator in the library should inherit from this class and implement the `apply_constraints` method,
    which is responsible for adding the corresponding Gurobi constraints.

    """

    def __init__(self, node, initializers):
        """
        Initializes the base operator with node and initializer data.

        Args:
            node (dict): A dictionary containing the ONNX node definition.
            initializers (dict): A dictionary of constant tensors or values.
        """
        self.node = node
        self.initializers = initializers

    @abstractmethod
    def apply_constraints(self, gurobi_model, variables):
        """
        Applies operator-specific constraints to the Gurobi model.

        Subclasses must override this method to add
        the necessary constraints for the corresponding operator.

        Args:
            gurobi_model (gurobipy.Model): The model to which constraints are added.
            variables (dict): Maps tensor names to either Gurobi variables or
                constant values used during the constraint-building process.
        """
        pass
