from abc import ABC, abstractmethod

class BaseParser(ABC):
    """
    Serves as the abstract base class for all parser implementations.

    Each parser in the library should inherit from this class and implement the `parses` method,
    which is responsible for parsing the corresponding ONNX node and updating the node list
    used ultimately in adding constraints to the Gurobi model.

    """
    @abstractmethod
    def parse(self, node, parser):
        """
        Parses the corresponding node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.
        """
        pass