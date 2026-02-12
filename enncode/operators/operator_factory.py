from .add import AddOperator
from .gemm import GemmOperator
from .matmul import MatMul
from .relu import ReLUOperator
from .sub import SubOperator
from .concat import ConcatOperator
from .reshape import ReshapeOperator
from .flatten import FlattenOperator
from .conv import ConvOperator
from .maxpool import MaxPoolOperator
from .averagepool import AveragePoolOperator
from .unsqueeze import UnsqueezeOperator
from .batch_normalization import BatchNormalization
from .dropout import DropoutOperator
from .identity import Identity
from .mul import MulOperator
from .div import DivOperator

class OperatorFactory:
    """
    Factory for creating operator instances based on node types.

    This factory maintains a registry that maps ONNX node types to corresponding operator classes.

    Attributes:
        node_handlers (dict): A mapping of ONNX operator to their respective operator classes.

    """

    def __init__(self):
        """
        Initializes the OperatorFactory with a predefined registry of operator classes.

        """
        self.node_handlers = {
            'Gemm': GemmOperator,
            'Add': AddOperator,
            'MatMul': MatMul,
            'Relu': ReLUOperator,
            'Sub': SubOperator,
            'Concat': ConcatOperator,
            'Reshape': ReshapeOperator,
            'Flatten': FlattenOperator,
            'Conv': ConvOperator,
            'MaxPool': MaxPoolOperator,
            'AveragePool': AveragePoolOperator,
            'Unsqueeze' : UnsqueezeOperator,
            'BatchNormalization' : BatchNormalization,
            'Dropout' : DropoutOperator,
            'Identity': Identity,
            'Mul': MulOperator,
            'Div': DivOperator
        }

    def create_operator(self, node, initializers):
        """
        Creates and returns an operator instance corresponding to the node type.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).

        Returns:
            Operator: An instance of the operator class associated with the node.

        Raises:
            NotImplementedError: If the node's type is not found in the registry.
        """
        op_type = node['type']
        handler_class = self.node_handlers.get(op_type)
        if not handler_class:
            raise NotImplementedError(f"Operator '{op_type}' is not supported.")
        return handler_class(node, initializers)
