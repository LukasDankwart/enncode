from .add_parser import AddParser
from .gemm_parser import GemmParser
from .matmul_parser import MatMulParser
from .relu_parser import ReluParser
from .sub_parser import SubParser
from .concat_parser import ConcatParser
from .reshape_parser import ReshapeParser
from .flatten_parser import FlattenParser
from .conv_parser import ConvParser
from .unsqueeze_parser import UnsqueezeParser
from .maxpool_parser import MaxPoolParser
from .averagepool_parser import AveragePoolParser
from .dropout_parser import DropoutParser
from .constant_parser import ConstantParser
from .identity_parser import IdentityParser
from .batch_normalization_parser import BatchNormalization
from .mul_parser import MulParser
from .div_parser import DivParser


class ParserFactory:
    """
    Factory for creating parser instances based on node types.

    This factory maintains a registry that maps ONNX node types to corresponding parser classes.

    Attributes:
        parsers (dict): A mapping of ONNX operators to their respective parser classes.

    """
    def __init__(self):
        self.parsers = {
            'Add': AddParser,
            'Gemm': GemmParser,
            'MatMul': MatMulParser,
            'Relu': ReluParser,
            'Sub': SubParser,
            'Concat': ConcatParser,
            'Constant': ConstantParser,
            'Reshape': ReshapeParser,
            'Flatten': FlattenParser,
            'Conv': ConvParser,
            'Unsqueeze': UnsqueezeParser,
            'MaxPool': MaxPoolParser,
            'AveragePool': AveragePoolParser,
            'Dropout': DropoutParser,
            'Identity': IdentityParser,
            'BatchNormalization': BatchNormalization,
            'Div': DivParser,
            'Mul': MulParser
        }

    def get_parser(self, op_type):
        """
        Creates and returns a parser instance corresponding to the node type.

        Args:
            op_type (string): The name of the ONNX node being parsed.

        Returns:
            Parser: An instance of the parser class associated with the node.

        Raises:
            NotImplementedError: If the node's type is not found in the registry.
        """
        parser = self.parsers.get(op_type)
        if not parser:
            raise NotImplementedError(f"Parser for operator '{op_type}' is not supported.")
        return parser
