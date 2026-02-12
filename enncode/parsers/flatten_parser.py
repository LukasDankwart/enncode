from .base_parser import BaseParser
from ..utils import _node_to_string
class FlattenParser(BaseParser):
    """
    Parses the ONNX Flatten node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Flatten operation.

    """
    def parse(self, node, parser):
        """
        Parses the Flatten node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Raises:
            ValueError: If the `axis_attribute` of the nodes is None or not equal 1 (limited to 1 at the moment).

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Updates `parser.current_shape` with the shape of the output.
            - Appends a new entry to `parser.nodes` describing the Flatten node.
        """
        axis_attribute = None

        for attribute in node.attribute:

            if attribute.name == 'axis':
                axis_attribute = attribute.i
                break

        if axis_attribute is None or axis_attribute != 1:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Unsupported axis in Flatten node '{node.name}'."
                )

        # Parse the real input shape, either from intermediate tensors or real inputs of network
        current_shape = None
        if node.input[0] in parser.intermediate_tensors_shapes:
            current_shape = parser.intermediate_tensors_shapes.get(node.input[0])
        elif node.input[0] in parser.input_output_tensors_shapes:
            current_shape = parser.input_output_tensors_shapes.get(node.input[0])
        elif node.input[0] in parser.initializer_shapes:
            current_shape = parser.initializer_shapes.get(node.input[0])
        else:
            raise KeyError(
                f"While parsing input {input} from node {node.name}, corresponding shape couldn't be extracted.")
        parser.current_shape = current_shape

        current_shape = parser.current_shape.copy()
        flattened_dim = 1

        for dim in current_shape:
            flattened_dim *= dim

        shape_tensor_out = [flattened_dim]
        inputs = [{'name': node.input[0], 'shape': current_shape}]
        outputs = [{'name': node.output[0], 'shape': shape_tensor_out}]
        parser.intermediate_tensors_shapes[node.output[0]] = shape_tensor_out.copy()
        parser.current_shape = shape_tensor_out.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': {'axis': axis_attribute},
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
