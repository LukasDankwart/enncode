from .base_parser import BaseParser
import onnx

class GemmParser(BaseParser):
    """
    Parses the ONNX Gemm node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Gemm operation.

    """
    def parse(self, node, parser):
        """
        Parses the Gemm node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Updates `parser.current_shape` with the shape of the output.
            - Appends a new entry to `parser.nodes` describing the Gemm node.
        """
        shape_weights = list(parser.initializer_shapes.get(node.input[1], [1]))
        shape_bias = list(parser.initializer_shapes.get(node.input[2], [1]))
        shape_input = [shape_weights[1]]
        shape_output = [shape_bias[0]]
        parser.current_shape = shape_output.copy()

        inputs = [
            {'name': node.input[0], 'shape': shape_input},
            {'name': node.input[1], 'shape': shape_weights},
            {'name': node.input[2], 'shape': shape_bias}
        ]
        outputs = [{'name': node.output[0], 'shape': shape_output}]
        attributes = {}
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.FLOAT:
                value = attribute.f
            elif attribute.type == onnx.AttributeProto.INT:
                value = attribute.i
            else:
                value = None
            attributes[attribute.name] = value

        parser.intermediate_tensors_shapes[node.output[0]] = shape_output
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
