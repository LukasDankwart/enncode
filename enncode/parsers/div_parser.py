import onnx
from .base_parser import BaseParser


class DivParser(BaseParser):
    """
    Parses the ONNX Div node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Div operation.

    """

    def parse(self, node, parser):
        """
        Parses the Div node and updates the parser's internal representation.

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
            - Appends a new entry to `parser.nodes` describing the Div node.
        """
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

        shape_input = parser.current_shape.copy()
        shape_output = shape_input

        inputs = [{'name': node.input[0], 'shape': shape_input}]

        # Second input is either in the initializers, the parser.intermediate_tensors_shapes, or it's a constant of shape [1]
        if node.input[1] in parser.initializer_shapes:
            inputs.append({'name': node.input[1], 'shape': parser.current_shape.copy()})
        elif node.input[1] in parser.intermediate_tensors_shapes:
            inputs.append({'name': node.input[1], 'shape': parser.intermediate_tensors_shapes[node.input[1]]})
        else:
            inputs.append({'name': node.input[1], 'shape': [1]})

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
        parser.current_shape = shape_output.copy()

        # Adding the new node to the list
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
