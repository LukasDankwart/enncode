from .base_parser import BaseParser

class DropoutParser(BaseParser):
    """
    Parses the ONNX Dropout node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Dropout operation.

    """
    def parse(self, node, parser):
        """
        Parses the Dropout node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Appends a new entry to `parser.nodes` describing the Dropout node.
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

        inputs = [{'name': node.input[0], 'shape': parser.current_shape.copy()}]
        outputs = [{'name': node.output[0], 'shape': parser.current_shape.copy()}]
        if len(node.output) > 1:
            outputs.append({'name': node.output[1], 'shape': parser.current_shape.copy()})
        ratio = 0.5
        training_mode = False
        for attr in node.attribute:
            if attr.name == 'ratio':
                ratio = attr.f
            elif attr.name == 'training_mode':
                training_mode = attr.i
        attributes = {
            "ratio" : ratio,
            "training_mode" : training_mode
            }

        parser.intermediate_tensors_shapes[node.output[0]] = parser.current_shape.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
