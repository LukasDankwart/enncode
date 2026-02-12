from .base_parser import BaseParser

class SubParser(BaseParser):
    """
    Parses the ONNX Sub node.

    This parser extracts the necessary inputs and outputs, determines their
    shapes, and adds an entry to the parser's node list representing the
    Sub operation.

    """
    def parse(self, node, parser):
        """
        Parses the Sub node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Appends a new entry to `parser.nodes` describing the Sub node.
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

        inputs = [{'name': node.input[0], 'shape': current_shape.copy()}]

        # Second input is either in the initializers, the parser.intermediate_tensors_shapes, or it's a constant of shape [1]
        if node.input[1] in parser.initializer_shapes:
            inputs.append({'name': node.input[1], 'shape': current_shape.copy()})
        elif node.input[1] in parser.intermediate_tensors_shapes:
            inputs.append({'name': node.input[1], 'shape': parser.intermediate_tensors_shapes[node.input[1]]})
        else:
            inputs.append({'name': node.input[1], 'shape': [1]})

        outputs = [{'name': node.output[0], 'shape': current_shape.copy()}]
        parser.intermediate_tensors_shapes[node.output[0]] = current_shape.copy()

        # Adding the new node to the list
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': {},
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
