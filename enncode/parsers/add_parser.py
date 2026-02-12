from .base_parser import BaseParser

class AddParser(BaseParser):
    """
    Parses the ONNX Add node.

    This parser extracts the necessary inputs and outputs, determines their
    shapes, and adds an entry to the parser's node list representing the
    Add operation.

    """

    def parse(self, node, parser):
        """
        Parses the Add node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Appends a new entry to `parser.nodes` describing the Add node.
        """
        inputs = []
        for input in node.input:
            if input in parser.intermediate_tensors_shapes:
                inputs.append({'name': input, 'shape': parser.intermediate_tensors_shapes.get(input)})
            elif input in parser.input_output_tensors_shapes:
                inputs.append({'name': input, 'shape': parser.input_output_tensors_shapes.get(input)})
            elif input in parser.initializer_shapes:
                inputs.append({'name': input, 'shape': parser.initializer_shapes.get(input)})
            else:
                raise KeyError(f"while Parsing input {input} from node {node.name}, corresponding shape couldn't be extracted.")

        current_shape = parser.current_shape.copy()
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
