from .base_parser import BaseParser
import numpy as np

class ReshapeParser(BaseParser):
    """
    Parses the ONNX Reshape node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Reshape operation.

    """
    def parse(self, node, parser):
        """
        Parses the Reshape node and updates the parser's internal representation.

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
            - Appends a new entry to `parser.nodes` describing the Reshape node.
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
        new_shape = list(parser.constant_values.get(node.input[1]))[1:]

        if -1 in new_shape:
            pos = new_shape.index(-1)
            input_elements = np.prod(np.array(shape_input))
            res_dimensions = np.prod(np.array(new_shape[:pos]))
            res_dimensions *= np.prod(np.array(new_shape[pos + 1:]))
            new_dimension = int(input_elements / res_dimensions)
            if not input_elements % res_dimensions == 0:
                raise ValueError(f"Error occured while parsing target shape. "
                                 f"New dimension at entry {pos} in target shape couldn't be determined properly.")
            new_shape[pos] = new_dimension

        shape_output = list(new_shape) if new_shape != -1 else [1]
        filtered_shape_tensor_out = [dim for dim in shape_output if dim > 0]
        inputs = [
            {'name': node.input[0], 'shape': shape_input},
            {'name': node.input[1], 'shape': new_shape}
        ]
        outputs = [{'name': node.output[0], 'shape': filtered_shape_tensor_out}]
        parser.intermediate_tensors_shapes[node.output[0]] = filtered_shape_tensor_out
        parser.current_shape = filtered_shape_tensor_out.copy()
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': {},
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })