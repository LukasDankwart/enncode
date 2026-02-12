from .base_parser import BaseParser

class ConcatParser(BaseParser):
    """
    Parses the ONNX Concat node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Concat operation.

    """
    def parse(self, node, parser):
        """
        Parses the Concat node and updates the parser's internal representation.

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
            - Appends a new entry to `parser.nodes` describing the Concat node.
        """

        # Number of tensors to be concatenated should be equal 2
        if len(node.input) != 2:
            raise ValueError(
                    f"Error in concat node '{node.name}':"
                    f"Number of inputs to the concat node isn't 2."
                )

        # Only supporting concatenation along axis 0
        axis = 0

        # Shapes of the inputs extracted either from the initializer_shapes, the intermediate_tensors_shapes dicts or from in/outputs
        shape_input1 = parser.initializer_shapes.get(node.input[0])
        if shape_input1 is None:
            shape_input1 = parser.intermediate_tensors_shapes.get(node.input[0])
        if shape_input1 is None:
            shape_input1 = parser.get_input_output_tensors_shapes().get(node.input[0])

        shape_input2 = parser.initializer_shapes.get(node.input[1])
        if shape_input2 is None:
            shape_input2 = parser.intermediate_tensors_shapes.get(node.input[1])
        if shape_input2 is None:
            shape_input2 = parser.get_input_output_tensors_shapes().get(node.input[1])

        if shape_input1 is None or shape_input2 is None:
            raise ValueError(
                    f"Error in concat node '{node.name}':"
                    f"Neither input_shape1 or input_shape2 should be NONE."
                )

        shape_input1 = list(shape_input1)
        shape_input2 = list(shape_input2)

        inputs = [
            {'name': node.input[0], 'shape': shape_input1},
            {'name': node.input[1], 'shape': shape_input2}
        ]

        # Only axis dimension can be different
        if shape_input1[1:] != shape_input2[1:]:
            raise ValueError(
                    f"Error in concat node '{node.name}':"
                    f"Input 1 and 2 can't be concatenated along axis 0 because their non-concatenating dimensions differ."
                )

        if len(shape_input1) == 1 and len(shape_input2) == 1:
            output_shape = [shape_input1[0] + shape_input2[0]]
        else:
            output_shape = [shape_input1[0] + shape_input2[0]] + shape_input1[1:]

        outputs = [{'name': node.output[0], 'shape': output_shape.copy()}]
        parser.intermediate_tensors_shapes[node.output[0]] = output_shape.copy()
        parser.current_shape = output_shape.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': {'axis' : axis},
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
