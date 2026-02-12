from .base_parser import BaseParser
import math

class MaxPoolParser(BaseParser):
    """
    Parses the ONNX MaxPool node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    MaxPool operation.

    """
    def parse(self, node, parser):
        """
        Parses the MaxPool node and updates the parser's internal representation.

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
            - Appends a new entry to `parser.nodes` describing the MaxPool node.
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

        shape_tensor_input = parser.current_shape.copy()
        kernel_shape = [1, 1]
        strides = [1, 1]
        pads = [0, 0, 0, 0]
        ceil_mode = 0
        dilations = [1, 1]

        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                kernel_shape = list(attr.ints)
            elif attr.name == 'strides':
                strides = list(attr.ints)
            elif attr.name == 'pads':
                pads = list(attr.ints)
            elif attr.name == 'ceil_mode':
                ceil_mode = attr.i
            elif attr.name == 'dilations':
                dilations = list(attr.ints)

        channels, height_in, width_in = shape_tensor_input
        kernel_height, kernel_width = kernel_shape
        stride_h, stride_w = strides
        pad_top, pad_left, pad_bottom, pad_right = pads

        if ceil_mode:
            height_out = math.ceil(((height_in + pad_top + pad_bottom) - kernel_height) / stride_h) + 1
            width_out = math.ceil(((width_in + pad_left + pad_right) - kernel_width) / stride_w) + 1
        else:
            height_out = math.floor(((height_in + pad_top + pad_bottom) - kernel_height) / stride_h) + 1
            width_out = math.floor(((width_in + pad_left + pad_right) - kernel_width) / stride_w) + 1

        shape_tensor_output = [channels, height_out, width_out]
        inputs = [{'name': node.input[0], 'shape': shape_tensor_input}]
        outputs = [{'name': node.output[0], 'shape': shape_tensor_output}]

        attributes = {
            "kernel_shape": kernel_shape,
            "strides": strides,
            "pads": pads,
            "ceil_mode": ceil_mode,
            "dilations": dilations,
            }


        parser.intermediate_tensors_shapes[node.output[0]] = shape_tensor_output.copy()
        parser.current_shape = shape_tensor_output.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
