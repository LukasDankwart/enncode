class InternalONNXRepresentation:
    """
    Represents the internal ONNX model after parsing.
    This representation supplies the ONNXToGurobi
    with the required attributes for building the Gurobi model.

    Attributes:
        initializers (dict): Contains the initial values from the parsed ONNX model.
        nodes (list): A list of dictionaries, each representing an ONNX node extracted by the parser.
        in_out_tensors_shapes (dict): Stores shapes for all input and output tensors.
    """
    def __init__(self, parser):
        self.initializers = parser.initializer_values
        self.nodes = parser.nodes
        self.in_out_tensors_shapes = parser.input_output_tensors_shapes
        self.input_node_name = parser.real_inputs_names
        self.output_node_name = parser.real_output_names