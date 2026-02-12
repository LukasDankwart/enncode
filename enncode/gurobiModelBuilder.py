from gurobipy import Model, GRB
from .operators.operator_factory import OperatorFactory
from .parser import ONNXParser
from .utils import _generate_indices

class GurobiModelBuilder:
    """
    Converts an ONNX model to a Gurobi optimization model by transforming the ONNX
    representation into an internal representation and then constructing the corresponding
    constraints for each operator.

    Attributes:
        model (gurobipy.Model): The Gurobi model being constructed.
        internal_onnx (InternalONNX): The internal representation of the parsed ONNX model,
            containing initializers, nodes, and input/output tensor shapes.
        initializers (dict): A dictionary containing the initial values extracted from the ONNX model.
        nodes (list): A list of dictionaries, each representing an ONNX node with its associated data.
        in_out_tensors_shapes (dict): A mapping of input and output tensor names to their shapes.
        operator_factory (OperatorFactory): Factory for creating operator instances based on node types.
        variables (dict): A mapping of tensor names to either Gurobi decision variables or constant values.
    """
    def __init__(self, onnx_model_path: str):
        """
        Initializes the ONNXToGurobi converter with the given ONNX model file path.

        This constructor loads the ONNX model, converts it into an internal representation,
        and initializes the attributes required for building the Gurobi model.

        Args:
            onnx_model_path (str): The file path to the ONNX model to be converted.
        """
        self.model = Model("NeuralNetwork")
        self.internal_onnx = ONNXParser(onnx_model_path)._parse_model()
        self.initializers = self.internal_onnx.initializers
        self.nodes = self.internal_onnx.nodes
        self.in_out_tensors_shapes = self.internal_onnx.in_out_tensors_shapes
        self.operator_factory = OperatorFactory()
        self.variables = {}

    def create_variables(self):
        """
        Creates Gurobi variables for the input/output tensors and intermediate nodes.

        """
        # Create variables for inputs and outputs
        for tensor_name, shape in self.in_out_tensors_shapes.items():
            indices = _generate_indices(shape)
            self.variables[tensor_name] = self.model.addVars(
                indices,
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                name=tensor_name
            )

        # Create variables for intermediate nodes
        for node in self.nodes:
            output_name = node['output'][0]['name']

            if node['type'] == "Constant":
                # Constants are not model variables
                if 'attributes' in node and node['attributes']:
                    self.variables[output_name] = node['attributes']['value']
                else:
                    self.variables[output_name] = 0

            elif node['type'] == "Relu":
                shape = node['output'][0]['shape']
                indices = _generate_indices(shape)
                """
                var_input = self.variables[node["input"][0]["name"]]

                # Create binary variables for ReLU indicator
                self.variables[f"relu_binary_{output_name}"] = self.model.addVars(
                    var_input.keys(),
                    vtype=GRB.BINARY,
                    name=f"relu_binary_{output_name}"
                )

                # Create output variables
                self.variables[output_name] = self.model.addVars(
                    indices,
                    vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY,
                    name=output_name
                )
                """
                self.variables[output_name] = self.model.addVars(
                    indices,
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    name=output_name
                )

            else:
                shape = node['output'][0]['shape']
                indices = _generate_indices(shape)
                self.variables[output_name] = self.model.addVars(
                    indices,
                    vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY,
                    name=output_name
                )

    def build_model(self):
        """
        Constructs the Gurobi model by creating variables and applying operator constraints.

        """
        self.create_variables()
        for node in self.nodes:
            if node['type'] != "Constant":
                operator = self.operator_factory.create_operator(node, self.initializers)
                operator.apply_constraints(self.model, self.variables)

    def get_gurobi_model(self):
        """
        Retrieves the Gurobi model after all constraints have been added.

        Returns:
            gurobipy.Model: The constructed Gurobi model reflecting the ONNX graph.
        """
        return self.model

    def get_input_vars(self):
        if len(self.internal_onnx.input_node_name) != 1:
            raise ValueError(f"The current model seems to have more than one input node, which isn't supported by this function.")
        input_name = self.internal_onnx.input_node_name[0]
        input_vars = self.variables.get(input_name)
        if input_vars is None:
            raise ValueError(f"Input variables couldn't be accessed.")
        return input_vars

    def get_output_vars(self):
        if len(self.internal_onnx.output_node_name) != 1:
            raise ValueError(f"The current model seems to have more than one output node, which isn't supported by this function.")
        output_name = self.internal_onnx.output_node_name[0]
        output_vars = self.variables.get(output_name)
        if output_vars is None:
            raise ValueError(f"Output variables couldn't be accessed.")
        return output_vars

