from gurobipy import Model, GRB
from .operators.operator_factory import OperatorFactory
from .parser import ONNXParser
from .utils import _generate_indices
from .compatibility import compatibility_check
import onnx
from onnxsim import simplify

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
    def __init__(self, onnx_model_path: str, simplification=True, compcheck=False, rtol=1e-03, atol=1e-03):
        """
        Initializes the ONNXToGurobi converter with the given ONNX model file path.

        This constructor loads the ONNX model, converts it into an internal representation,
        and initializes the attributes required for building the Gurobi model.

        Args:
            onnx_model_path (str): The file path to the ONNX model to be converted.
        """
        self.model = Model("NeuralNetwork")
        self.onnx_model_path = onnx_model_path

        # Use the simplified version by onnxsim for the given network
        if simplification:
            self.onnx_model_path = self.simplified_variant(self.onnx_model_path)

        # Ensure dynamic batch_dim and reexport as dynamic version
        self.onnx_model_path = self.add_dynamic_batch_dim(self.onnx_model_path, dynamic_name="batch_size")

        self.internal_onnx = ONNXParser(self.onnx_model_path)._parse_model()
        self.initializers = self.internal_onnx.initializers
        self.nodes = self.internal_onnx.nodes
        self.in_out_tensors_shapes = self.internal_onnx.in_out_tensors_shapes
        self.operator_factory = OperatorFactory()
        self.variables = {}

        # Parameters for optional compatibilty/equivalence check
        self.compcheck = compcheck
        self.rtol = rtol
        self.atol = atol


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
        Does perform a compatibility check afterward, if the flag was set to True.

        """
        self.create_variables()
        for node in self.nodes:
            if node['type'] != "Constant":
                operator = self.operator_factory.create_operator(node, self.initializers)
                operator.apply_constraints(self.model, self.variables)
        
        if self.compcheck:
            compatibility_check(
                self.onnx_model_path,
                iterative_analysis=False,
                output_dir=None,
                save_subgraphs=False,
                rtol=self.rtol,
                atol=self.atol
            )

    def get_gurobi_model(self):
        """
        Retrieves the Gurobi model after all constraints have been added.

        Returns:
            gurobipy.Model: The constructed Gurobi model reflecting the ONNX graph.
        """
        return self.model

    def get_input_vars(self):
        """
        Returns:
            dict: Dictionary with references to the GurobiModel input variables.
        """
        if len(self.internal_onnx.input_node_name) != 1:
            raise ValueError(f"The current model seems to have more than one input node, which isn't supported by this function.")
        input_name = self.internal_onnx.input_node_name[0]
        input_vars = self.variables.get(input_name)
        if input_vars is None:
            raise ValueError(f"Input variables couldn't be accessed.")
        return input_vars

    def get_output_vars(self):
        """
        Returns:
            dict: Dictionary with references to the GurobiModel output variables.
        """
        if len(self.internal_onnx.output_node_name) != 1:
            raise ValueError(f"The current model seems to have more than one output node, which isn't supported by this function.")
        output_name = self.internal_onnx.output_node_name[0]
        output_vars = self.variables.get(output_name)
        if output_vars is None:
            raise ValueError(f"Output variables couldn't be accessed.")
        return output_vars

    def simplified_variant(self, onnx_path):
        """
        Simplifies the given onnx model with onnxsim. If successful, the simplified model is saved and
        the path is returned. If simplification fails, a RunTimeError is raised.

        Returns:
            path (string): Path to the simplified onnx model.
        """
        base_model = onnx.load(onnx_path)
        print(f"\n ONNX input model has {len(base_model.graph.node)} nodes ({onnx_path}).")
        # With dynamic shapes
        model_simp, check = simplify(base_model)

        if check:
            path_to_simplified = onnx_path.removesuffix('.onnx') + "_simplified.onnx"
            print(f"ONNX model was simplified to {len(model_simp.graph.node)} nodes ({path_to_simplified}).")
            onnx.save(model_simp, path_to_simplified)
            return path_to_simplified
        else:
            raise RuntimeError(f"Simplification of {onnx_path} couldn't be validated.")

    def add_dynamic_batch_dim(self, onnx_path, dynamic_name="batch_size"):
        """
        Ensures the given onnx model has a dynamic batch dimension in the input and output shapes.
        But only if the input shape hase more than one dimension.

        Returns:
            path (string): Path to the onnx model with dynamic batch dimension.
        """
        base_model = onnx.load(onnx_path)
        # Ensuring the first dimension is a dynamic dimension "batch_size"
        graph = base_model.graph
        dynamic_change = False
        for input_tensor in graph.input:
            dims = input_tensor.type.tensor_type.shape.dim
            if len(dims) > 1:
                dynamic_change = True
                dims[0].dim_param = dynamic_name
        for output_tensor in graph.output:
            dims = output_tensor.type.tensor_type.shape.dim
            if len(dims) > 1:
                dynamic_change = True
                dims[0].dim_param = dynamic_name

        if dynamic_change:
            new_path = onnx_path.removesuffix('.onnx') + "_dynamic.onnx"
            onnx.save(base_model, new_path)
            print(f"Final ONNX model with dyn. batch dimension was stored ({new_path}).\n")
            return new_path
        else:
            print(f"ONNX model has only one input dimension: no dynamic batch dimension was added.\n")
            return onnx_path