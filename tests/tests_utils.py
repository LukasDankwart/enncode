import numpy as np
import onnxruntime as ort
import gzip
import onnx
from gurobipy import GRB
from vnnlib.compat import read_vnnlib_simple

from onnx_to_gurobi.gurobiModelBuilder import GurobiModelBuilder

def run_onnx_model(model_path, input_data, input_tensor_name='input', output_tensor_name='output'):
    """
    Runs an ONNX model with the given input and returns its first output.
    """
    session = ort.InferenceSession(model_path)
    onnx_outputs = session.run(None, {input_tensor_name: input_data})
    return onnx_outputs[0]

def solve_gurobi_model(model_path, input_data, input_tensor_name='input', output_tensor_name='output', expand_batch_dim=False):
    """
    Converts an ONNX model to a Gurobi model, assigns input values, optimizes, and returns the output.
    """
    converter = GurobiModelBuilder(model_path)
    converter.build_model()

    dummy_input = input_data
    input_shape = dummy_input.shape

    # Expand with batch dimension if not included in onnx-file
    if expand_batch_dim:
        dummy_input = np.expand_dims(dummy_input, axis=0)
        input_shape = dummy_input.shape

    # Set dummy input values in the Gurobi model.
    input_vars = converter.variables.get(input_tensor_name)
    if input_vars is None:
        raise ValueError(f"No input variables found for '{input_tensor_name}'.")
    
    for idx, var in input_vars.items():
        if isinstance(idx, int):
            md_idx = np.unravel_index(idx, input_shape[1:])  # Exclude batch dimension
        elif isinstance(idx, tuple):
            if len(idx) < len(input_shape) - 1:
                idx = (0,) * (len(input_shape) - 1 - len(idx)) + idx
            md_idx = idx
        else:
            raise ValueError(f"Unexpected index type: {type(idx)}")
        value = float(dummy_input[0, *md_idx])
        var.lb = value
        var.ub = value

    gurobi_model = converter.get_gurobi_model()
    gurobi_model.optimize()
    if gurobi_model.status != GRB.OPTIMAL:
        raise ValueError(f"Optimization ended with status {gurobi_model.status}.")

    # Extract the output from the Gurobi model.
    output_vars = converter.variables.get(output_tensor_name)
    if output_vars is None:
        raise ValueError(f"No output variables found for '{output_tensor_name}'.")
    output_shape = converter.in_out_tensors_shapes[output_tensor_name]
    gurobi_outputs = np.zeros([1] + output_shape, dtype=np.float32)

    output_shape = np.empty(output_shape).shape
    flat_vars = [output_vars[k] for k in sorted(output_vars.keys())]
    vars_array = np.array(flat_vars, dtype=object).reshape(output_shape)
    for idx in np.ndindex(vars_array.shape):
        if isinstance(idx, int):
            md_idx = np.unravel_index(idx, output_shape)
        elif isinstance(idx, tuple):
            md_idx = idx
        else:
            raise ValueError(f"Unexpected index type in output: {type(idx)}")
        gurobi_outputs[(0,) + md_idx] = vars_array[idx].x

    # If an artificial batch dim. was added (cause not included in onnx file), it has to be removed
    if expand_batch_dim:
        if gurobi_outputs.shape[0] != 1:
            raise ValueError(f"Something went wrong handling the batch-dimension expansion.")
        gurobi_outputs = np.reshape(gurobi_outputs, gurobi_outputs.shape[1:])

    return gurobi_outputs

def compare_models(model_path, input_data, input_tensor_name='input', output_tensor_name='output', atol=0.02, expand_batch_dim=False):
    """
    Runs both the ONNX model and the Gurobi model, then asserts that their outputs are close.
    """
    onnx_output = run_onnx_model(model_path, input_data, input_tensor_name, output_tensor_name)
    gurobi_output = solve_gurobi_model(model_path, input_data, input_tensor_name, output_tensor_name, expand_batch_dim=expand_batch_dim)
    
    if onnx_output.shape != gurobi_output.shape:
        raise ValueError(f"Shape mismatch: ONNX {onnx_output.shape} vs Gurobi {gurobi_output.shape}")

    np.testing.assert_allclose(onnx_output, gurobi_output, atol=atol)

    return onnx_output, gurobi_output


def run_onnx_model_two_inputs(model_path, tens1_data, tens2_data):
    """
    Runs an ONNX model when two dynamic inputs are used (in case for some unit tests).
    """
    session = ort.InferenceSession(model_path)

    input_feed = {
        'input_0': tens1_data.astype(np.float32),
        'input_1': tens2_data.astype(np.float32)
    }

    onnx_outputs = session.run(None, input_feed)
    return onnx_outputs[0]


def assign_input_values(converter, input_data, input_tensor_name):
    """
    Helper function for the solving gurobi model with two dynamic inputs version.
    """
    dummy_input = input_data
    input_shape = dummy_input.shape

    input_vars = converter.variables.get(input_tensor_name)
    if input_vars is None:
        raise ValueError(f"No input variables found for '{input_tensor_name}'.")

    for idx, var in input_vars.items():
        if isinstance(idx, int):
            md_idx = np.unravel_index(idx, input_shape[1:])
        elif isinstance(idx, tuple):
            if len(idx) < len(input_shape) - 1:
                idx = (0,) * (len(input_shape) - 1 - len(idx)) + idx
            md_idx = idx
        else:
            raise ValueError(f"Unexpected index type: {type(idx)}")

        value = float(dummy_input[0, *md_idx])
        var.lb = value
        var.ub = value


def solve_gurobi_model_two_inputs(model_path, data, input_data_names=['input_0','input_1'], output_tensor_name='output'):
    converter = GurobiModelBuilder(model_path)
    converter.build_model()

    # Assigning variables for each input tensor
    assign_input_values(converter, data[input_data_names[0]], input_data_names[0])
    assign_input_values(converter, data[input_data_names[1]], input_data_names[1])

    gurobi_model = converter.get_gurobi_model()
    gurobi_model.optimize()
    if gurobi_model.status != GRB.OPTIMAL:
        raise ValueError(f"Optimization ended with status {gurobi_model.status}.")

    output_vars = converter.variables.get(output_tensor_name)
    if output_vars is None:
        raise ValueError(f"No output variables found for '{output_tensor_name}'.")

    output_shape = converter.in_out_tensors_shapes[output_tensor_name]
    gurobi_outputs = np.zeros([1] + output_shape, dtype=np.float32)

    for idx, var in output_vars.items():
        if isinstance(idx, int):
            md_idx = np.unravel_index(idx, output_shape)
        elif isinstance(idx, tuple):
            md_idx = idx
        else:
            raise ValueError(f"Unexpected index type in output: {type(idx)}")
        gurobi_outputs[(0,) + md_idx] = var.x

    return gurobi_outputs


def compare_models_two_inputs(model_path, input_data, input_tensor_names=['input_0','input_1'], output_tensor_name='output', atol=0.02):
    """
    Runs both the ONNX model and the Gurobi model, then asserts that their outputs are close.

    Expects input_data to be a dictionary with corresponding keys in input_tensor_names
    """

    input_0 = input_data[input_tensor_names[0]]
    input_1 = input_data[input_tensor_names[1]]

    onnx_output = run_onnx_model_two_inputs(model_path, input_0, input_1)
    gurobi_output = solve_gurobi_model_two_inputs(model_path, input_data, input_tensor_names, output_tensor_name)

    if onnx_output.shape != gurobi_output.shape:
        raise ValueError(f"Shape mismatch: ONNX {onnx_output.shape} vs Gurobi {gurobi_output.shape}")

    np.testing.assert_allclose(onnx_output, gurobi_output, atol=atol)


def load_gzip_onnx_model(gzipped_path):
    """
    Loads a specified model in onnx.gz format and return its decompressed 'standard' onnx model.

    Args:
        gzipped_path: path to the specified compressed model *.onnx.gz

    Returns:
        model: returns the loaded onnx model

    Raises:
        ValueError: if none onnx model could be decompressed from given path
    """
    print(f"\n Loading compressed model from: {gzipped_path} ")

    with gzip.open(gzipped_path, 'rb') as f:
        model = onnx.load(f)

    if model is None:
        raise ValueError(f"Decompressing model from given path {gzipped_path} couldn't be done successfully.")
    print(f"Loading compressed model from {gzipped_path} was successful!")

    return model


def load_vnnlib_conditions(vnnlib_path, onnx_model):
    """
    Initializes one parser for a *.vnnlib.gz by with given path. The parser is a dictionary which stores defined
    preconditions at parser[0][0] and postconditions at parser[0][1], usually stored at the named indices.
    The onnx model is necessary for determining the number of input and outputs.

    Args:
        vnnlib_path: path to the specified compressed model *.vnnlib.gz specifications
        onnx_model: the onnx model which the specs should be analyzed

    Returns:
        model: returns a vnnlib parser initialized with given path

    Raises:
        ValueError (or internal vnnlib error): If loading the spec parser object failed cause spec/model mismatch or path issues.
    """
    graph = onnx_model.graph
    input = graph.input
    input_tensor = input[0]
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    num_inputs = int(np.prod(input_shape))
    print(f"Registered number of inputs from given onnx model is: {num_inputs}")

    output = graph.output
    output_tensor = output[0]
    output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
    num_outputs = int(np.prod(output_shape))
    print(f"Registered number of outputs from given onnx model is: {num_outputs}")

    spec_parser = read_vnnlib_simple(vnnlib_path, num_inputs, num_outputs)
    if spec_parser is None:
        raise ValueError(
            f"Loading specs from {vnnlib_path} of vnnlib.data couldn't be done successfully.")
    print(f"Loading specifications from {vnnlib_path} was successful! \n")

    return spec_parser