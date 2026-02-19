from enncode.parsers.parser_factory import ParserFactory
from onnx import utils
from vnnlib.compat import read_vnnlib_simple
from gurobipy import GRB
from enncode import gurobiModelBuilder
import gzip
import onnxruntime as ort
import os.path
import sys
import os
import numpy as np
import onnx
import glob

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

def run_onnx_model(model_path, input_data, input_tensor_name='input', output_tensor_name='output'):
    """
    Runs an ONNX model with the given input and returns its first output.
    """
    session = ort.InferenceSession(model_path)
    onnx_outputs = session.run(None, {input_tensor_name: input_data})
    return onnx_outputs[0]


def solve_gurobi_model(model_path, input_data, input_tensor_name='input', output_tensor_name='output',
                       expand_batch_dim=False):
    """
    Converts an ONNX model to a Gurobi model, assigns input values, optimizes, and returns the output.
    """
    converter = gurobiModelBuilder.GurobiModelBuilder(model_path, simplification=False)
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

def get_unsupported_node_types(onnx_path):
    """
    Helper function to extract existing node types in onnx.file that are currently supported by GurobiModelBuilder.

    Args:
        onnx_path: Path to the (original) onnx file which is analyzed
    Returns:
        unsupported: List of nodes in given onnx file (path), that are not supported
    """
    dummy_factory = ParserFactory()
    model = onnx.load(onnx_path)
    graph = model.graph
    used_ops_in_model = set([node.op_type for node in graph.node])
    supported_ops = set(dummy_factory.parsers.keys())
    unsupported_ops = []
    for op in used_ops_in_model:
        if op not in supported_ops:
            unsupported_ops.append(op)

    return unsupported_ops


def get_nodes_of_graph(onnx_path):
    """
    Helper function for extracting node-names, primary for writing log file.

    Args:
        onnx_path: path to the current onnx model
    """
    subgraph = onnx.load(onnx_path)
    subgraph = subgraph.graph
    inputs = [inp.name for inp in subgraph.input]
    outputs = [out.name for out in subgraph.output]
    nodes = []
    for i, node in enumerate(subgraph.node):
        name = node.name if node.name else f"Unnamed_{node.op_type}_{i}"
        nodes.append(name)
    return inputs, outputs, nodes


def extract_subgraph(onnx_path, subgraph_filename, model_input_names, target_output_names, log_file_path):
    """
    Helper function for extraction subgraphs and handling possible exceptions.

    Args:
        onnx_path: path to the current onnx model
        subgraph_filename: name/path of the new subgraph, stored at given subgraph_filename
        model_input_names: input names of the subgraph to be extracted (determined by caller method)
        target_output_names: output names of the subgraph to be extracted (determined by caller method)
        log_file_path: path to the log file of currently analyzed (base) onnx file
    """
    ind = " " * 5
    try:
        utils.extract_model(
            onnx_path,
            subgraph_filename,
            input_names=model_input_names,
            output_names=target_output_names
        )
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{ind}[PASSED]: Extracting current subgraph was successful, stored at {subgraph_filename}. \n")

    except Exception as e:
        print(f"Something went wrong extracting subgraph {subgraph_filename}. \n")
        print(e)
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{ind}[FAILED]: Current subgraph couldn't be extracted {subgraph_filename}. \n")
            f.write(f"{ind}{ind}{e}")
        sys.exit(1)


def add_dynamic_batch_dim(onnx_model, dynamic_name="batch_size"):
    """
    Expects onnx_model to be a loaded onnx model, where a dynamic input/output dimension for batch size is declared at
    the first dimension or is otherwise added.

    Args:
        onnx_model: onnx model where dynamic batch dim has to be determined (inplace), given by caller method.
        dynamic_name: name of the new dynamic axis
    """
    graph = onnx_model.graph
    for input_tensor in graph.input:
        dims = input_tensor.type.tensor_type.shape.dim
        if len(dims) > 1:
            dims[0].dim_param = dynamic_name
    for output_tensor in graph.output:
        dims = output_tensor.type.tensor_type.shape.dim
        if len(dims) > 1:
            dims[0].dim_param = dynamic_name


def check_equivalence(onnx_path, model_input, model_input_names, target_output_names, log_file_path, rtol=1e-05, atol=1e-08):
    """
    Runs inference on onnx model and checks compatibility with GurobiModelBuilder.

    Args:
        onnx_path: path to the current onnx model
        model_input: input from caller method for compatibility check
        model_input_names: name of model input
        target_output_names: determined output names from caller method
        log_file_path: path to the log file of currently analyzed (base) onnx file

    Returns:
        True: if model is compatible and shows equivalence for given random input
        False: if model is compatible but didn't show equivalence for given random input
        None: if model isn't compatible or gurobi output has different shape as onnx output for given random input
    """
    if not os.path.isfile(log_file_path):
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"Log-file from analysing of {onnx_path}. \n \n")

    onnx_output = run_onnx_model(onnx_path, model_input, input_tensor_name=model_input_names[0])[0]
    try:
        gurobi_output = solve_gurobi_model(
            onnx_path,
            model_input,
            input_tensor_name=model_input_names[0],
            output_tensor_name=target_output_names[0],
            expand_batch_dim=True
        )
        if onnx_output.shape != gurobi_output.shape:
            # In that case, the shape mismatch might be caused by expanding batch dimension.
            if len(gurobi_output.shape) - 1 == len(onnx_output.shape) and gurobi_output.shape[0] == 1:
                gurobi_output = gurobi_output[0]
            else:
                raise ValueError(f"Shape mismatch: ONNX {onnx_output.shape} vs Gurobi {gurobi_output.shape}")

        equivalence = np.allclose(onnx_output, gurobi_output, rtol=rtol, atol=atol)

        return equivalence

    except NotImplementedError as e:
        print("\n Current subgraph has unsupported node types.\n")
        print("GurobiModelBuilder misses support for following node type:")
        print(e)
        ind=" " * 5
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{ind}[FAILED]: Compatibility is missing. Subgraph has unsupported node type! \n")
            f.write(f"{ind}{ind}{e} \n")
            return None
    except ValueError as v:
        ind = " " * 5
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{ind}[PASSED]: Compatibility. \n")
            f.write(f"{ind}[FAILED]: Equivalence check results in different shapes for ONNX and Gurobi output. \n")
            f.write(f"{ind}{ind} {v} \n")
        sys.exit(1)


def iterative_analyze_subgraphs(onnx_path, output_dir, model_input, model_input_names, rtol, atol, save_subgraphs=True):
    """
    This method is used for analyzing GurobiModelBuilder parsing errors. It iteratively extracts subgraphs from given onnx
    model and tries to identify nodes, responsible for misconduct while parsing.

    Args:
        onnx_path: path to the initial onnx model
        model_input: input from caller method for compatibility check
        model_input_names: name of model input
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    # Checks for validity of output dir, otherwise, directory of onnx path is taken as output dir
    if not os.path.isdir(output_dir):
        output_dir = os.path.dirname(onnx_path)
    output_dir += "subgraphs"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Create log file for clearer overview
    log_file_path = output_dir + "/subgraphs_log.txt"
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Log-file from analysing of {onnx_path}. \n \n")

    for i, node in enumerate(graph.node):
        # For constants nodes, no subgraph is evaluated
        if node.op_type == "Constant":
            continue

        node_name = node.name if node.name else f"Node_{i}_{node.op_type}"
        target_output_names = list(node.output)
        subgraph_filename = os.path.join(output_dir, f"node_{i:03d}_{node_name.replace('/', '_')}.onnx")

        ind = " " * 5
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[NODE {i}]: {node_name}\n")

        # If subgraphs should not be stored permanently while analysing, they are removed in next iteration
        if not save_subgraphs:
            for file in glob.glob(os.path.join(output_dir, "*.onnx")):
                os.remove(file)

        extract_subgraph(onnx_path, subgraph_filename, model_input_names, target_output_names, log_file_path)

        equivalence = check_equivalence(
            subgraph_filename, model_input, model_input_names, target_output_names, log_file_path, rtol, atol
        )
        # Update log file
        with open(log_file_path, "a") as f:
            subgraph_inputs, subgraph_outputs, subgraph_nodes = get_nodes_of_graph(subgraph_filename)

            if equivalence:
                f.write(f"{ind}[PASSED]: Compatibility. \n")
                f.write(f"{ind}[PASSED]: Equivalence check for ({subgraph_filename}) has been successful. \n \n")
            else:
                # In that case, check_equivalence must have failed, caused by unsupported nodes
                if equivalence is None:
                    unsupported_types = get_unsupported_node_types(onnx_path)
                    f.write("\n Note: - Furthermore, the original model contains "
                            "following unsupported operation types, which are likely to cause further incompatibility: \n")
                    f.write(str(unsupported_types))
                    f.write("\n (Please see documentation for currently supported node-operations)")
                # If equivalence is neither true nor none, it is false, indicating compatibility but missing equivalence
                else:
                    f.write(f"{ind}[PASSED]: Compatibility. \n")
                    f.write(f"{ind}[FAILED]: Equivalence check for ({subgraph_filename}) failed. \n")
                    f.write(f"{ind} Inputs: {subgraph_inputs} \n")
                    f.write(f"{ind} Outputs: {subgraph_outputs} \n")
                    f.write(f"{ind} Included nodes: {subgraph_nodes} \n \n")

                sys.exit(1)


def compatibility_check(onnx_path, iterative_analysis=True, output_dir=None, save_subgraphs=True, rtol=1e-03, atol=1e-05):
    """
    This method implements an automated compatibility check for given onnx file. First, it is adjusted to have dynamic
    batch dimension and is stored as a new onnx file. For the adjusted onnx file is compatibility checked with a random
    input. If successful, equivalence to the corresponding onnx run is check with given rtol/atol deviation.

    If not successful, the user is asked if an iterative analysis of subgraphs should be done. If so, every subgraph
    in topological order is evaluated to be extracted and then tested for compatibility and equivalence with onnx run.

    Results of iterative analysis are written in a log file, stored in the specified output directory.

    Args:
        onnx_path: path to the onnx file to be checked.
        iterative_analysis: boolean flag, if iterative analysis of subgraphs should be done
        output_dir: path to the directory where log files and subgraphs are stored.
        save_subgraphs: boolean flag, if all subgraphs should be stored permanently or removed if checked
        rtol: the relative tolerance parameter for np.allclose
        atol: the absolute tolerance parameter for np.allclose
    """
    path = onnx_path

    # 1) First the suffix of given path is checked for the type of onnx format
    if path.endswith(".gz"):
        model = load_gzip_onnx_model(path)
        path = path.removesuffix(".gz")
        onnx.save(model, path)
    onnx_model = onnx.load(path)

    # Then we ensure that first dimension is always a dynamic batch dimension
    add_dynamic_batch_dim(onnx_model)
    # onnx_model = shape_inference.infer_shapes(onnx_model)
    path = path.removesuffix(".onnx") + "_modified.onnx"
    onnx.save(onnx_model, path)

    # 2) For given network, input and output names has to be filtered
    graph = onnx_model.graph
    input_names = [node.name for node in graph.input]
    initializer_names = {x.name for x in graph.initializer}
    real_inputs = [name for name in input_names if name not in initializer_names]

    input_nodes = [node for node in onnx_model.graph.input if node.name in real_inputs]
    input_tensor = input_nodes[0]
    input_shape = [max(dim.dim_value, 1) for dim in input_tensor.type.tensor_type.shape.dim]
    output_names = [node.name for node in onnx_model.graph.output]

    # 3) Basic compatibility check by random input
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    onnx_output = run_onnx_model(path, dummy_input, input_tensor_name=real_inputs[0])[0]
    try:
        gurobi_output = solve_gurobi_model(
            path,
            dummy_input,
            input_tensor_name=real_inputs[0],
            output_tensor_name=output_names[0],
            expand_batch_dim=True
        )
    except NotImplementedError as e:
        print(f"\n An error has occurred by solving gurobi model for given onnx file.")
        if iterative_analysis:
            if output_dir is None:
                output_dir = os.path.dirname(onnx_path) + "/"
            # The subgraphs are iteratively checked for compatibility
            iterative_analyze_subgraphs(path, output_dir, dummy_input, real_inputs, rtol, atol, save_subgraphs)
        else:
            print("\n No iterative check for compatibility is performed. \n")
            print("GurobiModelBuilder misses support for following node types:")
            unsupported_types = get_unsupported_node_types(path)
            print(str(unsupported_types))
        sys.exit(1)

    # 4) Check for equivalent outputs
    rtol = rtol
    atol = atol
    equivalence = np.allclose(onnx_output, gurobi_output, rtol=rtol, atol=atol)
    if equivalence:
        print("\n Given network has been compatible with GurobiModelBuilder parsing. \n")
        print(f"It has shown equivalence for rtol={rtol} and atol={atol}.")
        print(f"ONNX-output: {onnx_output}")
        print(f"Gurobi-output: {gurobi_output}")
    else:
        print("\n Given network has been compatible with GurobiModelBuilder parsing. \n")
        print(f"Unfortunately there is a deviation between ONNX and Gurobi output for rtol={rtol} and atol={atol}.")
        print(f"ONNX-output: {onnx_output}")
        print(f"Gurobi-output: {gurobi_output}")
        sys.exit(1)