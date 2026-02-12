import numpy as np
import onnx
import pytest
from tests.tests_utils import compare_models
from tests.tests_utils import load_gzip_onnx_model, load_vnnlib_conditions

@pytest.mark.integration
def test_conv1():
    """Tests a convolutional neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/conv1.onnx"
    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_conv2():
    """Tests a convolutional neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/conv2.onnx"
    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_conv3():
    """Tests a convolutional neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/conv3.onnx"
    input_data = np.random.randn(1, 1, 10, 10).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_fc1():
    """Tests a fully connected neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/fc1.onnx"
    input_data = np.random.randn(1, 28, 28).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_fc2():
    """Tests a fully connected neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/fc2.onnx"
    input_data = np.random.randn(1, 784).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_fc3():
    """Tests a fully connected neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/fc3.onnx"
    input_data = np.random.randn(1, 784).astype(np.float32)
    compare_models(model_path, input_data)

"""
    Integration tests for the concrete-version of networks (networks/concrete)
"""
@pytest.mark.integration_concrete
def test_concrete_backward():
    model_path = "./onnxgurobi/tests/networks/concrete/backward_processed.onnx"
    input_data = np.random.rand(9).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='x.1', output_tensor_name='54', expand_batch_dim=True)

@pytest.mark.integration_concrete
def test_concrete_classifier_medium():
    model_path = "./onnxgurobi/tests/networks/concrete/classifier_medium.onnx"
    input_data = np.random.rand(8).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='11', expand_batch_dim=True)

@pytest.mark.integration_concrete
def test_concrete_tiny():
    model_path = "./onnxgurobi/tests/networks/concrete/classifier_tiny.onnx"
    input_data = np.random.rand(8).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='5', expand_batch_dim=True)

@pytest.mark.integration_concrete
def test_concrete_flow():
    model_path = "./onnxgurobi/tests/networks/concrete/flow.onnx"
    input_data = np.random.rand(9).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='54', expand_batch_dim=True)

"""
    Integration tests for the diabetes-version of networks (networks/diabetes)
"""
@pytest.mark.integration_diabetes
def test_diabetes_backward():
    model_path = "./onnxgurobi/tests/networks/diabetes/backward_processed.onnx"
    input_data = np.random.rand(9).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='x.1', output_tensor_name='34', expand_batch_dim=True)

@pytest.mark.integration_diabetes
def test_diabetes_classifier_medium():
    model_path = "./onnxgurobi/tests/networks/diabetes/classifier_medium.onnx"
    input_data = np.random.rand(8).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='11', expand_batch_dim=True)

@pytest.mark.integration_diabetes
def test_diabetes_classifier_tiny():
    model_path = "./onnxgurobi/tests/networks/diabetes/classifier_tiny.onnx"
    input_data = np.random.rand(8).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='5', expand_batch_dim=True)

@pytest.mark.integration_diabetes
def test_diabetes_flow():
    model_path = "./onnxgurobi/tests/networks/diabetes/flow.onnx"
    input_data = np.random.rand(9).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='34', expand_batch_dim=True)

"""
    Integration tests for the power-version of networks (networks/power)
"""
@pytest.mark.integration_power
def test_power_classifier_medium():
    model_path = "./onnxgurobi/tests/networks/power/classifier_medium.onnx"
    input_data = np.random.rand(4).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='11', expand_batch_dim=True)

@pytest.mark.integration_power
def test_power_classifier_tiny():
    model_path = "./onnxgurobi/tests/networks/power/classifier_tiny.onnx"
    input_data = np.random.rand(4).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='5', expand_batch_dim=True)

@pytest.mark.integration_power
def test_power_flow():
    model_path = "./onnxgurobi/tests/networks/power/flow.onnx"
    input_data = np.random.rand(5).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='66', expand_batch_dim=True)

"""
    Integration tests for the wine-version of networks (networks/wine)
"""
@pytest.mark.integration_wine
def test_wine_classifier_backward():
    model_path = "./onnxgurobi/tests/networks/wine/backward_processed.onnx"
    input_data = np.random.rand(12).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='x.1', output_tensor_name='34', expand_batch_dim=True)

@pytest.mark.integration_wine
def test_wine_classifier_medium():
    model_path = "./onnxgurobi/tests/networks/wine/classifier_medium.onnx"
    input_data = np.random.rand(11).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='11', expand_batch_dim=True)

@pytest.mark.integration_wine
def test_wine_classifier_tiny():
    model_path = "./onnxgurobi/tests/networks/wine/classifier_tiny.onnx"
    input_data = np.random.rand(11).astype(np.float32)
    compare_models(model_path, input_data, input_tensor_name='onnx::MatMul_0', output_tensor_name='5', expand_batch_dim=True)


"""
    Basic integration tests for some networks from the vnn-comp2023 benchmark 
    
    Since most of the networks are compressed as *.onnx.gz files, every test loads its .gz file, decompresses it
    and stores it under the same path as standard .onnx file. This is done for simple reusing existing compare methods.
"""
@pytest.mark.integration_acasxu
def test_acasxu():
    # Load and store decompressed onnx file
    model_path = "./onnxgurobi/tests/vnncomp/acasxu/ACASXU_run2a_1_1_batch_2000.onnx.gz"
    model = load_gzip_onnx_model(model_path)
    onnx.save(model, model_path.removesuffix(".gz"))

    # Derive model input name/shape and output name
    graph = model.graph
    input_node = graph.input
    input_tensor = input_node[0]
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    input_names = [node.name for node in model.graph.input]
    initializer_names = {x.name for x in model.graph.initializer}
    real_inputs = [name for name in input_names if name not in initializer_names]
    output_names = [node.name for node in model.graph.output]

    rand_input = np.random.rand(*input_shape).astype(np.float32)
    onnx_output, gurobi_output = compare_models(model_path.removesuffix(".gz"), rand_input, input_tensor_name=real_inputs[0], output_tensor_name=output_names[0], expand_batch_dim=True)
    print(onnx_output)
    print(gurobi_output)


@pytest.mark.integration_subgraph_yolo
def test_subgraph_yolo():
    model_path = "./onnxgurobi/tests/vnncomp/yolo/TinyYOLO_modified.onnx"
    model = onnx.load_model(model_path)
    graph = model.graph
    input_node = graph.input
    input_tensor = input_node[0]
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    input_names = [node.name for node in model.graph.input]
    initializer_names = {x.name for x in model.graph.initializer}
    real_inputs = [name for name in input_names if name not in initializer_names]
    output_names = [node.name for node in model.graph.output]

    rand_input = np.random.rand(*input_shape).astype(np.float32)
    onnx_output, gurobi_output = compare_models(model_path, rand_input,
                                                input_tensor_name=real_inputs[0], output_tensor_name=output_names[0],
                                                expand_batch_dim=True)