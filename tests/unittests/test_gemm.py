import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator

@pytest.mark.gemm
def test_gemm_operation():
    tensor = np.random.rand(1, 24).astype(np.float32)
    weights = np.random.rand(24, 12).astype(np.float32)
    biases = np.random.rand(12).astype(np.float32)

    onnx_path = "./onnxgurobi/tests/models/tens_gemm.onnx"

    input_names, output_names = onnx_generator.GemmModule(weight_matrix=weights.T, biases=biases, alpha=1.0, beta=1.0).onnx_export(
        tensor, onnx_path=onnx_path
    )

    tests_utils.compare_models(onnx_path, tensor)