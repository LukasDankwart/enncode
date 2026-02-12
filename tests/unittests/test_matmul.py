import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator

@pytest.mark.matmul
def test_matmul_operation():
    tensor = np.random.rand(1, 24).astype(np.float32)
    weights = np.random.rand(24, 12).astype(np.float32)

    onnx_path = "./onnxgurobi/tests/models/tens_matmul.onnx"

    input_names, output_names = onnx_generator.MatMulModule(weight_matrix=weights).onnx_export(
        tensor, onnx_path=onnx_path
    )

    tests_utils.compare_models(onnx_path, tensor)