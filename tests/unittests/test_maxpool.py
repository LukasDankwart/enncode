import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator

@pytest.mark.maxpool
def test_maxpool():
    tensor = np.random.rand(1, 3, 24, 24).astype(np.float32)

    onnx_path = "./onnxgurobi/tests/models/tens_maxpool.onnx"

    input_names, output_names = onnx_generator.MaxPoolModule().onnx_export(
        tensor, onnx_path=onnx_path
    )

    tests_utils.compare_models(onnx_path, tensor)