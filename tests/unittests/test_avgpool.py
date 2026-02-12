import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator

@pytest.mark.avgpool
def test_avgpool():
    tensor = np.random.rand(1, 3, 24, 24).astype(np.float32)

    onnx_path = "./onnxgurobi/tests/models/tens_avgpool.onnx"

    input_names, output_names = onnx_generator.AvgPoolModule().onnx_export(
        tensor, onnx_path=onnx_path
    )

    tests_utils.compare_models(onnx_path, tensor)