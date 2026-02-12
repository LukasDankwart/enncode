import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator


@pytest.mark.flatten_operator
def test_flatten_tensor():
    tensor = np.random.rand(1, 784, 10).astype(np.float32)

    onnx_path = "./onnxgurobi/tests/models/tens_flatten.onnx"

    input_names, output_names = onnx_generator.FlattenModule().onnx_export(
        tensor, onnx_path=onnx_path
    )

    tests_utils.compare_models(onnx_path, tensor)
