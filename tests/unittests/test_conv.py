import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator

@pytest.mark.conv
def test_conv():
    tensor = np.random.rand(1, 3, 24, 24).astype(np.float32)
    in_channels = tensor.shape[1]
    out_channels = 5
    kernel_size = (3, 3)

    onnx_path = "./onnxgurobi/tests/models/tens_conv.onnx"

    input_names, output_names = onnx_generator.Conv2dModule(in_channels, out_channels, kernel_size).onnx_export(
        tensor, onnx_path=onnx_path
    )

    tests_utils.compare_models(onnx_path, tensor)

