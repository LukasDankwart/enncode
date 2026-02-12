import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator

@pytest.mark.batchnorm
def test_batchnormalization():
    tensor = np.random.rand(1, 3, 24, 24).astype(np.float32)
    num_features = tensor.shape[1]
    eps = 1e-05
    momentum = 0.1

    onnx_path = "./onnxgurobi/tests/models/tens_batchnorm.onnx"

    input_names, output_names = onnx_generator.BatchNorm2dModule(num_features, eps, momentum).onnx_export(
        tensor, onnx_path=onnx_path
    )

    tests_utils.compare_models(onnx_path, tensor)