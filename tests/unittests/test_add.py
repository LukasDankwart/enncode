import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator
import onnxruntime as ort
import torch

@pytest.mark.add_tensors
def test_add_tensors():
    tensor = np.random.rand(1, 56).astype(np.float32)
    weights = np.random.rand(1, 56).astype(np.float32)

    onnx_path = "./onnxgurobi/tests/models/tens_add.onnx"

    input_names, output_names = onnx_generator.AddModule(biases=weights).onnx_export(
        tensor, onnx_path=onnx_path
    )

    tests_utils.compare_models(onnx_path, tensor)

