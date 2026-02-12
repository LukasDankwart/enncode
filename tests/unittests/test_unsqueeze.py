import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator

@pytest.mark.unsqueeze
def test_unsqueeze_tensor():
    tensor = np.random.rand(1, 3, 5, 24, 24).astype(np.float32)
    dim = -1

    onnx_path = "./onnxgurobi/tests/models/tens_unsqueeze.onnx"

    input_names, output_names = onnx_generator.UnsqueezeModule(dim=dim).onnx_export(
        tensor, onnx_path=onnx_path
    )

    onnx_output, gurobi_output = tests_utils.compare_models(onnx_path, tensor)

    print(onnx_output.shape)
    print(gurobi_output.shape)