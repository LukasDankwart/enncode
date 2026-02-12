import numpy as np
import pytest
from tests import tests_utils
from tests import onnx_generator

@pytest.mark.concat
def test_concat():
    tensor1 = np.random.rand(1, 3, 24, 24).astype(np.float32)
    tensor2 = np.random.rand(1, 3, 24, 24).astype(np.float32)
    axis = 1

    data = {
        'input_0': tensor1,
        'input_1': tensor2
    }

    onnx_path = "./onnxgurobi/tests/models/tens_concat.onnx"

    input_names, output_names = onnx_generator.ConcatModule(axis=axis).onnx_export(
        tensor1, tensor2, onnx_path=onnx_path
    )

    tests_utils.compare_models_two_inputs(onnx_path, data)

