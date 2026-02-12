import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import shape_inference
import onnx

from tests import tests_utils

def export_model(model, dummy_input, filename):
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

class SplitDimModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_shape = [1, 2, 20, 1]

    def forward(self, x):
        return torch.reshape(x, self.target_shape)


@pytest.mark.reshape
def test_reshape_tensor():
    tensor = torch.from_numpy(np.random.rand(1, 2, 2, 10).astype(np.float32))
    model_split = SplitDimModule()
    onnx_path = "onnxgurobi/tests/models/tens_reshape.onnx"
    export_model(model_split, tensor, "onnxgurobi/tests/models/tens_reshape.onnx")
    tensor = np.random.rand(1, 2, 2, 10).astype(np.float32)
    model_onnx = onnx.load(onnx_path)
    model_onnx = shape_inference.infer_shapes(model_onnx)
    onnx.save(model_onnx, onnx_path)

    onnx_output, gurobi_output = tests_utils.compare_models(onnx_path, tensor, expand_batch_dim=True)
