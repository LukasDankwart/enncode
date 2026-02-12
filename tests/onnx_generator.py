import torch
import torch.nn as nn
import numpy as np


class FlattenModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, tens):
        return self.flatten(tens)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names


class ReluModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, tens):
        return self.relu(tens)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names


class IdentityModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, tens):
        return self.identity(tens)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names


class MaxPoolModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d((3,3), stride=1)

    def forward(self, tens):
        return self.pool(tens)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names

class AvgPoolModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d((3,3), stride=1)

    def forward(self, tens):
        return self.pool(tens)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names

class MatMulModule(nn.Module):

    def __init__(self, weight_matrix):
        super().__init__()
        weight_tensor = torch.from_numpy(weight_matrix).float()
        self.register_buffer('weights', weight_tensor)

    def forward(self, tens):
        return torch.matmul(tens, self.weights)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names



class AddModule(nn.Module):

    def __init__(self, biases):
        super().__init__()
        self.biases = biases

    def forward(self, tens):
        return torch.add(tens, self.biases)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names


class ConcatModule(nn.Module):

    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, tens1, tens2):
        return torch.concatenate((tens1, tens2), dim=self.axis)

    def onnx_export(self, tens1, tens2, onnx_path):
        input_names = ['input_0', 'input_1']
        output_names = ['output']

        if isinstance(tens1, np.ndarray):
            tens1 = torch.from_numpy(tens1)

        if isinstance(tens2, np.ndarray):
            tens2 = torch.from_numpy(tens2)

        torch.onnx.export(
            self,
            (tens1, tens2),
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input_0': {0: 'batch_size'},
                          'input_1': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names


class GemmModule(nn.Module):

    def __init__(self, weight_matrix, biases, alpha=1.0, beta=1.0):
        super().__init__()
        weight_tensor = torch.from_numpy(weight_matrix * alpha).float()
        bias_tensor = torch.from_numpy(biases * beta).float()

        in_features, out_features = weight_tensor.shape
        self.linear = nn.Linear(in_features, out_features)

        self.linear.weight = nn.Parameter(weight_tensor)
        self.linear.bias = nn.Parameter(bias_tensor)

    def forward(self, x):
        return self.linear(x)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names


class UnsqueezeModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tens):
        return torch.unsqueeze(tens, self.dim)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names


class ReshapeModule(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tens):
        return torch.reshape(tens, self.target_shape)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names


class Conv2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        #self.register_buffer('in_channels', in_channels)
        #self.register_buffer('out_channels', out_channels)
        #self.register_buffer('kernel_size', kernel_size)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, tens):
        return self.conv(tens)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names

class BatchNorm2dModule(nn.Module):

    def __init__(self, num_features, eps, momentum):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(num_features, eps, momentum)

    def forward(self, batch):
        return self.batchnorm(batch)

    def onnx_export(self, tens, onnx_path):
        input_names = ['input']
        output_names = ['output']

        if isinstance(tens, np.ndarray):
            tens = torch.from_numpy(tens)

        torch.onnx.export(
            self,
            tens,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        return input_names, output_names


class SoftmaxNet(nn.Module):
    # define layers of neural network
    def __init__(self, hidden_size=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, (4, 4), (2, 2), 0)
        self.conv2 = nn.Conv2d(2, 2, (4, 4), (2, 2), 0)
        self.hidden1 = nn.Linear(5 * 5 * 2, hidden_size)
        self.output = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # define forward pass of neural network
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.hidden1(x.view((-1, 5 * 5 * 2)))
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

def export_onnx(model, path, flatten=False):
    model.eval()

    rand_input = torch.rand(1, 784) if flatten else torch.randn(1, 1, 28, 28)
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        rand_input,
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )


if __name__ == "__main__":
    model = SoftmaxNet()
    export_onnx(model, r"./conv_softmax.onnx")