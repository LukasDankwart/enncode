import torch
import torch.nn as nn
import numpy as np
from enncode.gurobiModelBuilder import GurobiModelBuilder
from enncode.compatibility import get_unsupported_node_types


class SimpleFCModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, output_activation, onnx_path = "simple_fc_model.onnx"):
        """
        This class is used to generate a pytorch model, consisting of fc linear + relu layers.
        It is designed to be compatible and usable with the ONNXGurobi bib.

        Args:
            input_dim: list or integer, representing the input dimension / shape
            hidden_dim: list of dimensions for each hidden layer with target shape
            output_dim: list or integer, representing the output dimension / shape
            output_activation: nn.* activation function for the output layer
            onnx_path: path for the onnx exports
        """
        super().__init__()

        # Input dimension is ensured to be a list, so only idx -1 is relevant to connect with successor linear layer
        if isinstance(input_dim, list):
            self.input_dim = input_dim
        elif isinstance(input_dim, int):
            self.input_dim = [input_dim]
        else:
            raise RuntimeError(f"Input-dim must be neither a list of dimensions or a single int, but {type(input_dim)} was given.")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_activation = output_activation if not output_activation is None else nn.ReLU()
        self.onnx_path = onnx_path

        # Create model
        self.model = self.create_model()
        try:
            self.gurobi_model_builder = self.update_gurobi_model_builder()
        except NotImplementedError as e:
            print("\n Current module contains an unsupported node type or activation function.")
            _ = get_unsupported_node_types(self.onnx_path)


    def create_model(self):
        """
        Called by constructor when the model is created.

        Returns: nn.Sequential module consisting of fc linear + relu layers
        """
        layers = []

        if not self.hidden_dim is None:
            # Connect input to first layer
            layers.append(nn.Linear(self.input_dim[-1], self.hidden_dim[0]))
            layers.append(nn.ReLU())
            # Connect each layer to its successor
            for layer_idx in range(0, len(self.hidden_dim) - 1):
                layers.append(nn.Linear(self.hidden_dim[layer_idx], self.hidden_dim[layer_idx+1]))
                layers.append(nn.ReLU())
            # Finally connect last layer with output layer
            layers.append(nn.Linear(self.hidden_dim[-1], self.output_dim))
        else:
            layers.append(nn.Linear(self.input_dim[-1], self.output_dim))

        # Add given output activation function. If not specified, output has sigmoid activation
        layers.append(self.output_activation)

        return nn.Sequential(*layers)


    def forward(self, x) -> torch.Tensor:
        """
        Returns: output of neural network for input tensor x
        """
        return self.model(x)


    def export_onnx(self):
        """
        This method is used for exporting the current state of the model in onnx format. The model is exported to
        the onnx path, specified as a constructor parameter of the model instance.

        Since ONNXGurobi is expecting onnx files to have dynamic batch dimensions, an additional dynamic axis is added
        in the exported onnx file.
        """
        # Set model to eval for onnx export
        self.eval()

        # Declare dummy input with batch_dimension and input/output names
        input_dim = [1] + [dim for dim in self.input_dim]
        rand_input = torch.randn(input_dim)
        input_names = ["input"]
        output_names = ["output"]

        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }

        torch.onnx.export(
            self,
            rand_input,
            self.onnx_path,
            export_params=True,
            opset_version= 11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        print(f"ONNX model was exported successfully to given path: {self.onnx_path}")


    def train_model(self, dataloader, optimizer, loss_fn, epochs, device="cpu"):
        """
        Basic training set up. For more detailed training setups, extracting current pytorch model is recommended and
        manually train it. After that the optimized weights can be reloaded via model.load_state_dict(...)

        Args:
            dataloader: Is expected to be a torch DataLoader having inputs and ground truth outputs
            optimizer: Given optimizer used for training optimization
            loss_fn: Loss criteria to be optimized
            epochs: number of epochs
            device: device where optimization should be performed
        """
        self.model.train()
        self.model.to(device)

        for epoch in range(epochs):
            n_samples = 0.0
            epoch_loss = 0.0
            print(f"[EPOCH {epoch}] has started. \n")
            for batch_inputs, batch_outputs in dataloader:
                b_inputs = batch_inputs.to(device, non_blocking=True)
                b_outputs = batch_outputs.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(b_inputs)
                loss = loss_fn(logits, b_outputs)
                loss.backward()
                optimizer.step()

                b_size = batch_inputs.size(0)
                n_samples += b_size
                epoch_loss += loss.item() * b_size

            print(f"Epoch iteration has ended. \n")
            print(f"Avg. epoch loss: {epoch_loss / max(n_samples, 1)}")

        self.update_gurobi_model_builder()


    def update_gurobi_model_builder(self):
        """
        Updates the current instance of self.gurobi_model_builder, which is used to extract Gurobi model of current
        instance. Therefore, when changes like loading weights are done to a model instance, this update version should
        be called!

        Notes:
            For always updating Gurobi by the latest internal state, a new onnx version is exported and therefore used
            to update the internal Gurobi instance.
        """
        # Re-export to get the latest updated model state
        self.export_onnx()
        model_builder = GurobiModelBuilder(self.onnx_path)
        model_builder.build_model()
        self.gurobi_model_builder = model_builder
        return self.gurobi_model_builder


    def get_gurobi_with_input_assignment(self, input_data, eps=0.0):
        """
        Can be used to simply assign an input tensor to its corresponding input variables of the internal Gurobi model.
        The assignment restricts the input variables to an epsilon environment of the respective, specific input value.
        So for each input variable, constraints are added, representing input - eps <= input_var <= input + eps.

        Args:
            input_data: tensor holding the specific input values
            eps: threshold to define the width of the eps. environment around specific input value

        Returns:
            gurobi_model: A copy! of the current Gurobi instance, supplemented by the input constraints described above.

        Notes:
            Since this method returns a new instance, every change or new constraint made to the returned gurobi model
            will not further affect the internal Gurobi model of this nn.Module instance!
        """
        # Check if given amount of specific input values is equal to amount of input variables from Gurobi
        gurobi_input_vars = self.get_gurobi_input_vars()
        if gurobi_input_vars is None:
            raise RuntimeError("No variables found for 'input' tensor.")
        input_data = input_data.clone()
        input_data_flat = torch.flatten(input_data)
        if len(input_data_flat) != len(gurobi_input_vars):
            raise RuntimeError(f"Given input couldn't be assigned to input vars. "
                               f"Number of inputs {len(input_data_flat)} and variables {len(gurobi_input_vars)} vary.")

        gurobi_model = self.gurobi_model_builder.get_gurobi_model().copy()
        old_input_vars = list(gurobi_input_vars.values())
        new_input_vars = gurobi_model.getVars()
        var_map = {v_old.VarName: v_new for v_old, v_new in zip(old_input_vars, new_input_vars)}

        for idx, var in gurobi_input_vars.items():
            if isinstance(idx, int):
                md_idx = np.unravel_index(idx, input_data.shape[1:])  # Exclude batch dimension
            else:
                md_idx = idx
            original_value = float(input_data[0, *md_idx])
            lb = max(0.0, original_value - eps)
            ub = min(1.0, original_value + eps)
            var = var_map[gurobi_input_vars[idx].VarName]
            gurobi_model.addConstr(var >= lb, name=f"input_lb_{idx}")
            gurobi_model.addConstr(var <= ub, name=f"input_ub_{idx}")

        gurobi_model.update()

        return gurobi_model


    def get_gurobi_input_vars(self):
        """
        Returns: input variables of the current internal Gurobi instance.
        """
        return self.gurobi_model_builder.variables['input']


    def get_gurobi_output_vars(self):
        """
        Returns: output variables of the current internal Gurobi instance.
        """
        return self.gurobi_model_builder.variables['output']


    def get_torch_model(self) -> nn.Module:
        """
        Returns: current internal pytorch model.
        """
        return self.model
