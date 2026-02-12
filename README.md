# Overview

The eNNcode is a Python library that creates Gurobi models for neural networks in ONNX format.

The library has been designed to allow easy extensions, and it currently supports the following ONNX nodes:

- Add
- AveragePool
- BatchNormalization
- Concat
- Conv
- Div
- Dropout
- Flatten
- Gemm
- Identity
- MatMul
- MaxPool
- Mul
- Relu
- Reshape
- Sub
- Unsqueeze


# Installation

We highly recommend creating a virtual conda environment and installing the library within the environment by following the following steps:

1- Gurobi is not installed automatically. Please install it manually using:
```
    conda install -c gurobi gurobi
```
2- Make sure to switch to Python 11 inside your environment using:
```
    conda install python=11
``` 

3- Install the library using:
```
    pip install enncode
```

* Probably a known fact, but the Gurobi optimizer needs to be manually 
installed to the platform and a valid license has to be activated. 
* As the end of 2025, Gurobi has academic licenses available - at no cost -
to students, faculty, and staff at accredited degree-granting institutions.
# Getting Started

The ```GurobiModelBuilder``` class provides the central interface for converting an ONNX model into a Gurobi optimization model.

To get access to the class's methods and attributes, you need to import it using:

```
from enncode.GurobiModelBuilder import GurobiModelBuilder
```


The ```GurobiModelBuilder``` class:

- Parses the ONNX graph and constructs an internal representation of each operator and its corresponding tensor shapes.

- Creates a Gurobi model along with the necessary variables and constraints.

- Exposes all model components (decision variables, Gurobi Model object, node definitions, tensor shapes), allowing you to:

* Set or fix input variables to specific values.

* Introduce objectives.

* Add your own constraints.

* Solve the resulting MILP and then inspect or extract the outputs from the solution.


An overview of the class’s methods and attributes:

```
class GurobiModelBuilder:
    def build_model(self):
        """
        Constructs the Gurobi model by creating variables and applying operator constraints.

        """

    def get_gurobi_model(self):
        """
        Retrieves the Gurobi model after all constraints have been added.

        Returns:
            gurobipy.Model: The constructed Gurobi model reflecting the ONNX graph.
        """

    # Attributes:
    self.model               # The Gurobi Model object
    self.variables           # A dict mapping tensor names to Gurobi variables (or constants)
    self.in_out_tensors_shapes # Shapes of all input and output tensors
    self.nodes               # Node definitions parsed from ONNX
    self.initializers        # Constant tensors extracted from the ONNX graph

```

# How to Use

See [example1.py](./examples/example1.py) for a simple example.
See [example2.py](./examples/example2.py) or [exampe3.py](./examples/example3.py) for more detailed adversarial examples.
See [example4.py](./examples/example4.py) for the usage of the model builder, which is described down below.

## Compatibility
To get access to the following compatiblity checks, you need to import them by:
```
from onnx_to_gurobi.compatibility import compatibility_check, get_unsupported_note_types, check_equivalence
```

The compatiblity.py was implemented as an interface for checking GurobiModelBuilder compatibility for an onnx model. 
There are mainly three methods that can be used to check if an ONNX model can be parsed to a Gurobi model.

1. For a comprehensive review, it is recommended to use the ```compatibility_check``` method. The specified ONNX file can be stored in standard .onnx or compressed .onnx.gz format and can be checked as follows:
```
def compatibility_check(onnx_path, iterative_analysis=True, output_dir=None, save_subgraphs=True, rtol=1e-05, atol=1e-08):
    """
    This method is used for testing compatibility with GurobiModelBuilder. Tries to parse specified ONNX model to a Gurobi model.
    If successful, equivalence of an ONNX run and its corresponding Gurobi model is checked (with rtol/atol tolerance).
    
    If not successful, an iterative analysis can be started via function argument flag. If true, every subgraph is 
    extracted and checked for compatibility with GurobiModelBuilder. This is done to identify nodes which cause misconduct, 
    which might be not solely attributable to the node type itself. 
    
    Args:
        onnx_path: path to the initial onnx model.
        iterative_analysis: flag to start iterative analysis, if initial check fails.
        output_dir: name of output directory where subgraphs and log file can be stored.
        save_subgraphs: flag to control, if every subgraph should be stored or be deleted after successful check.
        rtol/atol: tolerances for the equivalence check (via np.allclose)
    """
```
 As GurobiModelBuilder expects a dynamic batch dimension, the compatibility check includes adding a dynamic batch dimension and 
restoring the ONNX model as '*_modified.onnx' in the directory of onnx path. If initial compatibility check fails and 
 iterative analysis should be done, a new directory 'subgraphs' is created at the given output path, where 
 the subgraphs are stored. At the same directory, a logfile.txt is written including following information for each subgraph:
```
    [NODE *]: NAME_OF_NODE
     [PASSED]: Extracting current subgraph was successful, stored at 'path/of/stored/onnx_subgraph_endingnode.onnx'. 
     [PASSED]: Compatibility. # Indicating no error occured while parsing 
     [PASSED]: Equivalence check for ('path/of/stored/onnx_subgraph_endingnode.onnx') has been successful.
```
Via the logfile the node causing misconduct can be found and its most likely reason for incompatibility.
Notes:
* If extracting current subgraph fails, there might be an error using 'onnx.utils.extract_model' being independent of GurobiModelBuilder.
* It was observed that equivalence check might fail because Gurobi doesn't find a valid solution (status code -3). 
A restart can sometimes still show equivalence before using the following function.

2. For a quick check, if all nodes included in the onnx model are supported by onnx to gurobi, the ```get_unsupported_note_types``` 
method can be used.
```
def get_unsupported_note_types(onnx_path):
    """
    Checks every node type, included in onnx path, and checks for support by the current GurobiModelBuilder version.
    Prints and returns a list of all node types that are not supported and most likely cause incompatibility. 
    
    Args: 
        onnx_path: path to the onnx file.
    """
```
If the returned/printed list isn't empty, included node types are not supported by the current version. 

3. Via ```check_equivalence```, a manual compatibility test can be set up. 
Therefor it is very important to determine input tensor, its shape, name and just like that the output tensor correctly. In addition, this method does not include adding a dynamic batch dimension to the onnx file. Therefore, this must be done and checked manually.
```
check_equivalence(onnx_path, model_input, model_input_names, target_output_names, log_file_path, rtol=1e-05, atol=1e-08):
    """
    Runs onnx inference and GurobiModelBuilder solving for given onnx model.
    
    Args:
        onnx_path: path to the onnx model.
        model_input: tensor of valid input shape of specified onnx model.
        model_input_names: corresponding names of model input tensor.
        target_output_names: names of the output tensors from the onnx model.
        ... (remaining args are similiar as above) 
    
    Returns:
        True: If check has shown compatibility and equivalance for ONNX/Gurobi run on same input instance.
        False: If check has shown compatibility but missing equivalence between both outputs.
        None: If check has show incompatiblity, leading to an exception (most likely written to the logfile)
    """
```
Please ensure the model has a dynamic batch dimension and input/output tensors have a shape of [1, remaining dimensions].

### Additional Notes for compatiblity checks: 
* All compatibility checks mentioned are designed so that the inputs expect only one tensor. The number of outputs tensors can be greater.
* Since most parsers are designed to exclude the batch dimension, the checks are designed to have a dynamic batch dimension at axis 0. If the ONNX model expects 1D inputs, please re-export with an additional batch dimension. If the model has input dimension >1, the first dimension is interpreted as batch dimension by the aforementioned 'compatibility_check'. 
* While testing aspects of the eNNcode library, it has been observed, that from a certain version onwards, Pytorch might ignore the onnx op version flag while exporting the onnx file. It is essential to ensure that the ONNX file is in opset version 11.
* In addition, for ensuring things like the modified onnx file has a dynamic batch dimension, correct opset version or correct input/output names, we recommend [Netron](https://netron.app/) as a visualization tool. 

## Model Builder
In addition to using existing ONNX models, a model builder was explicitly implemented for PyTorch. At the current state 
it is only capable of creating a simple fully connected neural network, consisting of nn.Linear and nn.ReLU layers. 
In this way, a network can be created from the aforementioned layer types, which is compatible for the use of the eNNcode library.
The following import is required for usage:

```
from onnx_to_gurobi.modelBuilder import SimpleFCModel
```
The use of the class ```SimpleFCModel``` is described below:
#### 1. Create instance of ```SimpleFCModel```: 
```
class SimpleFCModel(nn.Module):
    def __init__(input_dim, hidden_dim, output_dim, output_activation, onnx_path = "simple_fc_model.onnx"):
    """
        Args:
            input_dim: (1D) input dimension for the network (integer/list)
            hidden_dim: list of dimensions for each hidden layer 
            output_dim: output dimension of the network
            output_activation: nn.* activation function for the outputlayer
            onnx_path: path for the onnx export 
    """
        super().__init__()
        # Setting attributes
        ...
        
        # Creating PyTorch and (ONNX)Gurobi model
        self.model = self.create_model()
        self.gurobi_model_builder = self.update_gurobi_model_builder()
```
As described above, for every dimension given in the 'hidden_dims' argument a Linear + ReLU layer is added with corresponding dimension.
If no explicit activation function is given, also a ReLU activation is used for the output layer. As shown, the constructor also builds the corresponding PyTorch and Gurobi model.

Note: To export the model in ONNX format you can use ```self.export_onnx()``` method. It exports a ONNX format of the current state and also adds a dynamic batch dimension, since this is expected by the GurobiModelBuilder class.

#### 2. PyTorch model via ```create_model()```:
```
def create_model(self):
    """
    Called by constructor when the model is created. For each entry in self.hidden_dims a fc linear + relu layer
    is added with corresponding input dimensions of the previous layer.

    Returns: nn.Sequential module consisting of fc linear + relu layers
    """
```
The PyTorch model can be accessed with ```simple_fc_model.get_torch_model()```

Aside from that the class also has a train method which can be used for basic optimization. 
But fore more complex training, manually optimizing and reloading the optimized parameters into the ```SimpleFCModel``` instance is recommended.
```
def train_model(self, dataloader, optimizer, loss_fn, epochs, device="cpu"):
    """
    Performs (basic) optimization iterations with given arguments.
    
    Args:
        dataloader: torch dataloader containing batch inputs and ground truth outputs.
        optimizer: optimizer used for training network parameters.
        loss_fn: optimization criteria used for computing loss on current predictions.
        epochs: number of optimization iterations.
        device: device where the optimization is performed.
    """
    ...
```
Note: If you want to use the internal Gurobi model, the train method described above also calls ```update_gurobi_model()```, which is described in the following.
If you had optimized and reloaded the model parameters and still want to use the internal Gurobi model, always call ```update_gurobi_model()``` manually. 
Otherwise the Gurobi model is not updated with the new parameters.

#### 3. Internal GurobiModelBuilder model 
The ```self.gurobi_model_builder``` is an instance of the ```GurobiModelBuilder``` class, which was described earlier in this guide.
The attribute always stores an GurobiModelBuilder instance of the most recent state of the network.
To be consistent the ```update_gurobi_model()``` should be called, whenever changes to model parameters are made.

The Gurobi model can be accessed via:
```
gurobi_model = simple_fc_model.gurobi_model_builder.get_gurobi_model()

gurobi_input_vars = simple_fc_model.get_gurobi_input_vars()
gurobi_output_vars = simple_fc_model.get_gurobi_output_vars()

# Manually assign input assignments and output constraints
...
```
Required information like input names etc. can be accessed directly from the ```gurobi_model_builder```. You may want to retrieve the internal Gurobi model, but not to make every change or additional condition stored in the network's Gurobi model.
Therefor use ```gurobi_model.copy()``` to create a new object. 

In addition to that, the method ```get_gurobi_with_input_assignment(...)``` was implemented to get a copy of the current internal Gurobi model with additional (basic) input assignment:
```
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
    """
```
Note: Since this method returns a new instance, every change or new constraint made to the returned gurobi model will not further affect the internal Gurobi model of the ```SimpleFCModel``` instance!

#### 4. Usage of ```SimpleFCModel```
Although [example4.py](./examples/example4.py) already shows the use of the class, a small example follows to illustrate a typical process:
```
simple_fc_model = SimpleFCModel(
    input_dim=32,
    hidden_dim=[128, 64],
    output_dim=10,
    output_activation=nn.ReLU(),
    onnx_path="dummy_networks.onnx"
)
torch_model = simple_fc_model.get_torch_model()

# In case you might manually optimize the network extern, reload the weights and
# manually update the internal Gurobi model builder!
torch.save(torch_model.state_dict(), "path_to_optimized_weights.pth")
# ...
torch_model.load_state_dict(torch.load("path_to_optimized_weights.pth", weights_only=True))
_ = simple_fc_model.update_gurobi_model_builder()

# Generate dummy input with additional batch dimension!
dummy_input = torch.randn([1] + simple_fc_model.input_dim)

gurobi_input_vars = simple_fc_model.get_gurobi_input_vars()
gurobi_output_vars = simple_fc_model.get_gurobi_output_vars()
gurobi_model = simple_fc_model.get_gurobi_with_input_assignment(dummy_input, eps=1e-4)

for idx, output_var in gurobi_output_vars.items():
    # Since gurobi model is a new instance, accessing vars is done by the names, since they are unchanged
    var_name = output_var.VarName
    var = gurobi_model.getVarByName(var_name)

    gurobi_model.addConstr(
        var >= 0,
        name="just_a_dummy_constraint"
    )

gurobi_model.optimize()
```

# Important Notes

* Make sure your model is exported into ONNX using opset version 11.
* The library doesn't support recurrent neural networks (RNNs).
* The 3-D convolutional operation isn't supported.
* If an initializer/constant (e.g. weights) is used as an input to the MatMul, the node expects it to be the second input.
* The Concat node’s output must match the input shape of the following layer. In addition, the node expects only 2 inputs.
* Since our library is designed solely for production and not for training, we encode the Dropout node to
function only in inference mode, which means that its input passes through unchanged.