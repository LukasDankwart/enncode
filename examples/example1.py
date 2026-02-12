import numpy as np
from onnx_to_gurobi.gurobiModelBuilder import GurobiModelBuilder
from gurobipy import GRB

# 1) Convert the ONNX model to a Gurobi model
model_builder = GurobiModelBuilder("fc1.onnx")
model_builder.build_model()

# 2) Provide input data
dummy_input = np.random.rand(1, 28, 28).astype(np.float32)
input_vars = model_builder.variables.get("input")

# Assign each element of dummy_input to the corresponding Gurobi variable
input_shape = dummy_input.shape
for idx, var in input_vars.items():
    # Convert the variable index to the correct multi-dimensional index in dummy_input
    if isinstance(idx, int):
        md_idx = np.unravel_index(idx, input_shape[1:])  # Exclude batch dimension
    else:
        md_idx = idx
    # Fix the variable to the dummy_input value
    value = float(dummy_input[0, *md_idx])
    var.lb = value
    var.ub = value

# 3) Optimize
gurobi_model = model_builder.get_gurobi_model()
gurobi_model.optimize()
if gurobi_model.status != GRB.OPTIMAL:
    raise ValueError(f"Model couldn't be optimized to an optimal solution. Status: {gurobi_model.status}")

# 4) Retrieve outputs
output_vars = model_builder.variables.get("output")
output_shape = model_builder.in_out_tensors_shapes["output"]
model_output = np.zeros((1,) + tuple(output_shape), dtype=np.float32)
for idx, var in output_vars.items():
    if isinstance(idx, int):
        md_idx = np.unravel_index(idx, output_shape)
    else:
        md_idx = idx
    model_output[(0,) + md_idx] = var.x

print("Gurobi model output:")
print(model_output)
