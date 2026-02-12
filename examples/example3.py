from onnx_to_gurobi.gurobiModelBuilder import GurobiModelBuilder
from tests.tests_utils import run_onnx_model
from gurobipy import GRB, quicksum
import numpy as np

if __name__ == "__main__":

    # 1) Load specified concrete/diabetes/power/wine network and Conver the ONNX model to a Gurobi model
    path = "onnxgurobi/tests/networks/power/flow.onnx"
    model_builder = GurobiModelBuilder(path)
    model_builder.build_model()
    gurobi_model = model_builder.get_gurobi_model()
    print(f"Initializing ONNXGurobi Model from {path} was successful.")

    # 2) Define slight perturbations for input data
    eps = 0.2
    delta = 0.01
    M = 1000

    # 3) Provide input data with defined perturbation
    input_vars = model_builder.variables.get('onnx::MatMul_0')
    if input_vars is None:
        raise ValueError("No variables found for input tensor.")
    input_shape = model_builder.internal_onnx.in_out_tensors_shapes['onnx::MatMul_0']
    rand_input = np.expand_dims(np.random.rand(*input_shape).astype(np.float32), axis=0)
    for idx, var in input_vars.items():
        if isinstance(idx, int):
            md_idx = np.unravel_index(idx, input_shape[1:])  # Exclude batch dimension
        else:
            md_idx = idx
        original_value = float(rand_input[0, *md_idx])
        lb = max(0.0, original_value - eps)
        ub = min(1.0, original_value + eps)
        gurobi_model.addConstr(var >= lb, name=f"input_lb_{idx}")
        gurobi_model.addConstr(var <= ub, name=f"input_ub_{idx}")

    # 4) Determine GT output distribution for comparison
    onnx_output = run_onnx_model(path, rand_input[0], 'onnx::MatMul_0')
    onnx_output_dist = np.flip(np.argsort(onnx_output)) # In descending order

    # 5) Add constraints for same rank-ordering of output vars (for same 'argmax'-distribution)
    output_vars = model_builder.variables.get('66')
    binary_vars = {}
    if output_vars is None:
        raise ValueError("No variables found for output tensor.")
    for idx in range(len(onnx_output_dist) - 1):
        bigger_var = output_vars[(onnx_output_dist[idx],)]
        smaller_var = output_vars[(onnx_output_dist[idx + 1],)]

        binary_vars[idx] = gurobi_model.addVar(vtype=GRB.BINARY, name=f"swap_at_rank_{idx}")

        gurobi_model.addConstr(
            bigger_var - smaller_var <= -delta + M * (1 - binary_vars[idx]),
            name=f"constr_swap_rank_{idx}"
        )

    gurobi_model.addConstr(
        quicksum(binary_vars[r] for r in binary_vars) >= 1,
        name="at_least_one_swap_constr"
    )

    # 6) Optimize the model
    gurobi_model.optimize()

    if gurobi_model.status == GRB.OPTIMAL:
        output_shape = model_builder.internal_onnx.in_out_tensors_shapes['66']
        model_output = np.zeros((1,) + tuple(output_shape), dtype=np.float32)
        for idx, var in output_vars.items():
            if isinstance(idx, int):
                md_idx = np.unravel_index(idx, output_shape)
            else:
                md_idx = idx
            model_output[(0,) + md_idx] = var.X
        model_output_dist = np.flip(np.argsort(model_output[0]))

        print(f"\n Robustness regarding the issue ranking is refuted!")
        print(f"An input with {eps}-distance has been found and changes the output ordering.")
        print(f"ONNX output ordering: {onnx_output_dist}")
        print(f"Gurobi output ordering: {model_output_dist}")
        for idx in range(len(onnx_output_dist)):
            shift = idx - np.where(model_output_dist == onnx_output_dist[idx])[0]
            print(f"-> GT rank {idx} has shifted {shift} ranks.")

    elif gurobi_model.status == GRB.INFEASIBLE:
        print(f"\n Robustness regarding the issue ranking was proved for input {rand_input} with [{eps}] distance")

    else:
        raise ValueError(f"Model could not be optimized. Status: {gurobi_model.status}")
