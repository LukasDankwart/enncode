from onnx_to_gurobi.modelBuilder import SimpleFCModel
from gurobipy import GRB
import torch.nn as nn
import torch
import onnxruntime as ort
import numpy as np


if __name__ == "__main__":

    """
        This example is used for showcasing the usability of the networkBuilder.py interface.
        The following example has no substantively relevant meaning.
    """

    # 1) Define instance of the SimpleFCModel with specific input, hidden and output dimensions.
    model_path = "example4.onnx"
    simple_fc_model = SimpleFCModel(
        input_dim=[32],
        hidden_dim=[16, 64],
        output_dim=5,
        output_activation=nn.ReLU(),
        onnx_path=model_path
    )

    torch.save(simple_fc_model.state_dict(), model_path.removesuffix(".onnx") + ".pth")
    # ... (external training of the model and weights might be made) ...
    simple_fc_model.load_state_dict(torch.load(model_path.removesuffix(".onnx") + ".pth", weights_only=True))
    # When new weights are loaded, always call update_gurobi_model_builder() to update the internal Gurobi instance!
    _ = simple_fc_model.update_gurobi_model_builder()


    # 2) Define a dummy input and compute its output of an onnx run
    rand_input = torch.rand(1, 32)
    session = ort.InferenceSession(model_path)
    onnx_outputs = session.run(None, {'input': np.array(rand_input)})[0]

    # 3) At this point, the internal Gurobi model can be either extracted by...
    gurobi_model = simple_fc_model.gurobi_model_builder.get_gurobi_model()
    # ... or you can get a new instance of the Gurobi model with assigned and bounden input variables.

    print(len(simple_fc_model.gurobi_model_builder.get_gurobi_model().getConstrs()))                    # Length : 995
    gurobi_model = simple_fc_model.get_gurobi_with_input_assignment(rand_input, eps=1e-5)
    print(len(simple_fc_model.gurobi_model_builder.get_gurobi_model().getConstrs()))                    # Length : 995
    print(len(gurobi_model.getConstrs()))                                                               # Length : 1059

    # Note: Since gurobi-model is overwritten by the new copied instance, every further change (e.g. adding more constraints)
    #       will not influence the internal Gurobi instance in the simple_fc_model object!


    # 4) On the new instance, you might want to add new constraints
    gurobi_output_vars = simple_fc_model.get_gurobi_output_vars()
    output_cnt =  0
    for idx, output_var in gurobi_output_vars.items():
        var_name = output_var.VarName
        var = gurobi_model.getVarByName(var_name)
        gurobi_model.addConstr( var >= onnx_outputs[0][output_cnt], name=f"output_lb_onnx_{output_cnt}")
        gurobi_model.addConstr(var <= onnx_outputs[0][output_cnt], name=f"output_ub_onnx_{output_cnt}")
        output_cnt += 1

    gurobi_model.optimize()


    # 5) This dummy 'specification' just shows Gurobi proving that the input variables must be equal to the actual
    #    specific input values (rand_input) to compute the same output on the current state of simple_fc_model.
    gurobi_input_vars = simple_fc_model.get_gurobi_input_vars()
    if gurobi_model.status == GRB.OPTIMAL:
        gurobi_model_input = np.zeros(simple_fc_model.input_dim)
        cnt = 0
        for idx, input_var in gurobi_input_vars.items():
            var_name = input_var.VarName
            var = gurobi_model.getVarByName(var_name)
            gurobi_model_input[cnt] = var.X
            cnt +=1

        print(f"Equivalence for the same input is: {np.allclose(gurobi_model_input, rand_input, atol=1e-5)}")

    elif gurobi_model.status == GRB.INFEASIBLE:
        print(f"\n Robustness regarding the issue ranking was proved for input {rand_input} with [{0.0}] distance")
    else:
        raise ValueError(f"Model could not be optimized. Status: {gurobi_model.status}")


