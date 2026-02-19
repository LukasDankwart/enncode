#%%
from enncode.gurobiModelBuilder import GurobiModelBuilder


# %%
# Define the path to the desired ONNX network
path = "data/conv_net.onnx"

# %%
model_builder = GurobiModelBuilder(path)
model_builder.build_model()

# %%
gurobi_model = model_builder.get_gurobi_model()
input_vars = model_builder.get_input_vars()
output_vars = model_builder.get_output_vars()


# %%
model_builder = GurobiModelBuilder(path, compcheck=True, rtol=1e-3, atol=1e-4)
model_builder.build_model()
