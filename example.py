import onnx

from enncode.gurobiModelBuilder import GurobiModelBuilder
from enncode.compatibility import compatibility_check

if __name__ == "__main__":



    path = "tests/networks/power/flow.onnx"


    model_builder = GurobiModelBuilder(path, simplification=True, compcheck=True)
    model_builder.build_model()
    print(model_builder.onnx_model_path)

