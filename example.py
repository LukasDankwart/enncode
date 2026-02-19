import onnx

from enncode.gurobiModelBuilder import GurobiModelBuilder
from enncode.compatibility import compatibility_check

if __name__ == "__main__":


    #path = "tests/vnncomp/acasxu/ACASXU_run2a_1_1_batch_2000.onnx"

    path = "tests/models/example4.onnx"


    model_builder = GurobiModelBuilder(path, simplification=True)
    model_builder.build_model()
    print(model_builder.onnx_model_path)
    #compatibility_check(model_builder.onnx_model_path, iterative_analysis=False)
    """

    model = onnx.load(path)
    print("--- Inputs ---")
    for inp in model.graph.input:
        print(
            f"Name: {inp.name}, Shape: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]}")

    print("\n--- Zwischen-Shapes (Value Info) ---")
    if not model.graph.value_info:
        print("KEINE Zwischen-Shapes vorhanden! (Das könnte den KeyError erklären)")
    for vi in model.graph.value_info:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in vi.type.tensor_type.shape.dim]
        print(f"Tensor: {vi.name}, Shape: {shape}")

    print("\n--- Node Attribute Check (Beispiel Gemm) ---")
    for node in model.graph.node:
        if node.op_type == "Gemm":
            print(f"Node: {node.name}, Attribute: {[attr.name for attr in node.attribute]}")

    print(f"\n ========== AB JETZT SIMPLIFIED ========= \n")
    model = onnx.load(path.removesuffix(".onnx") + "_simplified.onnx")

    print("--- Inputs ---")
    for inp in model.graph.input:
        print(
            f"Name: {inp.name}, Shape: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]}")

    print("\n--- Zwischen-Shapes (Value Info) ---")
    if not model.graph.value_info:
        print("KEINE Zwischen-Shapes vorhanden! (Das könnte den KeyError erklären)")
    for vi in model.graph.value_info:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in vi.type.tensor_type.shape.dim]
        print(f"Tensor: {vi.name}, Shape: {shape}")

    print("\n--- Node Attribute Check (Beispiel Gemm) ---")
    for node in model.graph.node:
        if node.op_type == "Gemm":
            print(f"Node: {node.name}, Attribute: {[attr.name for attr in node.attribute]}")
    """