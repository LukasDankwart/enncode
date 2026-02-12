from gurobipy import GRB
from enncode.gurobiModelBuilder import GurobiModelBuilder

from examples.data import get_mnist_batch
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort


def main():
    mnist_inputs, labels = get_mnist_batch(5, flatten=True)

    # Path to the ONNX Net-file
    conv_onnx = "examples/conv_net.onnx"
    model_builder = GurobiModelBuilder(conv_onnx)
    model_builder.build_model()
    gurobi_model = model_builder.get_gurobi_model()

    input_vars = model_builder.get_input_vars()
    output_vars = model_builder.get_output_vars()

    tmp_constraints = []
    for idx in range(mnist_inputs.shape[0]):
        mnist_input = mnist_inputs[idx]
        label = labels[idx]
        # For searching CX access to input and output variables is needed
        # and 'distance' variables

        gurobi_model.setAttr("LB", list(input_vars.values()), 0.0)
        gurobi_model.setAttr("UB", list(input_vars.values()), 1.0)
        dist_vars = gurobi_model.addVars(len(input_vars), name="dist_vars")
        for flat_idx, (_, input_var) in enumerate(input_vars.items()):
            dist_var = dist_vars[flat_idx]
            c1 = gurobi_model.addConstr(dist_var >= mnist_input[flat_idx] - input_var)
            c2 = gurobi_model.addConstr(dist_var >= input_var - mnist_input[flat_idx])
            tmp_constraints.append([c1, c2])

        # Define target class for which a CX is desired
        target_class = np.random.randint(0, 10)
        target_idx = list(output_vars.keys())[target_class]
        for flat_idx, (_, output_var) in enumerate(output_vars.items()):
            if target_class != flat_idx:
                c_dist = gurobi_model.addConstr(output_vars[target_idx] >= output_var + 0.001)
                tmp_constraints.append([c_dist])

        gurobi_model.setParam("TimeLimit", 10)
        gurobi_model.setObjective(dist_vars.sum(), GRB.MINIMIZE)
        gurobi_model.optimize()

        results = []
        for input_var in input_vars.values():
            results.append(input_var.X)
        cx = np.array(results).reshape((28, 28))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(np.array(mnist_input).reshape((28, 28)), cmap='viridis')
        ax1.set_title(f"Original Input (Class: {label})")
        ax1.axis('off')

        ax2.imshow(cx, cmap='viridis')
        ax2.set_title(f"CX (predicted Class: {target_class})")
        ax2.axis('off')

        plt.savefig(f'cx_example_{idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
        cx = np.expand_dims(np.expand_dims(np.array(results).reshape((28, 28)), axis=0), axis=0)
        session = ort.InferenceSession(conv_onnx)
        nn_output_on_cx = session.run(None, {"input": cx.astype(np.float32)})
        nn_prediction_on_cx = np.argmax(nn_output_on_cx[0])
        print("Target Class: ", target_class)
        print("Predicted Class of CX: ", nn_prediction_on_cx)

        # Cleanup for next iteration
        gurobi_model.remove(tmp_constraints)


if __name__ == "__main__":
    main()