import numpy as np

from enncode.compatibility import compatibility_check
from examples.data import get_mnist_batch
import matplotlib.pyplot as plt

def main():
    path = "softmax/conv_softmax.onnx"

    #compatibility_check(path, iterative_analysis=False, rtol=1e-03, atol=1e-04)
    imgs, labels = get_mnist_batch(size=40)

    results = []
    two = False
    three = False
    for idx in range(imgs.shape[0]):
        if labels[idx] == 2 and not two:
            plt.imsave("mnist_original.png", np.array(imgs[idx][0]), cmap='gray')
            break


if __name__ == "__main__":
    main()