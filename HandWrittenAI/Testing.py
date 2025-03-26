import DenseLayer as dl
import ActivationFunc as af
import MnistDataset as MnistDs
from pathlib import Path

# Load dataset ------------------------------------------------------
mnist_test = MnistDs.MnistDataset()
test_images_path = Path("dataset") / "t10k-images-idx3-ubyte"
test_labels_path = Path("dataset") / "t10k-labels-idx1-ubyte"
mnist_test.load(test_images_path, test_labels_path)

# Rebuild network ----------------------------------------------------
layer_Adam = dl.DenseLayer(784, 128)
outputlayer_Adam = dl.DenseLayer(128, 10)
relu_Adam = af.ReLU()
softmax_Adam = af.Softmax()

layer_L2 = dl.DenseLayer(784, 128)
outputlayer_L2 = dl.DenseLayer(128, 10)
relu_L2 = af.ReLU()
softmax_L2 = af.Softmax()

layer_Basic = dl.DenseLayer(784, 128)
outputlayer_Basic = dl.DenseLayer(128, 10)
relu_Basic = af.ReLU()
softmax_Basic = af.Softmax()

T_layer_AdamL2 = dl.DenseLayer(784, 128)
T_layer2_AdamL2 = dl.DenseLayer(128, 128)
T_outputlayer_AdamL2 = dl.DenseLayer(128, 10)
T_relu_AdamL2 = af.ReLU()
T_relu2_AdamL2 = af.ReLU()
T_softmax_AdamL2 = af.Softmax()

# Load saved weights -------------------------------------------------
layer_Adam.load("layer_Adam.pkl")
outputlayer_Adam.load("out_layer_Adam.pkl")

layer_L2.load("layer_L2.pkl")
outputlayer_L2.load("out_layer_L2.pkl")

layer_Basic.load("layer_Basic.pkl")
outputlayer_Basic.load("out_layer_Basic.pkl")

T_layer_AdamL2.load("2layer_AdamL2.pkl")
T_layer2_AdamL2.load("2layer2_AdamL2.pkl")
T_outputlayer_AdamL2.load("out_layer_AdamL2.pkl")
# Testing forward pass ----------------------------------------------
layer_Adam.forward(mnist_test.images)
relu_Adam.forward(layer_Adam.outputs)
outputlayer_Adam.forward(relu_Adam.output)
softmax_Adam.forward(outputlayer_Adam.outputs)

# Accuracy ----------------------------------------------------------
acc = softmax_Adam.accuracy(mnist_test.onehotlabels)
print(f"Test Adam Accuracy: {acc * 100:.2f}%")

# Testing forward pass ----------------------------------------------
layer_L2.forward(mnist_test.images)
relu_L2.forward(layer_L2.outputs)
outputlayer_L2.forward(relu_L2.output)
softmax_L2.forward(outputlayer_L2.outputs)

# Accuracy ----------------------------------------------------------
acc = softmax_L2.accuracy(mnist_test.onehotlabels)
print(f"Test L2 Accuracy: {acc * 100:.2f}%")

# Testing forward pass ----------------------------------------------
layer_Basic.forward(mnist_test.images)
relu_Basic.forward(layer_Basic.outputs)
outputlayer_Basic.forward(relu_Basic.output)
softmax_Basic.forward(outputlayer_Basic.outputs)

# Accuracy ----------------------------------------------------------
acc = softmax_Basic.accuracy(mnist_test.onehotlabels)
print(f"Test Basic Accuracy: {acc * 100:.2f}%")

# Testing forward pass ----------------------------------------------
T_layer_AdamL2.forward(mnist_test.images)
T_relu_AdamL2.forward(T_layer_AdamL2.outputs)
T_layer2_AdamL2.forward(T_relu_AdamL2.output)
T_relu2_AdamL2.forward(T_layer2_AdamL2.outputs)
T_outputlayer_AdamL2.forward(T_relu2_AdamL2.output)
T_softmax_AdamL2.forward(T_outputlayer_AdamL2.outputs)

# Accuracy ----------------------------------------------------------
acc = T_softmax_AdamL2.accuracy(mnist_test.onehotlabels)
print(f"Test Two_Layer Adam + L2 Accuracy: {acc * 100:.2f}%")