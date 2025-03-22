import DenseLayer as dl
import ActivationFunc as af
import LossFunc as lf
import MnistDataset as MnistDs
from pathlib import Path

#Importing data: Images and Labels ---------------------------------------------------------
test_images_file = "t10k-images-idx3-ubyte"
test_labels_file = "t10k-labels-idx1-ubyte"
train_images_file = "train-images-idx3-ubyte"
train_labels_file = "train-labels-idx1-ubyte"

dataset_files = [test_images_file, test_labels_file, train_images_file, train_labels_file]

ds_folder_path = Path("dataset")

test_images_path = ds_folder_path / "t10k-images-idx3-ubyte"
test_labels_path = ds_folder_path / "t10k-labels-idx1-ubyte"
train_images_path = ds_folder_path / "train-images-idx3-ubyte"
train_labels_path = ds_folder_path / "train-labels-idx1-ubyte"

mnist_train = MnistDs.MnistDataset()
mnist_train.load(train_images_path, train_labels_path)

#NeuralNetwork Creation -------------------------------------------------------------------
layer = dl.DenseLayer(len(mnist_train.images[0]), 128)
outputlayer = dl.DenseLayer(128, 10)
relu = af.ReLU()
softmax = af.Softmax()

#Training ---------------------------------------------------------------------------------
batches_size = 64

#Forward-----------------------------------------------------------------------------------
layer.forward([mnist_train.images[0],mnist_train.images[1],mnist_train.images[2]])
relu.forward(layer.outputs)
outputlayer.forward(relu.output)
softmax.forward(outputlayer.outputs)

#Loss and Accuracy Calculations ------------------------------------------------------------
cross_ent = lf.CrossEntropy()
cross_ent.forward([mnist_train.onehotlabels[0], mnist_train.onehotlabels[1], mnist_train.onehotlabels[2]], softmax.outputs)
print("Loss: ", cross_ent.loss_mean)
print("Accuracy: ", softmax.accuracy([mnist_train.onehotlabels[0], mnist_train.onehotlabels[1], mnist_train.onehotlabels[2]]))

#Backward Propagation ----------------------------------------------------------------------------------
cross_ent.gradient()
dvalues_out = cross_ent.grad

outputlayer.backward(dvalues_out, relu.output)

relu.backward(outputlayer.dinputs, layer.outputs)
dvalues1 = relu.drelu

layer.backward(dvalues1, [mnist_train.images[0],mnist_train.images[1],mnist_train.images[2]])

#Update Weights and Biases ---------------------------------------------------------------------------
layer.update()
outputlayer.update()
