import DenseLayer as dl
import ActivationFunc as af
import LossFunc as lf
import MnistDataset as MnistDs
import PlotFuncs as plts
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
Epochs = 20
l_range = 0
h_range = 64
Loss_history = []
Acc_history = []
loss_f = 0
accu_f = 0

for i in range(Epochs):
    for i in range(930):
        inputs = mnist_train.images[l_range+i*batches_size:h_range+i*batches_size]
        labels = mnist_train.onehotlabels[l_range+i*batches_size:h_range+i*batches_size]
        #Forward-----------------------------------------------------------------------------------
        layer.forward(inputs)
        relu.forward(layer.outputs)
        outputlayer.forward(relu.output)
        softmax.forward(outputlayer.outputs)

        #Loss and Accuracy Calculations ------------------------------------------------------------
        cross_ent = lf.CrossEntropy()
        cross_ent.forward(labels, softmax.outputs)
        loss_f = cross_ent.loss_mean
        accu_f = softmax.accuracy(labels)

        #Backward Propagation ----------------------------------------------------------------------------------
        cross_ent.gradient()
        dvalues_out = cross_ent.grad

        outputlayer.backward(dvalues_out, relu.output)

        relu.backward(outputlayer.dinputs, layer.outputs)
        dvalues1 = relu.drelu

        layer.backward(dvalues1, inputs)

        #Update Weights and Biases ---------------------------------------------------------------------------
        layer.update()
        outputlayer.update()

    Loss_history.append(loss_f)
    Acc_history.append(accu_f)

layer.save("layer_Basic.pkl")
outputlayer.save("out_layer_Basic.pkl")

plts.loss_epochs(Loss_history, Epochs)
plts.accu_epochs(Acc_history, Epochs)
plts.correlation(Acc_history, Loss_history)
plts.loss_acc(Acc_history, Loss_history)
plts.vis_correlation(Acc_history, Loss_history)