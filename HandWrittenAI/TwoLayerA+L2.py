import DenseLayer as dl
import ActivationFunc as af
import LossFunc as lf
import MnistDataset as MnistDs
import PlotFuncs as plts
import Optimizers as opt
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
layer = dl.DenseLayer(len(mnist_train.images[0]), 128, lambda_l2=0.00001)
layer2 = dl.DenseLayer(128, 128, lambda_l2=0.00001)
outputlayer = dl.DenseLayer(128, 10, lambda_l2=0.00001)
relu = af.ReLU()
relu2 = af.ReLU()
softmax = af.Softmax()

#Training ---------------------------------------------------------------------------------
batches_size = 64
Epochs = 40
l_range = 0
h_range = 64
Loss_history = []
Acc_history = []
loss_f = 0
accu_f = 0
adam = opt.AdamOptimizer(learning_rate=0.01)

for i in range(Epochs):
    for i in range(930):
        inputs = mnist_train.images[l_range+i*batches_size:h_range+i*batches_size]
        labels = mnist_train.onehotlabels[l_range+i*batches_size:h_range+i*batches_size]
        #Forward-----------------------------------------------------------------------------------
        layer.forward(inputs)
        relu.forward(layer.outputs)
        layer2.forward(relu.output)
        relu2.forward(layer2.outputs)
        outputlayer.forward(relu2.output)
        softmax.forward(outputlayer.outputs)

        #Loss and Accuracy Calculations ------------------------------------------------------------
        cross_ent = lf.CrossEntropy()
        cross_ent.forward(labels, softmax.outputs)
        loss = cross_ent.loss_mean
        regloss = layer.get_l2_loss() + outputlayer.get_l2_loss()
        loss_f = loss+regloss
        accu_f = softmax.accuracy(labels)

        #Backward Propagation ----------------------------------------------------------------------------------
        cross_ent.gradient()
        dvalues_out = cross_ent.grad

        outputlayer.backward(dvalues_out, relu.output)

        relu2.backward(outputlayer.dinputs, layer2.outputs)
        layer2.backward(relu2.drelu, relu.output)

        relu.backward(layer2.dinputs, layer.outputs)
        dvalues1 = relu.drelu

        layer.backward(dvalues1, inputs)

        #Update Weights and Biases ---------------------------------------------------------------------------
        adam.update(layer)
        adam.update(layer2)
        adam.update(outputlayer)

    Loss_history.append(loss_f)
    Acc_history.append(accu_f)

layer.save("2layer_AdamL2.pkl")
layer2.save("2Layer2_AdamL2.pkl")
outputlayer.save("out_layer_AdamL2.pkl")

plts.loss_epochs(Loss_history, Epochs)
plts.accu_epochs(Acc_history, Epochs)
plts.correlation(Acc_history, Loss_history)
plts.loss_acc(Acc_history, Loss_history)
plts.vis_correlation(Acc_history, Loss_history)