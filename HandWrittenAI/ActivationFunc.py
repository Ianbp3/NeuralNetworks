import numpy as np

class ReLU:
    def forward(self, inputs: list[float]) -> None:
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues, inputs):
        self.drelu = dvalues.copy()
        self.drelu[inputs <= 0] = 0

class Softmax:
    def forward(self, inputs: list[float]) -> None:
        self.outputs = []
        for i in inputs:
            exp = np.exp(i - np.max(i))
            output = exp/np.sum(exp, axis=0)
            self.outputs.append(output)

    def backward(self, dvalues):
        self.dsoftmax = dvalues

    def accuracy(self, y_real):
        count = 0
        for i in range(len(y_real)):
            if (y_real[i].index(1) == list(self.outputs[i]).index(max(self.outputs[i]))):
                count += 1
        return count/len(y_real)