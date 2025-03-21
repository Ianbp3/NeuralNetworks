import numpy as np

class ReLU:
    def forward(self, inputs: list[float]) -> None:
        self.output = np.maximum(0, inputs)

    def backward(self):
        print("sdrawkcab gniog mI")

class Softmax:
    def forward(self, inputs: list[float]) -> None:
        self.outputs = []
        for i in inputs:
            exp = np.exp(i - np.max(i))
            output = exp/np.sum(exp, axis=0)
            self.outputs.append(output)

    def backward(self):
        print("sdrawkcab gniog mI")