import numpy as np

class ReLU:
    def forward(self, inputs: list[float]) -> None:
        self.output = np.maximum(0, inputs)

    def backward(self):
        print("sdrawkcab gniog mI")

class Softmax:
    def forward(self, inputs: list[float]) -> None:
        exp = np.exp(inputs - np.max(inputs))
        self.output = exp/np.sum(exp, axis=1)

    def backward(self):
        print("sdrawkcab gniog mI")