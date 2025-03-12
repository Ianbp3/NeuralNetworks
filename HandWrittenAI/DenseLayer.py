import numpy as np

class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.weights = np.random.randn(n_inputs,n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: list[int]) -> None:
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self):
        print("Im I training?")

    def update(self):
        print("Aaaaaaaaaaahhhh!")