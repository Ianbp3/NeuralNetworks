import numpy as np

class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int, lambda_l2=0) -> None:
        self.weights = np.random.randn(n_inputs,n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
        self.l2_lambda = lambda_l2

    def forward(self, inputs: list[int]) -> None:
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, inputs):
        inputs = np.array(inputs)
        self.dweights = np.dot(inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dweights += 2 * self.l2_lambda * self.weights

        self.dinputs = np.dot(dvalues, self.weights.T)

        return self.dinputs

    def update(self, learning_rate = 0.01):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

    def get_l2_loss(self):
        return self.l2_lambda * np.sum(self.weights ** 2)