import numpy as np

class CrossEntropy:
    def forward(self, y_real: list[int], y_pred: list[float]) -> None:
        self.y_pred = y_pred
        self.y_true = y_real
        self.outputs = []
        for i in range(len(y_real)):
            self.outputs.append(-np.sum(y_real[i]*np.log(y_pred[i] + 1e-15)))

        self.loss_mean = np.mean(self.outputs)

    def gradient(self):
        self.grad = np.array(self.y_pred) - np.array(self.y_true)