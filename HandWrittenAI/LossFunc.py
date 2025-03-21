import numpy as np

class CrossEntropy:
    def forward(self, y_real: list[int], y_pred: list[float]) -> None:
        self.outputs = []
        for i in range(len(y_real)):
            self.outputs.append(-np.sum(y_real[i]*np.log(y_pred[i])))

    def gradient(self):
        print("IDK...")

