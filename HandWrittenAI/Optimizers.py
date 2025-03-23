import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.decay = decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iteration = 0
        self.m_weights = {}
        self.v_weights = {}
        self.m_biases = {}
        self.v_biases = {}

    def update(self, layer):
        layer_id = id(layer)

        if self.decay:
            self.lr = self.initial_lr / (1 + self.decay * self.iteration)

        if layer_id not in self.m_weights:
            self.m_weights[layer_id] = np.zeros_like(layer.weights)
            self.v_weights[layer_id] = np.zeros_like(layer.weights)
            self.m_biases[layer_id] = np.zeros_like(layer.biases)
            self.v_biases[layer_id] = np.zeros_like(layer.biases)

        dw, db = layer.dweights, layer.dbiases

        mw = self.m_weights[layer_id]
        vw = self.v_weights[layer_id]
        mb = self.m_biases[layer_id]
        vb = self.v_biases[layer_id]

        mw[:] = self.beta1 * mw + (1 - self.beta1) * dw
        vw[:] = self.beta2 * vw + (1 - self.beta2) * (dw ** 2)
        mb[:] = self.beta1 * mb + (1 - self.beta1) * db
        vb[:] = self.beta2 * vb + (1 - self.beta2) * (db ** 2)

        t = self.iteration + 1
        mw_corr = mw / (1 - self.beta1 ** t)
        vw_corr = vw / (1 - self.beta2 ** t)
        mb_corr = mb / (1 - self.beta1 ** t)
        vb_corr = vb / (1 - self.beta2 ** t)

        layer.weights -= self.lr * mw_corr / (np.sqrt(vw_corr) + self.epsilon)
        layer.biases -= self.lr * mb_corr / (np.sqrt(vb_corr) + self.epsilon)

        self.iteration += 1