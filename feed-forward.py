import numpy as np

np.random.seed(98)


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = 0.1 * np.random.randn(input_size, output_size)
        self.bias = np.zeros(output_size)

    def forward(self, X):
        return np.dot(X, self.weights) + self.bias


X = np.random.randn(4)
layer1 = DenseLayer(4, 10)
layer2 = DenseLayer(10, 2)

net1 = layer1.forward(X)
out = 1 / (1 + np.exp(-layer2.forward(net1)))
print(out)
