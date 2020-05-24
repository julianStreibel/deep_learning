import numpy as np
from activation import sigmoid


class Dense:
    """
    Dense layer of Neurons:
    n: number of neurons in the layer
    input_shape: shape of the input data
    """
    def __init__(self, n, activation=sigmoid, input_shape=False):
        self.n = n
        self.activation = activation
        if input_shape:
            self.weights = np.random.rand(input_shape, n + 1)
        else:
            self.weights = False

    def pre_activation(self, x):
        x_c = np.append(x, np.ones((x.shape[0], 1)), axis=1)  # bias
        if self.weights is False:
            self.weights = np.random.rand(x_c.shape[1], self.n)
        return np.dot(x_c, self.weights)

    def predict(self, x):
        return self.activation(self.pre_activation(x))[0]

    def predict_in_train(self, x):
        pre_act = self.pre_activation(x)
        activation, activation_gradient = self.activation(pre_act)
        return pre_act, activation, activation_gradient

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
