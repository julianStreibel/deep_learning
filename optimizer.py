import numpy as np 
from loss import squared_error

class stochastic_gradient_descent():
    """
    learning_rate: learning rate for backpropagation
    loss: cost function for the model
    decay: weight for weight decay
    """

    def __init__(self, learning_rate=0.01, loss=squared_error, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.loss = loss
        self.decay = decay
        self.momentum = momentum
        self.velocity = None

    def optimize(self, model, x: np.array, y: np.array):
        self.pre_activation, self.activations, self.activation_gradient = model.predict_in_train(x)
        layers = model.get_layers()
        self.weights = [l.get_weights() for l in layers]
        self.m = len(layers)
        self.activations.insert(0, x)

        error_terms = [None for _ in range(self.m)]
        partial_derivatives = [None for _ in range(self.m)]
        total_gradients = [None for _ in range(self.m)]
        for k in range(self.m):
            i = self.m - k
            if k == 0:
                error_terms[i - 1] = self.loss(y, self.activations[i])[1]
            else:
                error_terms[i - 1] = self.activation_gradient[i- 1] * np.dot(error_terms[i], self.weights[i][:-1].T)

            partial_derivatives[i - 1] = self.activations[i - 1][:, :, np.newaxis] * error_terms[i - 1][:, np.newaxis, :]
            total_gradients[i - 1] = np.average(partial_derivatives[i - 1], axis=0)
            error_term_ave = np.average(error_terms[i - 1], axis=0)
            error_term_ave.shape = (1, error_term_ave.shape[0])
            weight_gradient = np.append(total_gradients[i - 1], error_term_ave, axis=0)
            if self.velocity == None:
                self.velocity = [0 for _ in range(self.m)]
            self.velocity[k] = self.momentum * self.velocity[k] - self.learning_rate * (weight_gradient + self.decay * self.weights[i - 1])
            layers[i - 1].set_weights(self.weights[i - 1] + self.velocity[k])
                
        return layers

    def get_loss(self):
        return self.loss