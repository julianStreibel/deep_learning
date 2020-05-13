import numpy as np 

class stochastic_gradient_descent():
    def __init__(self, learning_rate: np.float, loss: callable):
        self.learning_rate = learning_rate
        self.loss = loss

    def optimize(self, model, x: np.array, y: np.array):
        self.pre_activation, self.activations, self.activation_gradient = model.predict_in_train(x)
        layers = model.get_layers()
        self.weights = [l.get_weights() for l in layers]
        self.m = len(layers)
        self.activations.insert(0, x)

        # error terms
        error_terms = [None for _ in range(self.m)]
        for k in range(self.m):
            i = self.m - k
            if k == 0:
                error_terms[i - 1] = self.loss(y, self.activations[i])[1]
            else:
                error_terms[i - 1] = self.activation_gradient[i- 1] * np.dot(error_terms[i], self.weights[i][:-1].T)

        # partial derivatives
        partial_derivatives = [None for _ in range(self.m)]
        for i in range(self.m):
            partial_derivatives[i] = self.activations[i][:, :, np.newaxis] * error_terms[i][:, np.newaxis, :]

        # average for total gradients
        total_gradients = [None for _ in range(self.m)]
        for i in range(self.m):
            total_gradients[i] = np.average(partial_derivatives[i], axis=0)

        new_weights = self.weights
        for i in range(self.m):
            layers[i].set_weights(np.append(self.weights[i][: -1] - self.learning_rate * total_gradients[i], self.weights[i][-1:], axis=0))
                
        return layers