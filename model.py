import numpy as np
from loss import squared_error
from optimizer import stochastic_gradient_descent
from utils import batch_iterator

class Sequential():

    def __init__(self, optimizer=stochastic_gradient_descent()):
        self.layers = np.array([])
        self.optimizer = optimizer

    def add(self, layer):
        self.layers = np.append(self.layers, layer)

    def predict(self, x):
        res = x
        for layer in self.layers:
            res = layer.predict(res)
        return res

    def predict_in_train(self, x):
        pre_activation_arr = []
        activations_arr = []
        activation_gradient_arr = []
        for layer in self.layers:
            pre_activation, activation, activation_gradient = layer.predict_in_train(x)
            x = activation
            pre_activation_arr.append(pre_activation)
            activations_arr.append(activation)
            activation_gradient_arr.append(activation_gradient)
        return pre_activation_arr, activations_arr, activation_gradient_arr

    def fit(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, epochs: np.int, batch_size=16, _print=False, history=False):
        train_history = []
        test_history = []
        loss_history = []
        for i in range(epochs):
            if _print or history:
                acc_train = self.binary_accuracy(x_train, y_train)
                acc_test = self.binary_accuracy(x_test, y_test)
                train_loss = self.loss(x_train, y_train)
            if _print:
                print(f'Befor epoch {i + 1}:')
                print(f'Accuracy on train set: {round(acc_train * 100, 2)} %')
                print(f'Accuracy on test set: {round(acc_test * 100, 2)} %')
                print(f'Loss: {round(train_loss, 2)}')
            if history:
                train_history.append(acc_train)
                test_history.append(acc_test)
                loss_history.append(train_loss)

            for x_batch, y_batch in batch_iterator(x_train, y_train, batch_size, stochastic=True):
                self.layers = self.optimizer.optimize(self, x_batch, y_batch)
        return train_history, test_history, loss_history

    def binary_accuracy(self, x, y):
        y_hat = self.predict(x)
        count = 0
        for _y, _y_hat in zip(y, y_hat):
            if _y > 0.5 and _y_hat > 0.5 or _y <= 0.5 and _y_hat <= 0.5:
                count += 1
        return count / y_hat.shape[0]

    def loss(self, x, y):
        y_hat = self.predict(x)
        loss = self.optimizer.get_loss()(y, y_hat)[0]
        return loss.T.dot(loss)[0][0]

    def get_layers(self):
        return self.layers

    def set_layers(self, layers):
        self.layers = layers
