import numpy as np

class batch_iterator:
    """
    Iterrates over the data in batches
    X: list of x
    y: list of y
    batch_size: number of samples in a batch
    stochastic: random order of lines in [X, y]
    """

    def __init__(self, X: np.array, y: np.array, batch_size: np.float, stochastic=False):
        assert(X.shape[0] == y.shape[0])
        assert(batch_size > 0)
        if stochastic:
            rstate = np.random.RandomState(42)
            rstate.shuffle(X)
            rstate = np.random.RandomState(42)
            rstate.shuffle(y)
            self.X = X
            self.y = y
        else:
            self.X = X
            self.y = y
        self.batch_size = batch_size
        self.number_of_batches = np.ceil(X.shape[0] / batch_size)
        self.batch_number = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_number < self.number_of_batches:
            start = self.batch_number * self.batch_size
            stop = (self.batch_number + 1) * self.batch_size
            X_batch = self.X[start: stop]
            y_batch = self.y[start: stop]
            self.batch_number += 1
            return X_batch, y_batch
        else:
            raise StopIteration