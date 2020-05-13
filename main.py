import numpy as np
from model import Sequential
from layer import Dense
from activation import relu
from utils import batch_iterator
from dataset_reader import get_dataset

np.random.seed(42)

train_x, train_y, = get_dataset("DATA", "train")
test_x, test_y = get_dataset("DATA", "test")

nn = Sequential(learning_rate=0.25)
nn.add(Dense(50))
nn.add(Dense(2))
nn.add(Dense(1))

nn.fit(train_x, train_y, test_x, test_y, 500, batch_size=64, _print=True)
