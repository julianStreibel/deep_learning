import pandas as pd
import numpy as np
import os

label_pos = 1
label_neg = 0

data_dim = 100
data_dim_with_bias = data_dim + 1

def read_dataset(src):
    """Reads a dataset from the specified path and returns input vectors and labels in an array of shape (n, 101)."""
    with open(src, 'r') as src_file:
        # preallocate memory for the data
        num_lines = sum(1 for line in src_file)
        data = np.empty((num_lines, data_dim_with_bias), dtype=np.float16)
        labels = np.empty((num_lines, 1), dtype=np.float16)

        # reset the file pointer to the beginning of the file
        src_file.seek(0)
        for i, line in enumerate(src_file):
            _, str_label, str_vec = line.split('\t')
            labels[i] = label_pos if str_label.split('=')[1] == "POS" else label_neg
            data[i,:data_dim] = [float(f) for f in str_vec.split()]
            data[i,data_dim] = 1
    return data, labels

def get_dataset(src_folder, name="train"):
    path = os.path.join(src_folder, "rt-polarity.{}.vecs".format(name))
    return read_dataset(path)