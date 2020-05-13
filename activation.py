import numpy as np
import math
"""
Activation functions
Inputs: x a number
Output: [activation, gradient]
"""

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    gradient = sig * (1 - sig)
    return sig, gradient

def relu(x):
    rel = np.maximum(0, x)
    gradient = np.where(x > 0, 1, 0)
    return rel, gradient
