import numpy as np
 
def squared_error(y: np.array, y_hat: np.array) -> np.float:
    n = y.shape[0]
    u = y - y_hat
    squared_error = 0.5 * np.square(u)
    derivative_squared_error = -u
    return squared_error, derivative_squared_error