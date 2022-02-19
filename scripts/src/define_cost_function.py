import numpy as np
from create_sample_data.create_data import create_sample_regression_data


def h(X,w):

    """
    Define 1D linear regression function in the form:
    w_0 + w_1 * X = y
    """
    return (w[1] * np.array(X[:,0]) + w[0])

def cost(w,X,y):
    
    """
    Define cost_function for 1D linear regressor in the form:
    1/(2m) * sum((h(x(i)) - y(i))**2)
    """

    m = len(X[:,0])

    return (.5/m * np.sum(np.square(h(X,w) - np.array(y))))


