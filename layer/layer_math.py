"""Helper functions to be used in neural network layers"""
import numpy as np


def tanh(x):
    """Compute the hyperbolic tangent"""
    return np.tanh(x)

def tanh_dx(x):
    return 1.0 - (np.tanh(x)**2)

def msqerr(actual, predicted):
    """Compute the mean squared error"""
    return np.mean((actual - predicted)**2)

def msqerr_dx(actual, predicted):
    return 2 * (predicted - actual) / actual.size