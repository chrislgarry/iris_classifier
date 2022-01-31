"""Neural network layer implementations"""
import numpy as np
import layer.layer_math as lm
from layer.layer import Layer


class Dense(Layer):
    """A dense/fully connected layer"""
    def __init__(self, input_size, output_size):
        self.bias = np.random.rand(1, output_size)
        self.weights = np.random.rand(input_size, output_size)

    def forward_propagation(self, features):
        self.input = features
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class Activation(Layer):
    """Apply an activation function to each input for the next layer"""
    #TODO: If adding more activation functions, extract to config file
    def __init__(self, activation):
        if activation == 'tanh':
            self.activate = lm.tanh
            self.activate_dx = lm.tanh_dx
        else:
            raise Exception('Invalid activation function')

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activate(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # learning_rate is overloaded and not used for activation
        return self.activate_dx(self.input) * output_error