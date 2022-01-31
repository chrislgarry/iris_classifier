"""The Layer base class for implementing different network layers."""
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward_propagation(self, input):
        pass

    @abstractmethod
    def backward_propagation(self, output_error, learning_rate=None):
        pass
