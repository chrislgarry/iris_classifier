"""Neural network class consisting of training, validation, and loss functions"""
import layer.layer_math as lm
import logging as log
import sys


class NeuralNetwork:
    log.basicConfig(
        level=log.INFO,
        encoding='utf-8',
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            log.FileHandler("training.log"),
            log.StreamHandler(sys.stdout)
        ]
    )

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, features, labels, epochs, learning_rate):
        num_samples = len(features)
        for epoch in range(epochs):
            current_error = 0
            for sample in range(num_samples):
                output = features[sample]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                current_error += lm.msqerr(labels[sample], output)
                total_error = lm.msqerr_dx(labels[sample], output)
                for layer in reversed(self.layers):
                    total_error = layer.backward_propagation(total_error, learning_rate)
            current_error /= num_samples
            log.info('epoch %d/%d   error=%f' % (epoch+1, epochs, current_error))

    def predict(self, samples):
        prediction = []
        for sample in samples:
            for layer in self.layers:
                sample = layer.forward_propagation(sample)
            prediction.append(sample)
        return prediction