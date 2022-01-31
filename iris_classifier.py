"""Train a neural network on the Iris classification dataset and 
classify Iris gardener input manually
"""
import numpy as np
import pandas as pd
import re
from layer.layers import Activation, Dense
from network.network import NeuralNetwork
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from time import sleep

if __name__ == "__main__":
    # Load, encode, and normalize dataset
    label_encoder = LabelEncoder()
    dataset = pd.read_csv('dataset/Iris data.txt', header=None)
    dataset.columns = ['Sepal length', 'Sepal Width', 'Petal length',
                       'Petal width', 'Label']
    dataset['Label'] = label_encoder.fit_transform(dataset['Label'])
    features = dataset.loc[:, 'Sepal length': 'Petal width']
    features = features.values.reshape(150, 1, 4)
    labels = dataset['Label'].to_numpy()

    # Due to small size, sample dataset classes evenly for training and testing
    x_train, x_test, y_train, y_test = train_test_split(features, labels, 
                                                        test_size=0.1,
                                                        random_state=42,
                                                        stratify=labels)

    # Build the neural network
    nn = NeuralNetwork()
    nn.add_layer(Dense(4, 3))
    nn.add_layer(Activation('tanh'))
    nn.add_layer(Dense(3, 1))

    # Train the network and prompt user input for prediction
    nn.train(x_train, y_train, epochs=5000, learning_rate=0.01)
    print("\n===============Train/Test Info===============")
    print(f"Train instances by class: {np.bincount(y_train)}")
    print(f"Test instances by class: {np.bincount(y_test)}")

    # Print Gardener's validation samples for user to select from
    print("\n===============Gardener's Test Data===============")
    for flower in x_test:
        print("Gardener has logged an Iris measurement: " + 
              re.sub('\[|\]', '', np.array2string(flower, separator=',')))
        sleep(0.1)
    print("\n\nIRIS LEGEND: Iris Setosa: 0, Iris Versicolor: 1, Iris Virginica: 2\n")

    # Process user input and make prediction
    while True:
        gardener_input = input("Copy and paste a comma-separated entry"
                               " from the gardener's logs above to classify:\n")
        try:
            gardener_input = gardener_input.split(",")
            gardener_input = np.array([np.float64(i) for i in gardener_input])
            gardener_input = gardener_input.reshape(1, 1, 4)
        except Exception as e:
            print("Invalid input:", e)
            continue
        prediction = np.array2string(abs(np.around(nn.predict(gardener_input))))
        print(re.sub('\[|\]|\.', '', prediction))
