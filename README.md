# An Iris dataset classifier example using neural networks!

## Dependencies
This repository has been tested with Python version 3.9. It is recommend to use [venv](https://docs.python.org/3/library/venv.html) to install it and avoid cluttering your system.
1. After creating a Python 3.9 venv, clone this repo into your venv and install the required dependencies using the requirements.txt file: 
```
pip3 install -r ./requirements.txt
```

## Train and validate the network
1. Execute the ```iris_classifier.py``` script from command line to load the dataset and execute training. The epoch along with error will be printed to console, and you will be prompted to input an Iris sample to classify from the Gardener's logbook which is also printed to console.
```
python3 iris_classifier.py
```
If you would like to change the training and validation dataset sizes, in iris_classifier.py modify the ```train_test_split``` size. The number of epochs and learning rate can also be modified in this file.

## Example Output
```
2022-01-31 00:51:09,407 [INFO] epoch 4985/5000   error=0.058230
2022-01-31 00:51:09,414 [INFO] epoch 4986/5000   error=0.058230
2022-01-31 00:51:09,420 [INFO] epoch 4987/5000   error=0.058230
2022-01-31 00:51:09,426 [INFO] epoch 4988/5000   error=0.058230
2022-01-31 00:51:09,433 [INFO] epoch 4989/5000   error=0.058230
2022-01-31 00:51:09,439 [INFO] epoch 4990/5000   error=0.058230
2022-01-31 00:51:09,446 [INFO] epoch 4991/5000   error=0.058230
2022-01-31 00:51:09,452 [INFO] epoch 4992/5000   error=0.058230
2022-01-31 00:51:09,459 [INFO] epoch 4993/5000   error=0.058230
2022-01-31 00:51:09,465 [INFO] epoch 4994/5000   error=0.058230
2022-01-31 00:51:09,471 [INFO] epoch 4995/5000   error=0.058230
2022-01-31 00:51:09,478 [INFO] epoch 4996/5000   error=0.058230
2022-01-31 00:51:09,484 [INFO] epoch 4997/5000   error=0.058230
2022-01-31 00:51:09,491 [INFO] epoch 4998/5000   error=0.058230
2022-01-31 00:51:09,497 [INFO] epoch 4999/5000   error=0.058230
2022-01-31 00:51:09,503 [INFO] epoch 5000/5000   error=0.058230

===============Train/Test Info===============
Train instances by class: [45 45 45]
Test instances by class: [5 5 5]

===============Gardener's Measurements===============
Gardener has logged an Iris measurement: 6.6,2.9,4.6,1.3
Gardener has logged an Iris measurement: 6.1,2.6,5.6,1.4
Gardener has logged an Iris measurement: 6.5,3. ,5.2,2. 
Gardener has logged an Iris measurement: 5.6,2.5,3.9,1.1
Gardener has logged an Iris measurement: 7.3,2.9,6.3,1.8
Gardener has logged an Iris measurement: 4.4,3.2,1.3,0.2
Gardener has logged an Iris measurement: 4.4,3. ,1.3,0.2
Gardener has logged an Iris measurement: 5. ,3.4,1.5,0.2
Gardener has logged an Iris measurement: 6.4,2.8,5.6,2.2
Gardener has logged an Iris measurement: 6.6,3. ,4.4,1.4
Gardener has logged an Iris measurement: 5.8,4. ,1.2,0.2
Gardener has logged an Iris measurement: 6.5,3. ,5.5,1.8
Gardener has logged an Iris measurement: 6.4,3.2,4.5,1.5
Gardener has logged an Iris measurement: 6.7,3. ,5. ,1.7
Gardener has logged an Iris measurement: 5.2,3.4,1.4,0.2


IRIS LEGEND: Iris Setosa: 0, Iris Versicolor: 1, Iris Virginica: 2

Copy and paste a comma-separated entry from the gardeners logs above to classify:
6.6,2.9,4.6,1.3
1
Copy and paste a comma-separated entry from the gardeners logs above to classify:
6.1,2.6,5.6,1.4
2

```

### Notes
 This software was developed and tested on MacOS 11.5.02, Python version 3.9.10
