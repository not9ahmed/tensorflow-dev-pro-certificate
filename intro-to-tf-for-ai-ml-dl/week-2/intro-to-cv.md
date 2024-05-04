# Introduction to Computer Vision

The following week will cover computer vision problems with deep learning.


## Writing Code to Load Training Data

```python
import tensorflow as tf
from tensorflow import keras

# call keras datasets api to get fashion_mnist dataset
mnist = tf.keras.datasets.fashion_mnist

# return 4 lists of training images and their labels
# and test images and their labels
# label will be a number
# 1. computers do better with number than text
# 2. reduce bias to language, as th number can be shared across languages. label 9 is boot in english, arabic etc
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


```

## Coding a Computer Vision Neural Network

A neural network with 3 layers.

```python
model = keras.Sequential([

    # flatten layer with 28X28 cuz images are of the shapes
    # then will be transformed into linear array
    keras.layers.Flatten(input_shape=(28, 28)),

    # hidden layer with 128 neurons/units in it
    # like variables in function x_1, x_2
    keras.layers.Dense(128, activation=tf.nn.relu),

    # 10 units/neurons cuz we have 10 classes of the dataset
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```