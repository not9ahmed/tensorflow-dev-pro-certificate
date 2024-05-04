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


## Using Callbacks to Control Training

Callbacks are a way to stop training when I reach certain amount in terms of loss an accuracy.

On every epoch call back to code function check metrics then if satisfied, then stop the training.

The below is example code of call back function

```python
# instantiate the class for callback
callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist

# loading up the dataset
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalizing the dataset
training_images = training_images/255.0
test_images = test_images/255.0


# defining the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layer.Dense(512, activation=tf.nn.relu),
    tf.keras.layer.Dense(10, activation=tf.nn.softmax)
])

# compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


# here I'm passing the callbacks into the model training
model.fit(training_images, training_labels, epochs=5, callbacks = [callbacks])
```

The callback function in class is the below
```python
class myCallback(tf.keras.callbacks.Callback):

    # it is called by the callback whenever the epoch ends
    # logs objects which contains info of current training
    # logs.get('loss') is part of the logs
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            print("\nLoss is low so canceling training!")
            self.model.stop_training = True

```