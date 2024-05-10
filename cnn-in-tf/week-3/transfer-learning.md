# Transfer Learning

A way to use others trained model on larger dataset, and it can be used directly or use the features they have learned and apply them to my use case.

## Learning Objectives

- Master the keras layer type known as dropout to avoid overfitting

- Achieve transfer learning in code using the keras API Code a model that implements Keras’ functional API instead of the commonly used Sequential model

- Learn how to freeze layers from an existing model to successfully implement transfer learning

- Explore the concept of transfer learning to use the convolutions learned by a different model from a larger dataset

## Transfer Learning Concepts

The idea behind transfer learning is to take exisiting pre trained model which was trained on a very huge amounts of data. Then, use it to extract features from my own data. The later layers can be dense layeer which is defined by myself.

Layers can be frozen in order to not train specific layers from the model. The following week subject will focus on Inception CNN.

It can be seen that the first layer is extract the most common features and as I go through the model the layers will extract specific features.

## Inception Model

```python
import os
# to understand which layers to use, and which one to retrain
from tensorflow.keras import layers
from tensorflow.keras import Model

# copy of the model weight are stored below
# snapshot of the trained model
# https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels


# in order to use inception
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),

    # ignore the fully connected layers on top
    # and get straight to convolutions
    include_top=False,

    # don't want to use built in weight
    weights=None
)

pre_trained_model.load_weights(local_weights_file)

# to lock the model layers and make them non trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()
```