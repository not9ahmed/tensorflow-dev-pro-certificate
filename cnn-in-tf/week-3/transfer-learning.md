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

### All Layers, and it can be used to retrieve it

```python
# will fetch the output of a lot of convolutions
# of size 7X7
last_layer = pre_trained_model.get_layer('mixed7')

last_output = last_layer.output
```

### Defining new model which will take Inception last layer

The following model uses functional api instead of Sequential

```python
from tensorflow.keras.optimizers import RMSprop

# passing the inception mixed7 layer to flatten
x = layers.Flatten()(last_output)

x = layers.Dense(1024, activation='relu')(x)

# output layer
x = layser.Dense(1, activation='sigmoid')(x)

# defining the model
# will take the inception input layer
# then add layers definitions just created
model  = Model(pre_trained_model.input, x)

model.compile(
    optimizer=RMSprop(lr=0.0001),
    loss='binary_crossentropy',
    metrics=['acc']
)

# Add Data augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale = 1./255.,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)


# Getting the images form directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 20,
    class_mode = 'binary',
    target_size = (150, 150)
)
```

### Training/Fitting the model

```python
history = model.fit(
    train_generator,
    validation_data = validation_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_steps = 50,
    verbose = 2
)

```

## Dropout

Layers in neural networks can sometimes have similar weights and can impact each other which is leading to overfitting.

Neighbors not affecting each other to much, which results in removing overfitting.

**Intiuition:**
Can't rely on any one feature, so have to spread out weights across units.

Dropout is very important in CNN, as it can help the model in overcoming overfitting. Because there is not enough data.

Use probability for each layer, in which

**Downside:**  
- The cost function $J$ is not well defined, on every iteration a couple of nodes are killed.
- Added extra hyperparameter

So it's better to avoid it before, just to calculate the cost function $J$, then make sure it's decreasing over iterations.

Then turn dropout on, in order to overcome the overfitt problem.


### Dropout in Code

```python
from tensorflow.keras.optimizers import RMSprop

# passing the output of Inception model
x = layers.Flatten()(last_output)

# hidden layer with 1024 units
x = layers.Dense(1024, activation='relu')(x)

# added dropout layer
# between 0 and 1 fraction of units to drop
# which is 20% of units will be dropped
x = layers.Dropout(0.2)(x)

# output layer
x = layers.Dense(1, activation='sigmoid')(x)

# function api
model = Model(pre_trained_model.input, x)

# compiling the model
model.compile(
    optimizer = RMSprop(lr = 0.0001),
    loss = 'binary_crossentropy',
    metrics = ['acc']
)

```