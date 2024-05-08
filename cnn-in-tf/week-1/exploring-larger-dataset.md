# Exploring Larger Dataset

The following course will explore CNN but with larger dataset (Cats vs Dogs)

## Learning Objectives

- Gain understanding about Keras’ utilities for pre-processing image data, in particular the ImageDataGenerator class

- Develop helper functions to move files around the filesystem so that they can be fed to the ImageDataGenerator

- Learn how to plot training and validation accuracies to evaluate model performance

- Build a classifier using convolutional neural networks for performing cats vs dogs classification

## Training with the Cats vs Dogs Dataset


### The following code will be used to load the images from directory

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# for train generator
train_gen = ImageDataGenerator(rescale(1./255))

train_generator = train_gen.flow_from_directory(

    # will point at train directory
    train_dir,
    target_size=(150,150),

    # total is 2000 images
    # each batch will have 100 batches
    # 2000 images /20 batch size => 100 batches
    batch_size=20,
    class_mode='binary'
)


# for validation generator
validation_gen = ImageDataGenerator(rescale(1./255))

validation_generator = validation_gen.flow_from_directory(

    # will point at validation directory
    validation_dir,
    target_size=(150,150),

    # total is 2000 images
    # each batch will have 100 batches
    # 2000 images /20 batch size => 100 batches
    batch_size=20,
    class_mode='binary'
)
```

### The model architecture will be the following

```python
model  = tf.keras.models.Sequential([
    
    # conv layer with 16 filters of size 3X3
    # relu activation
    # input shape of (150 width, 150 height, 3 colors)
    tf.keras.layers.Conv2D(
        16,
        (3,3)
        activation='relu',
        input_shape=(150,150,3)
    ),
    tf.keras.layers.MaxPooling2D(2,2),


    # conv layer with 32 filters of size 3X3
    # relu activation
    # input shape of (150 width, 150 height, 3 colors)
    tf.keras.layers.Conv2D(
        32,
        (3,3)
        activation='relu',
    ),
    tf.keras.layers.MaxPooling2D(2,2),

    
    # conv layer with 64 filters of size 3X3
    # relu activation
    # input shape of (150 width, 150 height, 3 colors)
    tf.keras.layers.Conv2D(
        64,
        (3,3)
        activation='relu',
    ),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Flatten(),
    
    # hidden layer with 512 units
    tf.keras.layers.Dense(512, activation='relu'),

    # output layer with 1 unit
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()
```

### Model Compilation and setting Loss and Optimizer

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['acc']
)

```

### Model training

```python
history = model.fit(

    # passing train generator data as input
    train_generator,
    
    # same as number of batches
    # 2000 images/ 20 batch size = 100 number of batches
    steps_per_epochs=100,

    # passing validation generator data
    validation_data=validation_generator,

    # depends on validation size
    validation_steps=50,

    # display info when training
    verbose=2
)
```

### Observation

By cropping the image I can classify correctly a class

What about cropping the images during the training, what will be the impact?

