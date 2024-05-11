# Multiclass Classification

We have use CNN to solve binary classification problem, and this week we will cover multiclass classification.

## Learning Objectives

- Build a multiclass classifier for the Sign Language MNIST dataset

- Learn how to properly set up the ImageDataGenerator parameters and the model definition functions for multiclass classification

- Understand the difference between using actual image files vs images encoded in other formats and how this changes the methods available when using ImageDataGenerator

- Code a helper function to parse a raw CSV file which contains the information of the pixel values for the images used


## Code Required for Multiclass Model

### For the image data generator the below code is used

```python
train_datagen = ImageDataGenerator(rescale=1./255)

train_datagenerator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (300,300),
    batch_size = 128,

    # will be change from binary to categorical
    class_mode = 'categorical'
)
```

### For building the cnn model the below code is used

```python

tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3) activation='relu', input_shape(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (3,3) activation='relu', ),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (3,3) activation='relu', ),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    # Output layer is changed from 1 to 3 which are 3 classes
    # Rock, Paper, Scissors
    tf.keras.layers.Dense(3, activation='softmax'),
])
```

### For compiling the model

The following code can be used:

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(
    # this will be changed binary to categorical
    # sparse is another alternative
    loss='categorical_crossentropy',
    optimizer=RMSprop(lr= 0.001),
    metrics=['acc']
)
```


