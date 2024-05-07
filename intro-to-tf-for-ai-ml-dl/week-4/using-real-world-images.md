# Using Real World Images

Addressing the shortcomings when data are basic, and when images are larger. Also, when images are not in place. This chapter will cover complex images with CNN.

## Learning Objectives

- Reflect on the possible shortcomings of your binary classification model implementation
- Execute image preprocessing with the Keras ImageDataGenerator functionality
- Carry out real life image classification by leveraging a multilayer neural network for binary classification

## Understanding ImageDataGenerator

The limitations before were the images datasets were very uniformed where the subject in center, and images size were small 28X28.

For datasets before they were split into training and test subset for us. However, it's not always the case and have to be done by ourselves.

The ImageDataGenerator can be pointed to training directory and will have labels based on directory name, and all the images inside the directory will be autolabeled.

The same can be done for validation dataset.
![image of ImageDataGenerator](images/ImageDataGenerator.png)



#### For Training

```python
from tensorflow.keras..preprocessing.image import ImageDataGenerator

# rescale to normalize the data
train_datagen = ImageDataGenerator(rescale=1./255)


# to load images from directory and subdirectory
# images-dataset/training/image.png
# so it should be pointed at images-dataset/training

# name of directory will be the label
train_generator = train_datagen.flow_from_directory(
    
    # the root directory of images/training dataset
    train_dir,

    # images should all be same size
    # this will line will resize the dataset
    target_size=(300,300),

    # images loaded for training and validation in batches
    # for better performance
    batch_size=128,

    # because it's binary classification
    class_mode='binary'
)
```

#### For Validation

```python

# rescale to normalize the data
test_datagen = ImageDataGenerator(rescale=1./255)


# name of directory will be the label for test
validation_generator = test_datagen.flow_from_directory(
    
    # the root directory of images dataset
    validation_dir,

    # images should all be same size
    # this will line will resize the dataset
    target_size=(300,300),

    # images loaded for training and validation in batches
    # for better performance
    batch_size=128,

    # because it's binary classification
    class_mode='binary'
)
```


## Defining ConvNet to Use Complex Images

Convolutional Neural Networks to classify horses vs human

The following model has 3 Sets of Convolutional + Pooling Layers, Because of higher complexity of images

```python
model = tf.keras.models.Model([



    # 1 convolutional layer with 16 filters
    # each filter of size (3X3)
    # relu activation
    # input shape (300, 300, 3) => (width, height, colors)
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),

    # 1 max pooling with (2,2)
    tf.keras.layers.MaxPooling2D(2,2)


    # 2 convolutional layer with 32 filters
    # each filter of size (3X3)
    # relu activation
    # input shape (300, 300, 3) => (width, height, colors)
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 3)),

    # 2 max pooling with (2,2)
    tf.keras.layers.MaxPooling2D(2,2)



    # 2 convolutional layer with 64 filters
    # each filter of size (3X3)
    # relu activation
    # input shape (300, 300, 3) => (width, height, colors)
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(300, 300, 3)),

    # 2 max pooling with (2,2)
    tf.keras.layers.MaxPooling2D(2,2)


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    # output layer change
    # 1 unit and sigmoid activation because it's binary
    tf.keras.layers.Dense(1, activation='sigmoid'),

])

```

## Training the ConvNet

For compiling the model an defining the loss, optimizer and metrics

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(
            # it is binary classification problem
            loss='binary_crossentropy',

            # RMSprop is type of gradient descent instead of adam
            # lr is the learning rate
            optimizer=RMSprop(lr=0.001),
            metrics=['accuracy']
            )
```

For training the model the below code used:

```python
history = model.fit(

    # streams images from the training directory
    # 1024 images from training
    train_generator,

    # batch size was 128 from train_generator
    # so they will be loaded 8 times
    # 128 * 8 => 1024 images (total)
    step_per_epoch=8,

    # going through the dataset
    epochs=15,

    # data from validation_generator which loads images
    # from the validation dataset
    # 256 images total
    validation_data=validation_generator,

    # 256 images and want to be handled in batches of 32
    # 32 * 8 = 256 images (total)
    validation_steps=8,

    # how much to display while training
    # hide training epoch progress
    verbose=2
)
```


Following code can be found below which is Google colab code to upload image, then making prediction for the uploaded image

```python
import numpy as np
from google.colab import files
from tensorflow.keras.utils import load_img, img_to_array

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = load_img(path, target_size=(300, 300))
  x = img_to_array(img)
  x /= 255
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
    
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
```


