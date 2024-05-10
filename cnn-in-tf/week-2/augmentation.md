# Augmentation

To overcome the overfitting, we can collect more data and this will allow the model to become more generalized. However it's not always possible. So in this case we are going for Image Augmentation. It is is when you tweak the training set to increase the diversit of subjects it covers.

## Learning Objectives


- Recognize the impact of adding image augmentation to the training process, particularly in time

- Demonstrate overfitting or lack of by plotting training and validation accuracies

- Familiarize with the ImageDataGenerator parameters used for carrying out image augmentation

- Learn how to mitigate overfitting by using data augmentation techniques

## Introducing Augmentation

It is a way to generate new examples form existing training set withour overriding original data. For example rotating, flipping and adding effects to images.

**Overfitting:**

Very good at spotting something from dataset, but getting confused when seeing something does not match expectations.

### ImageDataGenerator was used for rescaling before

We have already done little augmentation with ImageDataGenerator class but only rescaling.

```python
train_datagen = ImageDataGenerator(rescale=1./255)
```

### Updated ImageDataGenerator to do image augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,

    # 0 to 180 degree to randomly rotate image
    rotation_range=40,

    # randomly move image around inside it's frame
    width_shift_range=0.2,
    height_shift_range=0.2,

    # skewing image around z axis
    shear_range=0.2,

    # zooming on image
    zoom_range=0.2,

    # flipping image
    horizontal_flip=True,

    # fill any pixels lost by operations
    # use neighbors of the pixel to keep uniformity
    flil_mode='nearest'
    )
```

## Augmentation Takeaway for Horses vs Humans Dataset

Image Augmentation will introduce radnom elemnent to training image, but if validation set does not have the same randomness.

The result will fluctuate a lot during the model training validation accuracy and loss

Broad set of images of training and test is required to overcome this problem.