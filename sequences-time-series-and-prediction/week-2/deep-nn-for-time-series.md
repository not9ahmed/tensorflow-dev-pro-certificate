# Deep Neural Network for Time Series

Teaching neural network to recognize and predict time series.

## Preparing Features and Labels

The number of values in the series will be treated as feature, the window size.

Will take a window of data and train the ML model to predict the next value.

**For example:**

- Use 30 examples of time series data at a time
- Use 30 values a feature and the next value as label
- Train the neural network on 30 values to predict the single label

The following code will create dataset

```python
# create dataset of range 1..10
dataset = tf.data.Dataset.range(10)

# shift=1 shift values by 1 each iteration
# drop_remainder to only have windows of 5 item
dataset = dataset.window(5, shift=1, drop_remainder=True)

# prepare  windows to be tensors instead of the Dataset structure
dataset = dataset.flat_map(lambda window: window.batch(5))

# split into everything but last one with -1
# split only last one with -1
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# shuffle data before training
# buffer_size a number equal or greater than total for better shuffling
dataset = dataset.shuffle(buffer_size=10)

# batching the data
# batch the data into sets of 2
# prefetch optimizes the execution time when model is already training
# Tensorflow will prepare next batch in advance while current consumed by model
dataset = dataset.batch(2).prefetch(1)

```

### Why Suffle Time Series Data?

To reduce sequence bias while training your model. 
- It's when neural network overfitting to the order of inputs
- So it will not perform well when it does not see that particular order when testing.
- You don't want the sequence of training inputs to impact the network this way so it's good to shuffle them up.

## Feeding Windowed Dataset into Neural Network

A number of input values on x typically called a window on data.


```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    '''
    data: data series
    window_size: size of window we want
    batch_size: size of batch to use when training
    shuffle_buffer: how data will be shuffled
    '''
    
    # create dataset from the series
    dataset = tf.data.Dataset.from_tensor_slices(series)


    # to slice data into windowed dataset
    # each one shifted by one
    # drop_remainder=True to make them all the same size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)


    # flatten data to make it easier to handle
    # and it will flattened into chunks of size of window_size+1
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # shuffle the windowed data
    # shuffle_buffer speeds it up
    # .map() splits data into xs and ys
    dataset = dataset.shuffle(shuffle_buffer)
    .map(lambda window: (window[:-1], window[-1]))

    # dataset is batched into seleteced batch size and returned
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset
```

## Single Neural Network

The following section will build a neural network, and feed it the windowed dataset

### Splitting the dataset into training and validation set

```python
split_time = 1000

time_train = time[:split_time]
x_train = series[:split_time]

time_valid= time[split_time:]
x_valid = series[split_time:]
```

### Creating Simple Neural Network

```python
# constants to pass to windowed_dataset function
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

# create windowed dataset
dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)

# create single dense layers
# input shape being window_size
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])

# define the model
model = tf.keras.Sequential([l0])


# compile the model
# loss of type mean squared error
# optimizer of type stochastic gradient descent
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=1e-6,
        momentum=0.9
    )
)

# training the model
model.fit(dataset, epochs=100, verbose=0)


# inspect the weights
# l0.get_weights() will get the weights of layer 0
print("layer weights {}".format(l0.get_weights()))
```

## Input Window Features and Labels Naming

Input window of 20 values wide:  
$x_{0}, x_{1}, x_{2}, ..., x_{19}$

- It's not the value of the horizontal x-axis
- It's the value of time series at that point on horizontal axis

The value at time 0 $t_{0}$ which is 20 steps before the current value is called $x_{0}$, and $t_{1}$ is called $x_{1}$

For the output, the value at the current time to be the y.


