# Real-World Time Series Data

The following section will focus on adding cobvolutions on top of DNNs and RNNs, and using real-world time series data. In particular, time series data for measuring sunspot activity over hundards of years, and try to predict using it

## Convolutions

Combining convolutions with LSTM to get a nicely fitting data.

The following is the LSTM code that was considered in week-3 of the course.

```python
model = tf.keras.models.Sequential([

    # added 1D convolution layer
    # will take 5 number window
    # and it mutliply it by 32 filters (same as convolution)
    # 
    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        strides=1,
        padding="casual",
        activation="relu",
        input_shape=[None, 1] # specified input shape, not Lambda
    ),

    # 2 unidirectional lstm layers
    tf.keras.layers.LSTM(32, return_seqeunces=True),
    tf.keras.layers.LSTM(32),

    # output layer
    tf.keras.layers.Dense(1),

    tf.keras.layers.Lambda(lambda x: x * 200)
])

# defining optimizer here
# the learning rate can be found using callbacks
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)

# compiling the model
model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=optimizer,
    metrics=["mae"]
    )

# training the model
model.fit(dataset, epochs=500)

```

The following code to generate windowed dataset

```python

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    # create dataset object from numoy series
    ds = tf.data.Dataset.from_tensor_slces(series)


    # create windows subsets of the dataset
    # shift each window by 1
    # if subset windows have les size than window + 1
    # then drop the subset
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)

    # flatten the dataset into batches of size window_size + 1
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))

    # shuffle the dataset given the shuffle buffer
    ds = ds.shuffle(shuffle_buffer)

    # will create tuple of features and labels
    ds = ds.map(lambda w: (w[:-1], w[-1]))

    # return the dataset in the specfied batch_size
    # prefetch 1 batch to memory to speed up
    return ds.batch(batch_size).prefetch(1)
```