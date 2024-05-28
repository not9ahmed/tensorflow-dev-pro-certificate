# Recurrent Neural Networks for Time Series

This week will focus on using Recurrent Neural Networks (RNN) and Long Short Term Memeory Neural Networks (LSTM) on time series data.

## Intro

### Lambda Layer

Allows us to write aribitary code as a layer in the neural networks.

**For example:**  
Scaling data with explicit pre-processing step and then feed it to neural networks.

Instead, I can have Lambda Layer that implemented as layer in the neural network, that resends data, scales it. The preproecssing step is part of the neural network.

## Conceptual Overview

### Recurrent Neural Networks

It's a neural network that contains recurrent layers. 

- It is desgined to sequentially process a sequence of inputs

- Able to process al types of sequences

- It will be feed batches of sequences

- and It will output a batches of forecasts

- Shape of RNN will be 3D

- shape = [batch size, # time steps, # dims of inputs at each time stamp]
  - Univariate will be 1
  - Multivariate will be 2+

The following is the architecture of the Recurrent Neural Network to be used for time series.
![image of rnn](images/rnn.png)

### Recurrent Layer

#### Why Recurrent Neural Network they are called ?

because the values recur due to the ouput of cell. A one-step is being feed back into itself at the next time step.

**Helpful for determining states:**  
- As the location of word in sentance can determine it's semantics and meaning.
- Same for numeric value as the closer numbers have closer impact than far numbers

![image of recurrent layer](images/recurrent-layer.png)

## Shape of the Inputs to the RNN

### Example

**Window size of 30 timestamps, Batch is 4:**  

shape = [batch size, # time steps, # dims]

The shape will be 4 X 30 X 1

Each timestamp, the memory cell input will be a 4 X 1 Matrix.


$$
\text{Input} =
x_{0} = 
\begin{bmatrix}
    1 \\
    2 \\
    3 \\
    4 \\
\end{bmatrix}
$$

Series Dimensionality (Univariate=1)

$$
\text{Output} =
\vec{Y}_{0} = 
\begin{bmatrix}
    1 && 5 && 7\\
    2 && 6 && 8\\
    3 && 7 && 9\\
    4 && 8 && 8\\
\end{bmatrix}
$$

- If the memory cell contains 3 neurons
- Then the output matrix will be 4 batch size X 3 neurons

- Because batch size coming is 4 and it has 3 neurons

- So full output of layer will be three dimensional
  - shape = [batch size, # time steps, # dims of number units]
  - shape = [4 X 30 X 3]

**In Simple RNN**

State Output $H$ is just a copy of the output matrix $Y$

- $H_{0}$ is just a copy of $\vec{Y}_{0}$
- $H_{1}$ is just a copy of $\vec{Y}_{1}$, etc

At each timestamp the memory cell gets the current input and the previous output.

**In some cases:**

We want to input sequence but don't want output one
- So we want to get single vector for each instance in a batch

Can be done by
**Sequence to Vector RNN**
- Which is ignoring all outputs except the last

**In Keras it's the default**

- Use `returns_sequence=True` to return a sequence from RNN when creating the layer
- Needs to be done when stacking one LSTM over the other

## Outputing a Sequence

```python
model = keras.models.Sequential([

    # RNN Layers
    # will output a sequence to the other layer
    # input_shape=[None, 1] =>  (batch, timestamp, dims of inputs)
    # (can have any size,
    # None timestamp can handle sequences of any length
    # 1 cuz it's univariate sequence)
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),

    # another RNN layer with 20 units
    keras.layers.SimpleRNN(20),

    # output layer
    keras.layers.Dense(1)

])
```

## Lambda Layers

The following code will add new layers called Lambda

Allows us to do arbitary operations to effectively expand the functionality of TensorFlow's Keras.
Can be done in model definition itself.

```python
model = keras.models.Sequential([
    
    # helps in dimensionality
    # no need to change the window helper function from
    # (batch size, # timestamps)
    # to add the # series dims
    # expand the array by 1 dimension to be 3D
    # (batch size, # timestamps,  # series dims)
    # input_shape=[None] => can sequence of any length
    keras.layers.Lambda(lambda x: tf.expands_dims(x, axis=-1), input_shape=[None]),

    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),

    # another RNN layer with 20 units
    keras.layers.SimpleRNN(20),

    # output layer
    keras.layers.Dense(1)

    # scale up the output to help in training
    # default activation of RNN  is Tanh => -1<value<1
    keras.layers.Lambda(lambda x: x * 100.0)

])
```

## Adjusting the Learning Rate Dynamically

The following code to train RNN

```python
train_set = windowed_dataset(x_train, window_set, batch_size=128, shuffle_buffer=shuffle_buffer_size)

model = tf.keras.models.Sequential([

    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),

    tf.keras.layers.SimpleRNN(40, return_seqeunces=True),

    tf.keras.layers.SimpleRNN(40),

    tf.keras.layers.Dense(1),

    tf.keras.layers.Lambda(lambda x: x * 100.0),
])

# creating callback for learning rate
# every epoch it changes the epoch by little
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

# defining the optimizer of type stochastic gradient descent
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-6, momentum=0.9)


# compiling the model
# metric od type mean absolute error
model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=optimizer.
    metrics=["mae"]
)

# training the model
model.fit(train_set, epochs=100, callbacks=[lr_schedule])
```

