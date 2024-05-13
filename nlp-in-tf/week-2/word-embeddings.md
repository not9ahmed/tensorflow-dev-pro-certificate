# Word Embedddings

This week will cover Embeddings, where token are mapped as vectors in a high dimension space. With labeled examples and embeddings, the vectors can be configured so that words with similar meaning will have similar direction in vector space.

This will start the process of training a neural network to learn sentiment in text. The focus will be on movie reviews labeled "positive" and "negative", and determine the words that drive the meaning in sentence.

## Introduction

Instead of representing the words as number from 1 to 10,000, we will have a better way to represent words.

Before we looked at Tokenizing words which is converting a sentence into a sequence of numbers, where the number if value of key-pair. Convert for example TensorFlow into "9".

Using TensorFlow I was able to process a strings to get indices of words in corpus of string, then convert them into a matrices of numbers.

## IMDB Datasets


### Loading the Datasets

The following code will load the IMDB dataset before training the model.

```python
import tensorflow as tf
print(tf.__version__)

import tensorflow_datasets as tfds

# loading the dataset from api
# along with meta data
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)


import numpy as np

# 25k for train and 25K for test
# data type of each row is tf.Tensor()
train_data, test_data = imdb['train'], imdb['test']


training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []


# Loop over all training examples and save sentences and labels
# 25K rows
# sentence, label
for s, l in training_data:
    
    # values are tensors so it will be convered to numpy arrays
    training_sentences.append(s.numpy().decode('utf8')) 
    training_labels.append(l.numpy()) 


# Loop over all training examples and save sentences and labels
# 25K rows
# sentence, label
for s, l in test_data:

    # values are tensors so it will be convered to numpy arrays
    testing_sentences.append(s.numpy().decode('utf8')) 
    testing_labels.append(l.numpy()) 


# converting list of labels into numpy arrays
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

```

### Tokenizing the Sentences

```python
vocab_size = 10000
embeddinng_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"


# importing the required libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequence

# defining the tokenizer
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

# feeding the training sentences into the tokenizer
tokenizer.fit_on_texts(training_sentences)

# getting the word index from tokenizer
word_index = tokenizer.word_index

# creating sequences
sequences = tokenizer.text_to_sequences(training_sentences)

# padding the sequences
padded = pad_sequence(sequences, maxlen = max_length,  truncating = trunc_type)


## For Testing ##

# creating sequences for testing_sentences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequence(testing_sequences, maxlen = max_length)

```

### Defining the Neural Network Model 

Thw following code will showcase an example of neural network to handle text.

```python
model = tf.keras.Sequential([

    # defining kayer to handle the text embedding
    # result of embedding will be 2D array
    # with length of sentence and embedding dimension as size
    tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        input_length = max_length),


    # In NLP different Flatten layer used which is Global Average Pooling 1D
    #tf.keras.layers.Flatten(),

    # averages across vector to flatten it out
    # simpler and it's faster then flatten
    tf.keras.layers.GlobalAveragePooling1D(),


    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

```

### Training and Compiling the Model

The following code will compile and train the model

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


num_epochs = 10

model.fit(
    # passing the padded sentences to train the model
    padded,

    # passing the training dataset labels
    training_labels_final,

    # defining the number of epochs
    epochs = num_epochs,

    # passing the testing dataset and their labels 
    validation_data = (testing_padded, testing_labels_final)
)


```


### Demonstrating the Embeddings

The following code will d
```python

e = model.layers[0]

weights = e.get_weights()[0]

# shape: (vocab_size, embedding_dim)
print(weights.shape)
# (10000, 16)


# the following is the reverse of word_index
# to able to plot, word index should be reversed
# hello : 1 
reverse_word_index = tokenizer.index_word
# 1: hello

```

###  Vectors and their Metadata

The following code with write vectors and their metadata out of files

```python
import io

# opening the vectors and their metadatafile
out_v = io open('vecs.tsv', 'w', encoding='utf-8')
out_m = io open('meta.tsv', 'w', encoding='utf-8')

# looping over the vocab_size=10,000

for word_num in range(1, vocab_size):


    word = reverse_word_index[word_num]
    embeddings = weights[word_num]

    # for metadat we just write the words
    out_m.write(word + "\n" )

    # write value of each items in array of embeddings
    # coeffcient of each dimension on the vector for word
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")


out_v.close()
out_m.close()

```

## How Embedding Work

Words of sentence, and word that have similar meaning are close to each other.

Word found togther have similar vector, over time the words will begin to cluster. The meaning of the word will come from the label of the sentence.

As neural network starts training, it will learn vectors and associate them with labels and will come up with embedding.


## Sarcasm Dataset Classifier

### Extracting the dataset and splitting into training and testing set

```python

import json
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Defining the constant and parameters
vocab_size = 10000
embedding_dim = 16
max_length = 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV"
training_size = 20000

# opening the file and loading the json into datastore json object
with open ("./sarcasm.json", "r") as f:
    datastore = json.load(f)


sentences = []
labels = []


for item in datastore:
    sentences.append(datastore['headline'])
    labels.append(datastore['is_sarcastic'])


# For splitting the sentences into training and validation
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

# For splitting the labels into training and validation
testing_labels = labels[0:training_size]
training_labels =labels[training_size:]

```


### Defining the Tokenizer and Sequence the Sentences

```python
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# defining training sequences
training_sequences = tokenizer.texts_to_sequences(training_sentences)

# padding the training sequences
training_padded = pad_sequences(
    training_sequences,
    maxlen = max_length,
    padding = padding_type
    truncating = trunc_type)


# defining testing sequences
testing_sequences = tokenizer,texts_to_sequences(texting_sentences)

# padding the testing sequences
testing_padded = pad_sequences(
    testing_sequences,
    maxlen = max_length,
    padding = padding_type
    truncating = trunc_type)

```

### Defining the Neural Network

The following code will handle the definition of the neural network model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        input_length=max_length),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
```

### Training The Model

The following code will handle the training of the model.

```python
history = model.fit(
    training_padded,
    training_labels,
    epochs = num_epochs,
    validation_data = (testing_padded, testing_labels),
    verbose=2
)
```

### Plotting the Result

```python
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history, history[string])
    plt.plot(history, history['val_' + string])

    plt.xlabel("Epochs")
    plt.ylabel("Epochs")

    plt.legend([string, 'val_'+string])
    plt.show

plot_graphs(history, "acc")
plot_graphs(history, "loss")
```