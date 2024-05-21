# Sequence Models and Literature

The following week will focus on prediction the next words of a text from a given phrase.

## Text Prediction

- We can get a body of text and extract full vocabulary from it
- then create dataset from that
- where we make a phrase X an the next word in that phrase to be Y

## Code to Predict Text

```python
tokenizer = Tokenizer()

data = "In the town of Athy one Jeremy Langian \n Battered away ... ..."

# create python list of data then convert to lower text
corpus = data.lower().split("\n")

# dictionary of words in the overall corpus
# key => word
# value => token
tokenizer.fit_on_texts(corpus)

# to get the total number of words
total_words = len(tokenizer.word_index) + 1

```

### Code to Take Corpus and Turn It into Training Data

```python

# training Xs
input_sequences = []

for line in corpus:
    # generate token list
    # [ahme is cool] => [0 1 49]
    token_list = tokenizer.texts_to_sequences([line])[0]

    # iterate over list of tokens and create a number of
    # n gram of sequences
    # line: [1 2 3 4 5 6]
    # input sequence:
    # [1 2]
    # [1 2 3]
    # [1 2 3 4]
    # [1 2 3 4 5]
    # [1 2 3 4 5 6]
    for i in range(1, len(token_list)):

        n_gram_sequence = token_list[:i+1]

        input_sequences.append(n_gram_sequence)


# length of longest sentence in the corpus
max_sequence_len = max([len(x) for x in input_sequences])


input_sequence = np.array(

    # pre pad to easier extract the labels
    pad_sequences(
            input_sequences,
            maxlen= max_sequence_len,
            padding='pre'
        )
    )

    # line: [1 2 3 4 5 6]
    # padded input sequence:

    # (input         )(label)
    # (x             )(y)
    # [0 0 0 0 0 0 0 1 2]
    # [0 0 0 0 0 0 1 2 3]
    # [0 0 0 0 0 1 2 3 4]
    # [0 0 0 0 1 2 3 4 5]
    # [0 0 0 0 1 2 3 4 5 6]


# splitting the sentences into Xs and Ys

# all rows and for columns from start till before the end
xs = input_sequences[:,:-1]  

# all rows and for columns only the last column
labels = input_sequences[:,:-1]

# convert list to categorical
# will create one hot encode of the labels
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
```

### Building the Model

```python
# defining the model
model = tf.keras.Sequential()

# adding the Embedding layer
# to handle all the words => total_words
# 64 is number of dimensions to plot the vector for a word
# size of input dimension which is max_sequence_len
# it is -1 because we removed one for the label
model.add(Embedding(
    total_words,
    64,
    input_length=max_sequence_len - 1
    ))

# adding LSTM layer
# they carry context along with them
# 20 cell states
model.add((LSTM(20)))

# adding dense layer => output layer
# of size same as the one used in one hot encode
model.add(Dense(total_words, activation='softmax'))

# compiling the model
# and definind loss, optimizer, and accuracy
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# training the model
model.fit(xs, yx, epochs=200, verbose=1)
```


### Changing the Model to Bidirectional

The model above suffered from problem where the words get repeated

```python
model = Sequential()

model.add(Embedding(
    total_words,
    64,
    input_length=max_sequence_len - 1
    ))


# adding bidirectional layer
model.add(Bidirectional(LSTM(20)))


# adding the output layer with total_words as the total units
model.add(Dense(total_words, activation='softmax'))


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )


# training the modle
model.fit(xs, ys, epochs=500, verbose=1)
```


### Predicting a Word

Word is "Laurence went to Dublin"

```python
seed_text = "Laurence went to Dublin"

# will tokenize the text to sequences
# Laurence will be ignored, because it's not of corpus
# [134,  13, 59]
token_list = tokenizer.texts_to_sequences([seed_text])[0]


# will pad the sequence to match the ones in training set
# [0 0 0 0 134 13 59]
token_list = pad_sequences(
    [token_list],
    maxlen=max_sequence_len - 1,
    padding='pre'
    )

# will pass the padding to the model as prediction
predicted = model.predict(token_list)

# will give the token of the word most likely to be next in sequence
predicted = np.arrgmax(probabilities, axis = -1)[0]


# do reverse index to convert token back to word
output_word = tokenizer.index_word[predicted]

# adding the text to seed text
seed_text += " " + output_word



# the next code will do it for 10 times
seed_text = "Laurnce went to dublin"
next_words = 10


for _ in range(next_words):
    
    # convert text to sequences
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # padding the sequences
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding= 'pred'
        )
    
    # making prediction of the next token
    predicted = model.predicted_classes(token_list, verbose=0)

    # getting the word from index
    output_word = output_word = tokenizer.index_word[predicted]


    # adding the predicted word to seed text
    seed_text += " " + output_word

print(seed_text)
```


