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


```