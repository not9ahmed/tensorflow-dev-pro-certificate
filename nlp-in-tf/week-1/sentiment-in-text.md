# Sentiment in Text

First step of understanding the sentiment of text is the Tokenization of the text. The process is to convert text into numeric values, and the number repesenting character or text. This week will cover Tokenizer and pad_sequences APIs in TensorFlow, and how to use them to prepare and encode text and sentences to get them prepare to be feed into Neural Networks.

## Word Based Encoding

We can take character encoding, for example their ascii charcters. But it will not help with understanding the meaning.

### Encoding Based on Character Example

**LISTEN**   
Can be encoded into ascii letters, but the semantic of the words are not encoded into the letters. So it's not useful.

**SILENT**  
Has the same letters, but it has the opposite meaning.


### Encoding Based on Words Example

**I love my dog**   
Each word can be encoded into  
001 002 003 004

**I love my cat**  
Each word can be encoded into  
001 002 003 005

There is some similarity between the 2 sentences.

## Using APIs

The following code generating the dictionary of words encoded and creating vectors out of the sentences.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

# creating list of words
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
]

# creating instance of the Tokenizer object
# take top 100 words by volumes and encode them
# it is hyperparameter
tokenizer = Tokenizer(num_words = 100)

# takes data and encode it
tokenizer.fit_on_text(sentences)

# returns dictonary of key value pair
# key is word
# value is the token for for word
# it removes capital letters and special characters
word_index = tokenizer.word_index


print(word_index)
```

## Text to Sentence

```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?',
]

tokenizer = Tokenizer(num_words = 100)

# takes data and encode it
tokenizer.fit_on_texts(sentences)

# create dictionary of texts and token
word_index = tokenizer.word_index

# turn them into set of sequences
# can take any set of sentences 
# then encode based on word set passed in fit_on_texts
sequences = tokenizer.texts_to_sequences(sentences)


print(word_index)
print(sequences)


# for testing the text_to_sequences
test_data = [
    'i really love my dog',
    'my dog loves my manatee',
]

test_seq = tokenizer.texts_to_sequences()
print(test_seq)
```

Need a lot of data in order to have a broad set of vocabulary

In order to not have words, not incluced in the words index.

Instead of ignoring unseen word, but add special value when they're encountered.

```python
from tensorflow.keras.processing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing'
]

# the token 00V for out of vocabulary
# for words not in word_index
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# create dictionary of texts and token
word_index = tokenizer.word_index

# turn them into set of sequences
# can take any set of sentences 
# then encode based on word set passed in fit_on_texts
sequences = tokenizer.texts_to_sequences(sentences)


# for testing words not in the word_index
test_data = [
    'i really love my dog',
    'my dog loves my manatee',
]

test_seq = tokenizer.texts_to_sequences()
print(test_seq)
```

**For images**  
They need to be uniform in size, Generator was used to resize images to fit. So that they can be feed into a neural network.

**For Text**  
The also need some uniformity in size. In order, to be feed into neural network. <u>Padding</u> will be used for this

## Padding

```python
from tensorflow.keras.processing.text import Tokenizer
from tensorflow.keras.processing.sequence import pad_sequences


sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing'
]

# the token 00V for out of vocabulary
# for words not in word_index
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# create dictionary of texts and token
word_index = tokenizer.word_index

# turn senetences them into set of sequences
sequences = tokenizer.texts_to_sequences(sentences)

# padding the text
padded = pad_sequences(sequences)

print(word_index)


# Sequences = 
# [[5, 3, 2, 4],
# [5, 3, 2, 7], 
# 6, 3, 2, 4],
# [8, 6, 9, 2, 4, 10, 11]]
print(sequences)


# the text will now be padded into a matrix
# Padded Sequences:
# [[ 0  5  3  2  4]
#  [ 0  5  3  2  7]
#  [ 0  6  3  2  4]
#  [ 9  2  4 10 11]]
# each row has same length
print(padded)


# padding='post' it will happen after the sentence
# maxLen=5 to override sequence having of max size
# and will lose information
# truncating='post' will lose information from the ens
padded = pad_sequences(
    sequences,
    padding='post',
    truncating='post',
    maxlen=5)

```