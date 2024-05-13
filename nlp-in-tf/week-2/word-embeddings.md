# Word Embedddings

This week will cover Embeddings, where token are mapped as vectors in a high dimension space. With labeled examples and embeddings, the vectors can be configured so that words with similar meaning will have similar direction in vector space.

This will start the process of training a neural network to learn sentiment in text. The focus will be on movie reviews labeled "positive" and "negative", and determine the words that drive the meaning in sentence.

## Introduction

Instead of representing the words as number from 1 to 10,000, we will have a better way to represent words.

Before we looked at Tokenizing words which is converting a sentence into a sequence of numbers, where the number if value of key-pair. Convert for example TensorFlow into "9".

Using TensorFlow I was able to process a strings to get indices of words in corpus of string, then convert them into a matrices of numbers.

## IMDB Datasets

