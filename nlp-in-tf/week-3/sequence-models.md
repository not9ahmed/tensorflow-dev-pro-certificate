# Sequence Models

The following chapter will focus on using sequences to determine sentiment analysis. As we previously focused on words seperately, but word such as "not fun" should be labeled as negative class.

The order/sequence of words matters for the meaning of the sentence.

RNN can perserve the context from timestamp to timestamp, but it can be lost in long context. However, LSTM can perserve that context it has cell state, and it can perserve the context for long sentence.

## Introduction

Context of words was hard to follow when words is broken into sub-words, and the sequence which tokens for sub-words appear becomes very important in understanding their meaning.

Neural Network is like a function where it takes

**Input:**
- Data
- Labels

**Output:**
- Rules


$$
f(Data, Labels) = \text{Rules}
$$

But it does not take into account the sequence!

## Example of Fibonacci Sequence

The following image showcases the fibonacci sequence
![image of fibonacci sequence](images/fibonacci-sequence.png)


Fibonacci sequence can be visualized as the below.
Where 2 input are carried to be summed to the next value, and the second value is carried to next summation.

It forms the base concept for Recurrent Neural Network (RNN)
![image of fibonacci sequence visualized](images/fibonacci-sequence-visualized.png)


The following image showcases RNN where it applies the same concept of sequencing from fibonacci. An input + output of previous function will be take as input to current function.
![image of rnn](images/rnn.png)