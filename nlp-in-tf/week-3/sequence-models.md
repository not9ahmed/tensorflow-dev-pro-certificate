# Sequence Models

The following chapter will focus on using sequences to determine sentiment analysis. As we previously focused on words seperately, but word such as "not fun" should be labeled as negative class.

The order/sequence of words matters for the meaning of the sentence.

RNN can perserve the context from timestamp to timestamp, but it can be lost in long context. However, LSTM can perserve that context it has cell state, and it can perserve the context for long sentence.