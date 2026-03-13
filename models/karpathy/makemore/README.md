## makemore

Makes more of the things that you give it.

Takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an
LLM, with a wide choice of models from bigrams, Transformer.

For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names.

Character level model = model sequence of characters and predict next character in that sequence.
Language models = Bigram, Bag of Words, MLP, RNN, GRU, Transformer

names.txt = Consists of random names (training data set)

Predicting next sequence of characters GIVEN some characters before it.
