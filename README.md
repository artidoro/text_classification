# Text Classification

Implements CBOW, CNN, LSTM, LSTM+Attention, LSTM+CNN, Ensembling for text classification.
Includes logging, command line arguments, and evaluation code.

Note that building the vocabulary and iterators was done through the Torch DataLoaders included in TorchText.
The embeddings were manually loaded from fastText although TorchText allows to directly load them.

Usage:
`python main.py -h`
