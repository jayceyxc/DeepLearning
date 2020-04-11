#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-04-16 12:32
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : intro_to_word_embeddings.py
# @Software: PyCharm
# @Description https://www.tensorflow.org/alpha/tutorials/sequences/word_embeddings

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers, models, activations, losses, metrics, optimizers
from tensorflow.python.keras.datasets import imdb
import matplotlib.pyplot as plt
import io

# The Embedding layer takes at least two arguments:
# the number of possible words in the vocabulary, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 32.
# embedding_layer = layers.Embedding(1000, 32)

vocab_size = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[0])

# Convert the integers back to words
# # A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))


# 使用pad_sequences来标准化评论的长度
maxlen = 500
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=maxlen)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=maxlen)

print(train_data[0])

# Create a simple model
embedding_dim = 16
model = keras.Sequential([
    # The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and looks up the embedding
    # vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the
    # output array. The resulting dimensions are: (batch, sequence, embedding)`
    layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    # Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the
    # sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
    layers.GlobalAveragePooling1D(),
    # This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
    layers.Dense(16, activation=activations.relu),
    # The last layer is densely connected with a single output node. Using the sigmoid activation function, this value
    # is a float between 0 and 1, representing a probability (or confidence level) that the review is positive.
    layers.Dense(1, activation=activations.sigmoid)
])
print(model.summary())

# Compile and train the model
model.compile(optimizer=optimizers.Adam(),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
history = model.fit(train_data, train_labels, epochs=30, batch_size=512, validation_split=0.2)

history_dict = history.history
print(history_dict)
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation add')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5, 1))
plt.show()

# Next, let's retrieve the word embeddings learned during training
embedding_layer = model.layers[0]
weights_array = embedding_layer.get_weights()
weights = embedding_layer.get_weights()[0]
print(weights.shape)

# We will now write the weights to disk. we will upload two files in tab separated format: a file of vectors
# (containing the embedding), and a file of meta data (containing the words).
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + '\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
out_v.close()
out_m.close()


