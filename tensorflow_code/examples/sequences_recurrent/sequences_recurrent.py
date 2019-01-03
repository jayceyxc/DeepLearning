#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-12-04 19:02
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : sequences_recurrent.py
# @Software: PyCharm
# @Description 循环神经网络

import tensorflow as tf
from tensorflow.contrib import rnn

time_steps = 10
batch_size = 64
num_features = 10
lstm_size = 10
words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])

lstm = rnn.BasicLSTMCell(lstm_size)

# Initial state of the LSTM memory
state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
probabilities = []
loss = 0.0
for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(inputs=current_batch_of_words, state=state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)


rnn.MultiRNNCell