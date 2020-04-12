#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-20 07:57
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : imdb_comments_using_simple_rnn.py
# @Software: PyCharm
# @Description 使用Embedding层和SimpleRNN来做IMDB电影评论分类

from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras import models, layers, activations, optimizers, losses, metrics

from DeepLearningWithPython.tools.plot_utils import plot_history

max_features = 10000
maxlen = 500
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (sample x time)')
input_train = sequence.pad_sequences(x_train, maxlen=maxlen)
input_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# 用Embedding层和SimpleRNN层来训练模型
model = models.Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(1, activation=activations.sigmoid))

model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

plot_history(history=history)
