#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-18 16:00
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : using_word_embeddings.py
# @Software: PyCharm
# @Description 使用词嵌入

from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras import preprocessing, models, layers, activations, optimizers, losses, metrics

max_features = 10000
maxlen = 40

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = models.Sequential()
# 指定Embedding层的最大输入长度，以便后面将嵌入输入展平。Embedding层激活的状态为(samples, maxlen, 8)
model.add(layers.Embedding(10000, 8, input_length=maxlen))
model.add(layers.Flatten())  # 将三维的嵌入张量展平成形状为(samples, maxlen * 8)的二维张量
model.add(layers.Dense(1, activation=activations.sigmoid))
model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)
print(model.summary())
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
print(history.history)




