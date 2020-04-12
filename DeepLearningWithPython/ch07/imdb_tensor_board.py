#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-22 09:19
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : imdb_tensor_board.py
# @Software: PyCharm
# @Description 对IMDB评论进行分类，使用TensorBoard进行可视化

from tensorflow.python.keras import models, layers, activations, optimizers, losses, metrics, callbacks
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence

from DeepLearningWithPython.tools.plot_utils import plot_history

max_features = 2000
max_len = 500

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (sample x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('input_train shape:', x_train.shape)
print('input_test shape:', x_test.shape)

model = models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation=activations.relu))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation=activations.relu))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(1))
print(model.summary())
model.compile(
    optimizer=optimizers.RMSprop(),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)

callbacks = [
    callbacks.TensorBoard(
        log_dir='imdb_tensor_board_log_dir',    # 日志文件将被写入这个位置
        histogram_freq=1,                   # 每一轮之后记录激活直方图
        embeddings_freq=1,                   # 每一轮之后记录嵌入数据
    )
]

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)

print('train finished')
plot_history(history)

