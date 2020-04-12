#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-11 08:02
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : imdb_comment_classifier.py
# @Software: PyCharm
# @Description IMDB数据集评论分类（二分类问题）

from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras import models, layers, activations, optimizers, losses, metrics, callbacks
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(x_train[0])
print(y_train[0])

print(max(max(sequence) for sequence in x_train))

# word_index是一个将单词映射为整数索引的字典
word_index = imdb.get_word_index()
# 键值颠倒，将整数索引映射为单词
reverse_word_index = dict([value, key] for (key, value) in word_index.items())
# 将评论解码。注意：索引减去了3，因为0、1、2是为‘padding’（填充）、'start of sequence'（序列开始）'unknown'（未知词）分别保留的索引
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
"""
? this film was just brilliant casting location scenery story direction everyone's really suited the part they played 
and you could just imagine being there robert ? is an amazing actor and now the same being director ? father came from 
the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks 
throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for ? and
 would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you 
 know what they say if you cry at a film it must have been good and this definitely was also ? to the two little boy's 
 that played the ? of norman and paul they were just brilliant children are often left out of the ? list i think because
  the stars that play them all grown up are such a big profile for the whole film but these children are amazing and 
  should be praised for what they have done don't you think the whole story was so lovely because it was true and was 
  someone's life after all that was shared with us all
"""
print(decoded_review)


# 准备数据
def vectorize_sequences(sequences, dimension=10000):
    """
    将整数序列编码为二进制矩阵
    :param sequences:
    :param dimension:
    :return:
    """
    # 创建一个形状为(len(sequences), dimension)的零矩阵
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.   # 将results[i]的指定索引设为1

    return results


# 将训练和测试数据向量化
x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

print(x_train[0])

# 将标签向量化
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')


# 构建网络
model = models.Sequential()
model.add(layers.Dense(16, activation=activations.relu, input_shape=(10000, )))
model.add(layers.Dense(16, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# 验证你的方法
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(x=partial_x_train,
                    y=partial_y_train,
                    batch_size=512,
                    epochs=4,
                    # callbacks=[callbacks.EarlyStopping],
                    validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())
print(history_dict)


def show_train_loss_val_loss(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')  # 'bo' 表示蓝色圆点
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # 'b'表示蓝色实线
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def show_train_loss_val_acc(history):
    history_dict = history.history
    acc_values = history_dict['binary_accuracy']
    val_acc_values = history_dict['val_binary_accuracy']

    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'bo', label='Training acc')  # 'bo' 表示蓝色圆点
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')  # 'b'表示蓝色实线
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


show_train_loss_val_loss(history)
show_train_loss_val_acc(history)
print(model.evaluate(x_test, y_test))
print(model.predict(x_test))


