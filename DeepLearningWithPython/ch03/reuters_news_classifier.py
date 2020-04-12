#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-13 08:22
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : reuters_news_classifier.py
# @Software: PyCharm
# @Description 路透社新闻分类

from tensorflow.python.keras.datasets import reuters
from tensorflow.python.keras import models, layers, activations, optimizers, losses, metrics, callbacks
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def show_example(x_train):
    # word_index是一个将单词映射为整数索引的字典
    word_index = reuters.get_word_index()
    # 键值颠倒，将整数索引映射为单词
    reverse_word_index = dict([value, key] for (key, value) in word_index.items())
    # 将评论解码。注意：索引减去了3，因为0、1、2是为‘padding’（填充）、'start of sequence'（序列开始）'unknown'（未知词）分别保留的索引
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
    print(decoded_newswire)


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
    acc_values = history_dict['categorical_accuracy']
    val_acc_values = history_dict['val_categorical_accuracy']

    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'bo', label='Training acc')  # 'bo' 表示蓝色圆点
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')  # 'b'表示蓝色实线
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def one_hot_version():
    (x_train, y_train), (x_test, y_test) =reuters.load_data(num_words=10000)
    print(len(x_train))
    print(len(x_test))
    print(x_train[10])
    print(y_train[10])
    # 将训练和测试数据向量化
    x_train = vectorize_sequences(x_train)
    x_test = vectorize_sequences(x_test)


    # 使用自定义函数来转换标签
    # def to_one_hot(labels, dimension=46):
    #     results = np.zeros((len(labels), dimension))
    #     for i, label in enumerate(labels):
    #         results[i, label] = 1.
    #
    #     return results
    #
    #
    # one_hot_train_labels = to_one_hot(y_train)  # 将标签数据向量化
    # one_hot_test_labels = to_one_hot(y_test)    # 将标签数据向量化

    # 使用tensorflow内置的函数
    one_hot_train_labels = to_categorical(y_train)
    one_hot_test_labels = to_categorical(y_test)

    # 构建网络
    model = models.Sequential()
    model.add(layers.Dense(64, activation=activations.relu, input_shape=(10000, )))
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(46, activation=activations.softmax))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])


    # 验证你的方法
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    history = model.fit(x=partial_x_train,
                        y=partial_y_train,
                        batch_size=512,
                        epochs=20,
                        # callbacks=[callbacks.EarlyStopping],
                        validation_data=(x_val, y_val))


    history_dict = history.history
    print(history_dict.keys())
    print(history_dict)

    show_train_loss_val_loss(history)
    show_train_loss_val_acc(history)
    print(model.evaluate(x_test, one_hot_test_labels))
    predictions = model.predict(x_test)
    print(predictions[0].shape)
    print(np.sum(predictions[0]))
    print(np.argmax(predictions[0]))


def vector_version():
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
    print(len(x_train))
    print(len(x_test))
    print(x_train[10])
    print(y_train[10])
    # 将训练和测试数据向量化
    x_train = vectorize_sequences(x_train)
    x_test = vectorize_sequences(x_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 构建网络
    model = models.Sequential()
    model.add(layers.Dense(64, activation=activations.relu, input_shape=(10000,)))
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(46, activation=activations.softmax))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.sparse_categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])

    # 验证你的方法
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]

    history = model.fit(x=partial_x_train,
                        y=partial_y_train,
                        batch_size=512,
                        epochs=20,
                        # callbacks=[callbacks.EarlyStopping],
                        validation_data=(x_val, y_val))

    history_dict = history.history
    print(history_dict.keys())
    print(history_dict)

    show_train_loss_val_loss(history)
    show_train_loss_val_acc(history)
    print(model.evaluate(x_test, y_test))
    predictions = model.predict(x_test)
    print(predictions[0].shape)
    print(np.sum(predictions[0]))
    print(np.argmax(predictions[0]))


if __name__ == '__main__':
    one_hot_version()
    # vector_version()
