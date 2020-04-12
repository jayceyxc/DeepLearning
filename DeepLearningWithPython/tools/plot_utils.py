#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-18 14:29
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : plot_utils.py
# @Software: PyCharm
# @Description 绘图工具类

import matplotlib.pyplot as plt


# 绘制验证分数
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


def plot_smoothed_history(history):
    """
    绘制平滑后的准确度和损失曲线，要求compile的metrics为[metrics.binary_accuracy]
    :param history:
    :return:
    """
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def plot_history(history):
    """
    绘制准确度和损失曲线，要求compile的metrics为[metrics.binary_accuracy]
    :param history:
    :return:
    """
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
