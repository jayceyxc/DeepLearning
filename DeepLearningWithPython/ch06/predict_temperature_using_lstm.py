#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-20 08:27
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : predict_temperature_using_lstm.py
# @Software: PyCharm
# @Description 用LSTM来预测温度

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras import models, layers, activations, optimizers, losses, metrics

data_dir = "/Users/yuxuecheng/TF_data/jena_climate"
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))
print(len(lines[0]))

# 解析数据
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

temp = float_data[:, 1]  # 温度（单位：摄氏度）
# 绘制温度时间列
plt.plot(range(len(temp)), temp)
# 绘制前10天的温度时间序列
plt.plot(range(1440), temp[:1440])
plt.show()

# 准备数据
## 数据标准化
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def genenrator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6, reverse=False):
    """
    数据生成器，他生成了一个元组(samples, targets)，其中samples是输入数据的一个批量，targets是对应的目标温度数组
    :param data: 浮点数数据组成的原始数组
    :param lookback: 输入数据应该包括过去多少个时间步
    :param delay: 目标应该在未来多少个时间步之后
    :param min_index: data数组中的索引，用于界定需要抽取哪些时间步，这有助于保存一部分数据用于验证、另一部分用于测试
    :param max_index: data数组中的索引，用于界定需要抽取哪些时间步，这有助于保存一部分数据用于验证、另一部分用于测试
    :param shuffle: 是否打乱样本
    :param batch_size: 每个批量的样本数
    :param step: 数据采样的周期（单位：时间步）。
    :param reverse: 是否对samples进行反向倒序
    :return: 生成的数据
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        if reverse:
            yield samples[:, ::-1, :], targets
        else:
            yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = genenrator(float_data,
                       lookback=lookback,
                       delay=delay,
                       min_index=0,
                       max_index=200000,
                       shuffle=True,
                       step=step,
                       batch_size=batch_size)

val_gen = genenrator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=200001,
                     max_index=300000,
                     shuffle=True,
                     step=step,
                     batch_size=batch_size)

test_gen = genenrator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=300001,
                      max_index=None,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

train_reverse_gen = genenrator(float_data,
                               lookback=lookback,
                               delay=delay,
                               min_index=0,
                               max_index=200000,
                               shuffle=True,
                               step=step,
                               batch_size=batch_size,
                               reverse=True)

val_reverse_gen = genenrator(float_data,
                             lookback=lookback,
                             delay=delay,
                             min_index=200001,
                             max_index=300000,
                             shuffle=True,
                             step=step,
                             batch_size=batch_size,
                             reverse=True)

test_reverse_gen = genenrator(float_data,
                              lookback=lookback,
                              delay=delay,
                              min_index=300001,
                              max_index=None,
                              shuffle=True,
                              step=step,
                              batch_size=batch_size,
                              reverse=True)

val_steps = (300000 - 200001 - lookback) // batch_size  # 为了查看整个验证集，需要从val_gen中抽取多少次
test_steps = (len(float_data) - 300001 - lookback) // batch_size  # 为了查看整个测试集，需要从test_gen中抽取多少次


# 一种基于常识的、非机器学习的基准方法
def evaluate_navie_method():
    """
    计算符合常识的基准方法的MAE
    :return:
    """
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    return np.mean(batch_maes)


def plot_mae_loss(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()


# 一种基本的机器学习方法
def basic_machine_learning_method():
    """
    构建一个全连接的神经网络来预测
    :return:
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
    model.add(layers.Dense(32, activation=activations.relu))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss=losses.mae
    )
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    plot_mae_loss(history, title='Training and Validation loss (FCN)')


def gru_model_basis():
    """
    基于GRU的模型的基准
    :return:
    """
    model = models.Sequential()
    model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss=losses.mae
    )
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    plot_mae_loss(history, title='Training and Validation loss (GRU)')


def gru_model_with_dropout():
    """
    基于GRU的模型，增加了dropout来防止过拟合
    :return:
    """
    model = models.Sequential()
    model.add(layers.GRU(32,
                         dropout=0.2,
                         recurrent_dropout=0.2,
                         input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss=losses.mae
    )
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    plot_mae_loss(history, title='Training and Validation loss (GRU with dropout)')


def stacked_gru_model_with_dropout():
    """
    堆叠GRU的模型，增加了dropout来防止过拟合
    :return:
    """
    model = models.Sequential()
    model.add(layers.GRU(32,
                         dropout=0.1,
                         recurrent_dropout=0.5,
                         return_sequences=True,
                         input_shape=(None, float_data.shape[-1])))
    model.add(layers.GRU(64,
                         activation=activations.relu,
                         dropout=0.1,
                         recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss=losses.mae
    )
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    plot_mae_loss(history, title='Training and Validation loss (stacked GRU with dropout)')


def gru_model_reverse():
    """
    基于GRU的模型，但是数据是反向的
    :return:
    """
    model = models.Sequential()
    model.add(layers.GRU(32,
                         dropout=0.2,
                         recurrent_dropout=0.2,
                         input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss=losses.mae
    )
    history = model.fit_generator(train_reverse_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=val_reverse_gen,
                                  validation_steps=val_steps)
    plot_mae_loss(history, title='Training and Validation loss (GRU reverse model)')


if __name__ == '__main__':
    mae_mean = evaluate_navie_method()
    celsius_mae = mae_mean * std[1]
    print(celsius_mae)
    basic_machine_learning_method()
    gru_model_basis()
    gru_model_with_dropout()
    stacked_gru_model_with_dropout()
    gru_model_reverse()
