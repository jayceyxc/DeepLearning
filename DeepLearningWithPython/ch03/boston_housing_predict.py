#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-14 07:38
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : boston_housing_predict.py
# @Software: PyCharm
# @Description 波士顿房价预测。

from tensorflow.python.keras.datasets import boston_housing
from tensorflow.python.keras import models, layers, activations, losses, optimizers, metrics
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train)

# 数据标准化：普遍采用对每个特征做标准化，即对输入数据的每个特征（输入数据举证中的列），减去特征平均值，再除以标准差，这样
# 得到的特征平均值为0，标准差为1.
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

# 注意：用于测试数据标准化的均值和标准差都是在训练数据上计算得到的。
x_test -= mean
x_test /= std


def build_model():
    """
    构建模型
    :return:
    """
    model = models.Sequential()
    model.add(layers.Dense(64, activation=activations.relu, input_shape=(x_train.shape[1], )))
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.mse,
                  metrics=[metrics.mae])
    return model


# K折验证
k = 4
num_val_samples = len(x_train) // 4
num_epochs = 100
all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targes = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([x_train[: i * num_val_samples], x_train[(i + 1) * num_val_samples:]], axis=0)
    partial_targets_data = np.concatenate([y_train[: i * num_val_samples], y_train[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(partial_train_data,
                        partial_targets_data,
                        validation_data=(val_data, val_targes),
                        epochs=num_epochs,
                        batch_size=1,
                        verbose=0)
    # 保存每折的验证结果
    print(history.history)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

    val_mse, val_mae = model.evaluate(val_data, val_targes, verbose=0)
    # 保存每折的评估结果
    all_scores.append(val_mae)


print(all_scores)
print(np.mean(all_scores))
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


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


smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

