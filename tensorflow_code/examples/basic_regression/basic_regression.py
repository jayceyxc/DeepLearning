#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time    : 2018/11/15 18:47
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : basic_regression.py
# @Software: PyCharm
# @Description Predict house prices. Builds a model to predict the median price of homes in a Boston
# suburb during the mid-1970s

"""
The dataset contains 13 different features:

1. Per capita crime rate.   人均犯罪率
2. The proportion of residential land zoned for lots over 25,000 square feet. 占住宅用地面积逾25,000平方英尺的比例
3. The proportion of non-retail business acres per town.  每个城镇的非零售商业面积比例
4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise). 查尔斯河哑变量，是否临河
5. Nitric oxides concentration (parts per 10 million).  一氧化氮浓度(千万分之一)
6. The average number of rooms per dwelling.   每个住宅的平均房间数
7. The proportion of owner-occupied units built before 1940. 1940年以前建造的业主自住单位的比例
8. Weighted distances to five Boston employment centers. 加权距离波士顿五个就业中心
9. Index of accessibility to radial highways.  径向公路可达性指数
10. Full-value property-tax rate per $10,000.  每1万美元的全额财产税税率
11. Pupil-teacher ratio by town. 学生与教师的比率按城市而定
12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.  1000 * (Bk - 0.63) ** 2，其中Bk是城镇黑人的比例。
13. Percentage lower status of the population.  人口地位较低的百分比
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))  # 102 examples, 13 features

print(train_data[0])

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)
print(df.head())

print(train_labels[0:10])  # Display first 10 entries

# Normalize features
# Test data is *not* used when calculating the mean and std
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized


# Create the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    return model


model = build_model()
model.summary()


# Train the model


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

EPOCHS = 500
# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()


error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
plt.show()