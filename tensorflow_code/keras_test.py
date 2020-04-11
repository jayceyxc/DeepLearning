#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-01-23 10:38
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : keras_test.py
# @Software: PyCharm
# @Description 测试keras接口

import tensorflow as tf
from tensorflow.contrib.keras import layers
import numpy as np

# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or
layers.Dense(64, activation=tf.keras.activations.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer=tf.keras.initializers.orthogonal)

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

# 训练和评估
# 构建好模型后，通过调用 compile 方法配置该模型的学习流程：
# 构建简单的模型
model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layer=layers.Dense(64, activation=tf.keras.activations.relu))
# Add another
model.add(layer=layers.Dense(64, activation=tf.keras.activations.relu))
# Add a softmax layer with 10 output units:
model.add(layer=layers.Dense(10, activation=tf.keras.activations.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

# Configure a model for mean-squared error regression.
# model.compile(optimizer=tf.train.AdamOptimizer(0.01),
#               loss='mse',       # mean squared error
#               metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
# model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#               loss=tf.keras.losses.categorical_crossentropy,
#               metrics=[tf.keras.metrics.categorical_accuracy])

data = np.random.random((1000, 32)).astype(np.float32)
labels = np.random.random((1000, 10)).astype(np.float32)

val_data = np.random.random((1000, 32)).astype(np.float32)
val_labels = np.random.random((1000, 10)).astype(np.float32)

# model.fit(data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

# 输入tf.data数据集
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()
# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30, validation_data=val_dataset, validation_steps=3)

evaluate_data = np.random.random((1000, 32)).astype(np.float32)
evaluate_labels = np.random.random((1000, 10)).astype(np.float32)

model.evaluate(evaluate_data, evaluate_labels, batch_size=32)

