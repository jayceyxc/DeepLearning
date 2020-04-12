#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-10 08:05
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : mnist_example.py
# @Software: PyCharm
# @Description 利用深度学习来分类MNIST

import tensorflow as tf
from tensorflow import data
from tensorflow.python import keras, layers
from tensorflow.python.keras.datasets import mnist

x_train, y_train, x_test, y_test = mnist.load_data()
x_train = x_train.reshape((60000, 28 * 28))
x_train = x_train.astype('float32') /255

