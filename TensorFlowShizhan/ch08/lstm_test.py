#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-24 17:24
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : lstm_test.py
# @Software: PyCharm
# @Description LSTM神经网络

import tensorflow as tf

lstm_hidden_size = 100
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)