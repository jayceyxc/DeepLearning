#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-20 10:20
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : graph_use.py
# @Software: PyCharm
# @Description 计算图的使用

import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量 v，并设置初始值为0.
    v = tf.get_variable('v', shape=[1], initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量 v，并设置初始值为1.
    v = tf.get_variable('v', shape=[1], initializer=tf.ones_initializer)

# 在计算图g1中读取变量"v"的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g1中，变量“v”的取值应该为0，所以下面这行会输出[0.]
        print(sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量"v"的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g2中，变量“v”的取值应该为1，所以下面这行会输出[1.]
        print(sess.run(tf.get_variable("v")))

