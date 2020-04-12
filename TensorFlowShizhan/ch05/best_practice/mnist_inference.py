#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-22 20:57
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : mnist_inference.py
# @Software: PyCharm
# @Description 定义前向传播的过程以及神经网络中的参数

import tensorflow as tf

# 定义神经网络结构相关的参数
INPUT_NODE = 784    # 输入节点
OUTPUT_NODE = 10    # 输出节点

LAYER1_NODE = 500   # 隐藏层数


# 同过tf.get_variable函数来获取变量。在训练神经网络时会创建这些变量;在测试时会通
# 过保仔的模型加载这些变量的取值。而且更加方便的是，因为可以在变量加载时将滑动平均变量
# 重命名，所以可以直撞通过同样的名字在训练时使用变量自身，而在测试时使用变量的滑动平
# 均值。在这个的函数中也会将变量的正则化损失加入损失集合。
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        'weights', shape=shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights


def inference(input_tensor, regularizer):
    """
    定义神经网络的前向传播过程
    :param input_tensor:
    :param regularizer:
    :return:
    """
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    # 返回最后前向传播的结果
    return layer2

