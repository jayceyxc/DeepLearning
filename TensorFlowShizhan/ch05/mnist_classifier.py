#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-22 17:41
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : mnist_classifier.py
# @Software: PyCharm
# @Description MNIST手写数字识别

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers

MNIST_DATA_PATH = '/Users/yuxuecheng/TF_data/MNIST_data'

mnist = input_data.read_data_sets(MNIST_DATA_PATH, one_hot=True)

# 打印 Training data size：55000
print('Training data size: {0}'.format(mnist.train.num_examples))

# 打印 Validating data size
print('Validating data size: {0}'.format(mnist.validation.num_examples))

# 打印 Testing data size
print('Testing data size: {0}'.format(mnist.test.num_examples))

# 打印 Example training data
print('Example training data: {0}'.format(mnist.train.images[0]))

# 打印 Example training data label:
print('Example training data label: {0}'.format(mnist.train.labels[0]))

# MNIST 数据集相关的参数
INPUT_NODE = 784    # 输入节点
OUTPUT_NODE = 10    # 输出节点

LAYER1_NODE = 500   # 隐藏层数
BATCH_SIZE = 100    # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    """
    定义辅助函数来计算前向传播结果，使用ReLU做为激活函数。
    :param input_tensor:
    :param avg_class:
    :param weights1:
    :param biases1:
    :param weights2:
    :param biases2:
    :return:
    """
    if avg_class is None:
        # 不使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))

        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    """
    定义训练过程
    :param mnist:
    :return:
    """
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    regularizer = layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After {0} training step(s), validation accuracy using average model is {1}'.format(i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After {0} training step(s), test accuracy using average model is {1}'.format(i, test_acc))


def main(argv=None):
    train(mnist)


if __name__ == '__main__':
    main()
