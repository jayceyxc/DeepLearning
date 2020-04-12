#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-22 21:07
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : mnist_train.py
# @Software: PyCharm
# @Description 定义了神经网络的训练过程


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers
import os

# 加载mnist_inference.py中定义的常量和前向传播的函数
from TensorFlowShizhan.ch05.best_practice import mnist_inference

MNIST_DATA_PATH = '/Users/yuxuecheng/TF_data/MNIST_data'

BATCH_SIZE = 100    # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'model'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-output')

    # 损失函数的计算
    regularizer = layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
        .minimize(loss=loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    # 初始化会话，并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 循环的训练神经网络
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print('After {0} training step(s), loss on training batch is {1}'.format(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets(MNIST_DATA_PATH, one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
