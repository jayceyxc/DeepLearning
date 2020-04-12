#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-22 21:19
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : mnist_eval.py
# @Software: PyCharm
# @Description 定义了测试过程

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from TensorFlowShizhan.ch05.best_practice import mnist_inference
from TensorFlowShizhan.ch05.best_practice import mnist_train

EVAL_INTERVAL_SECS = 10

MNIST_DATA_PATH = '/Users/yuxuecheng/TF_data/MNIST_data'


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-output')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After {0} training step(s), validation accuracy = {1}'.format(global_step,
                                                                                         accuracy_score))
                else:
                    print('No checkpoint file found')

                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets(MNIST_DATA_PATH, one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()