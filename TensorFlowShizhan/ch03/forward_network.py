#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-20 12:28
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : forward_network.py
# @Software: PyCharm
# @Description 前向传播神经网络

import tensorflow as tf

# 声明w1、w2两个变量。这里还通过seed参数设定了随机种子，这样可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量。注意这里x是一个1*2的矩阵
# x = tf.constant([0.7, 0.9])
x = tf.placeholder(tf.float32, shape=(1, 2), name='input')
print(x.shape)
# x = tf.constant([[0.7, 0.9], ])
# print(x.shape)

a = tf.matmul(x, w1)
print(a.shape)
y = tf.matmul(a, w2)
print(y.shape)

sess = tf.Session()
# sess.run(w1.initializer) # 初始化w1
# sess.run(w2.initializer) # 初始化w2
sess.run(tf.global_variables_initializer())  # 调用全局变量初始化函数。

# 当定义为placeholder后，下面的代码会报错
# InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'input' with dtype float and shape [1,2]
# print(sess.run(y))

print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))

# 使用sigmoid函数将y转换为0~1之间的数值。转换后y代表预测是正样本的概率，1-y代表预测是负样本的概率
y_ = tf.sigmoid(y)
# 定义损失函数来刻画预测值与真实值的差距
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))
)

learning_rate = 0.001

# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

sess.close()


