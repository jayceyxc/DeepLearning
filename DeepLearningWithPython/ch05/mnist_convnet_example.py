#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-15 08:07
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : mnist_convnet_example.py
# @Software: PyCharm
# @Description 使用卷积神经网络来进行MNIST数字的分类

from tensorflow.python.keras import models, layers, activations, optimizers, metrics, losses
from keras.datasets import mnist
from keras.utils import to_categorical


def build_model():
    """
    构造卷积神经网络模型
    :return:
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
    print(model.summary())
    '''
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 32)        320       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
    =================================================================
    Total params: 55,744
    Trainable params: 55,744
    Non-trainable params: 0
    _________________________________________________________________
    None
    '''
    # 在卷积神经网络上添加分类器。需要将前面输出的张量（大小为(3,3,64)）输入到一个密集连接分类器网络中，即Dense层的堆叠。
    # 由于Dense网络可以处理1D张量，但是现在的输出时3D张量。首先，需要将3D张量展平为1D，然后在上面添加几个Dense层。
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(10, activation=activations.softmax))
    print(model.summary())

    '''
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 32)        320       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
    =================================================================
    Total params: 55,744
    Trainable params: 55,744
    Non-trainable params: 0
    _________________________________________________________________
    None
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 32)        320       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
    _________________________________________________________________
    flatten (Flatten)            (None, 576)               0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                36928       （576 * 64 + 64   weights参数数量+biases参数数量）   
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 93,322
    Trainable params: 93,322
    Non-trainable params: 0
    _________________________________________________________________
    None
    '''
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])
    return model


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = build_model()
    model.fit(train_images, train_labels, epochs=10, batch_size=64)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_loss)
    print(test_acc)


