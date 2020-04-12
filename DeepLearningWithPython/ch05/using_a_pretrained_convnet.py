#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-18 14:03
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : using_a_pretrained_convnet.py
# @Software: PyCharm
# @Description 使用预训练的卷积神经网络

import os
import numpy as np
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import models, layers, optimizers, activations, losses, metrics
import matplotlib.pyplot as plt


# 绘制验证分数
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


def plot_smoothed_history(history):
    """
    绘制平滑后的准确度和损失曲线，要求compile的metrics为[metrics.binary_accuracy]
    :param history:
    :return:
    """
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def plot_history(history):
    """
    绘制准确度和损失曲线，要求compile的metrics为[metrics.binary_accuracy]
    :param history:
    :return:
    """
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def feature_extract():
    """
    不使用数据增强的特征抽取
    :return:
    """
    conv_base = VGG16(
        weights='imagenet',         # 指定模型初始化的权重检查点
        include_top=False,          # 指定模型最后是否包含密集连接分类器
        input_shape=(150, 150, 3))  # 输入到网络中的图像张量的形状
    print(conv_base.summary())

    base_dir = "/Users/yuxuecheng/Kaggle_data/cats_and_dogs_small"
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 20

    def extract_features(directory, sample_count):
        """
        抽取特征
        :param directory:
        :param sample_count:
        :return:
        """
        features = np.zeros(shape=(sample_count, 4, 4, 512))
        labels = np.zeros(shape=(sample_count))
        generator = datagen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary'
        )
        i = 0
        for inputs_batch, labels_batch in generator:
            features_batch = conv_base.predict(inputs_batch)
            features[i * batch_size : (i + 1) * batch_size] = features_batch
            labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count:
                break

        return features, labels

    train_features, train_labels = extract_features(train_dir, 2000)
    validation_features, validation_labels = extract_features(validation_dir, 1000)
    test_features, test_labels = extract_features(test_dir, 1000)

    # 提取到形状是(samples, 4, 4, 512)，需要将其展平为(samples, 8192)
    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

    # 现在可以定义自己的密集连接分类器（注意要使用dropout正则化），并在刚刚保存的数据和标签上训练这个分类器
    model = models.Sequential()
    model.add(layers.Dense(256, activation=activations.relu, input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation=activations.sigmoid))
    model.compile(
        optimizer=optimizers.RMSprop(lr=2e-5),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy]
    )

    history = model.fit(
        train_features,
        train_labels,
        epochs=30,
        batch_size=20,
        validation_data=(validation_features, validation_labels)
    )

    plot_history(history=history)


def feature_extract_with_augment():
    """
    不使用数据增强的特征抽取
    :return:
    """
    conv_base = VGG16(
        weights='imagenet',         # 指定模型初始化的权重检查点
        include_top=False,          # 指定模型最后是否包含密集连接分类器
        input_shape=(150, 150, 3))  # 输入到网络中的图像张量的形状
    print(conv_base.summary())

    base_dir = "/Users/yuxuecheng/Kaggle_data/cats_and_dogs_small"
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'     # 因为使用了binary_accuracy，所以需要使用二进制标签
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'     # 因为使用了binary_accuracy，所以需要使用二进制标签
    )

    # 现在可以定义自己的密集连接分类器（注意要使用dropout正则化），并在刚刚保存的数据和标签上训练这个分类器
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Dense(256, activation=activations.relu, input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation=activations.sigmoid))
    print('This is the number of trainable weights '
          'before freezing the conv base:', len(model.trainable_weights))
    conv_base.trainable = False
    print('This is the number of trainable weights '
          'after freezing the conv base:', len(model.trainable_weights))
    model.compile(
        optimizer=optimizers.RMSprop(lr=2e-5),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy]
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )

    plot_history(history=history)


def fine_tuning_with_augment():
    """
    使用数据增强的模型微调
    微调网络的步骤如下：
    （1）在已经训练好的基网络（base network）上添加自定义层。
    （2）冻结基网络
    （3）训练所添加的部分
    （4）解冻基网络的一些层
    （5）联合训练解冻的这些层和添加的部分
    :return:
    """
    conv_base = VGG16(
        weights='imagenet',         # 指定模型初始化的权重检查点
        include_top=False,          # 指定模型最后是否包含密集连接分类器
        input_shape=(150, 150, 3))  # 输入到网络中的图像张量的形状
    print(conv_base.summary())

    base_dir = "/Users/yuxuecheng/Kaggle_data/cats_and_dogs_small"
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'     # 因为使用了binary_accuracy，所以需要使用二进制标签
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'     # 因为使用了binary_accuracy，所以需要使用二进制标签
    )

    # 现在可以定义自己的密集连接分类器（注意要使用dropout正则化），并在刚刚保存的数据和标签上训练这个分类器
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Dense(256, activation=activations.relu, input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation=activations.sigmoid))
    print('This is the number of trainable weights '
          'before freezing the conv base:', len(model.trainable_weights))
    conv_base.trainable = False
    print('This is the number of trainable weights '
          'after freezing the conv base:', len(model.trainable_weights))
    model.compile(
        optimizer=optimizers.RMSprop(lr=2e-5),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy]
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )

    plot_history(history=history)

    # 训练完新添加的部分，下面解冻基网络的一些层，并联合训练解冻的这些层和添加的部分
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(
        optimizer=optimizers.RMSprop(lr=2e-5),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy]
    )
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )
    plot_smoothed_history(history)


if __name__ == '__main__':
    feature_extract()
    feature_extract_with_augment()
    fine_tuning_with_augment()
