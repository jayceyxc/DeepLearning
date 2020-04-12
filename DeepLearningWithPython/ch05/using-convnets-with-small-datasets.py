#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-18 10:58
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : using-convnets-with-small-datasets.py
# @Software: PyCharm
# @Description 在小数据集上运行卷积神经网络

import os
import shutil
from tensorflow.python.keras import layers, models, activations, losses, optimizers, metrics
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
import matplotlib.pyplot as plt

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = '/Users/yuxuecheng/Kaggle_data/cats_and_dogs/train'

# The directory where we will
# store our smaller dataset
base_dir = '/Users/yuxuecheng/Kaggle_data/cats_and_dogs_small'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)

# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)


def build_dataset():
    """
    构建数据集
    :return:
    """
    # Copy first 1000 cat images to train_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    # Copy next 500 cat images to validation_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    # Copy next 500 cat images to test_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    # Copy first 1000 dog images to train_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # Copy next 500 dog images to validation_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # Copy next 500 dog images to test_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    print('total training cat images:', len(os.listdir(train_cats_dir)))
    print('total training dog images:', len(os.listdir(train_dogs_dir)))
    print('total validation cat images:', len(os.listdir(validation_cats_dir)))
    print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
    print('total test cat images:', len(os.listdir(test_cats_dir)))
    print('total test dog images:', len(os.listdir(test_dogs_dir)))


def build_model():
    """
    构建卷积神经网络模型
    :return:
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation=activations.relu, input_shape=(150, 150, 3)))  # 148 * 148
    model.add(layers.MaxPooling2D((2, 2)))  # 74 * 74
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))  # 72 * 72
    model.add(layers.MaxPooling2D((2, 2)))  # 36 * 36
    model.add(layers.Conv2D(128, (3, 3), activation=activations.relu))  # 35 * 34
    model.add(layers.MaxPooling2D((2, 2)))  # 17 * 17
    model.add(layers.Conv2D(128, (3, 3), activation=activations.relu))  # 15 * 15
    model.add(layers.MaxPooling2D((2, 2)))  # 7 * 7
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation=activations.relu))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    print(model.summary())

    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=[metrics.binary_accuracy])

    return model


def build_model_with_dropout():
    """
    构建卷积神经网络模型，带有dropout层
    :return:
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation=activations.relu, input_shape=(150, 150, 3)))  # 148 * 148
    model.add(layers.MaxPooling2D((2, 2)))  # 74 * 74
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))  # 72 * 72
    model.add(layers.MaxPooling2D((2, 2)))  # 36 * 36
    model.add(layers.Conv2D(128, (3, 3), activation=activations.relu))  # 35 * 34
    model.add(layers.MaxPooling2D((2, 2)))  # 17 * 17
    model.add(layers.Conv2D(128, (3, 3), activation=activations.relu))  # 15 * 15
    model.add(layers.MaxPooling2D((2, 2)))  # 7 * 7
    model.add(layers.Flatten())
    model.add(layers.Dropout())   # 增加dropout层
    model.add(layers.Dense(512, activation=activations.relu))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    print(model.summary())

    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=[metrics.binary_accuracy])

    return model


def preprocess_image():
    """
    预处理图像
    :return:
    """
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    for data_batch, labels_batch in train_generator:
        print('data batch shape: ', data_batch.shape)
        print('test batch shape: ', labels_batch.shape)
        break

    return train_generator, validation_generator


def preprocess_image_with_augment():
    """
    预处理图像，使用数据增强
    :return:
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,      # 角度值，表示图像随机旋转的角度范围
        width_shift_range=0.2,  # 图像在水平方向上平移的范围
        height_shift_range=0.2, # 图像在垂直方向上平移的范围
        shear_range=0.2,        # 随机错切变换的角度
        zoom_range=0.2,         # 图像随机缩放的范围
        horizontal_flip=True,   # 是否随机将一半的图像水平翻转
        fill_mode='nearest'     # 用于填充新创建像素的方法。
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    for data_batch, labels_batch in train_generator:
        print('data batch shape: ', data_batch.shape)
        print('test batch shape: ', labels_batch.shape)
        break

    return train_generator, validation_generator


def view_augment():
    train_datagen = ImageDataGenerator(
        rotation_range=40,      # 角度值，表示图像随机旋转的角度范围
        width_shift_range=0.2,  # 图像在水平方向上平移的范围
        height_shift_range=0.2, # 图像在垂直方向上平移的范围
        shear_range=0.2,        # 随机错切变换的角度
        zoom_range=0.2,         # 图像随机缩放的范围
        horizontal_flip=True,   # 是否随机将一半的图像水平翻转
        fill_mode='nearest'     # 用于填充新创建像素的方法。
    )
    fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
    img_path = fnames[3]    # 选择一张图像进行增强
    img = image.load_img(img_path, target_size=(150, 150))  # 读取并调整图像的大小
    x = image.img_to_array(img) # 将其转换为(150, 150, 3)的Numpy数组
    x = x.reshape((1,) + x.shape) # 将其形状改变为(1, 150, 150, 3)
    i = 0
    for batch in train_datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break

    plt.show()


def print_accuracy_and_loss(history):
    """
    打印准确度和损失
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


def train_model():
    """
    训练模型，不使用数据增强和dropout层
    :return:
    """
    model = build_model()
    train_generator, validation_generator = preprocess_image()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=50)
    model.save('cats_and_dogs_small_1.h5')
    history = history.history
    print(history)
    print_accuracy_and_loss(history)


def train_model_with_augment():
    """
    训练模型，使用数据增强和dropout层
    :return:
    """
    model = build_model_with_dropout()
    train_generator, validation_generator = preprocess_image_with_augment()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
    model.save('cats_and_dogs_small_2.h5')
    history = history.history
    print(history)
    print_accuracy_and_loss(history)


if __name__ == '__main__':
    # build_dataset()
    # view_augment()
    train_model()
    # train_model_with_augment()







