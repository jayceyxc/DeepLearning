#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-27 09:41
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : load_images.py
# @Software: PyCharm
# @Description 加载图像数据

from __future__ import absolute_import, division
import tensorflow as tf
import pathlib
import random
import IPython.display as display

tf.enable_eager_execution()
print(tf.VERSION)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# retrieve the images
data_root = tf.keras.utils.get_file('flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
data_root = pathlib.Path(data_root)
print(data_root)

for item in data_root.iterdir():
    print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

print(all_image_paths[:10])

attributions = (data_root/"LICENSE.txt").read_text(encoding="utf8").splitlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)


def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return 'Image (CC BY 2.0) ' + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


for n in range(3):
    image_path = random.choice(all_image_paths)
    display.display(display.Image(image_path))
    print(caption_image(image_path))
    print()

