#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-18 16:42
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : imdb_comments_using_glove.py
# @Software: PyCharm
# @Description 使用GloVe词向量坐IMDB影评分类

import os
import numpy as np
import codecs
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import models, layers, activations, optimizers, losses, metrics

from DeepLearningWithPython.tools.plot_utils import plot_history

imdb_dir = '/Users/yuxuecheng/TF_data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = codecs.open(os.path.join(dir_name, fname), encoding='utf8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


# 对数据进行分词
maxlen = 100    # 在100个单词后截断评论
training_samples = 200      # 在200个样本上训练
validation_samples = 10000  # 在10000个样本上进行验证
max_words = 10000   # 只考虑数据集中前10000个最常见的单词

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts=texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s uniqe tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# 解析GloVe词嵌入文件
glove_dir = '/Users/yuxuecheng/TF_data/models/glove.6B'
embedding_index = {}
f = codecs.open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embedding_index))

# 准备GloVe词嵌入矩阵
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# 定义模型
model = models.Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))
print(model.summary())

# 在模型中加载GloVe嵌入
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# 训练模型
model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

plot_history(history)
