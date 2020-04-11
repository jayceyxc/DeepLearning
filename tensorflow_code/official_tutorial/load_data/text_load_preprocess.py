#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_datasets as tfds
import os


# In[ ]:


'''
本教程中使用的文本文件已经进行过一些典型的预处理，主要包括删除了文档页眉和页脚，行号，章节标题。请下载这些已经被局部改动过的文件。
'''


# In[2]:


DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)


# In[3]:


parent_dir = os.path.dirname(text_dir)
print(parent_dir)


# In[4]:


'''
将文本加载到数据集中
迭代整个文件，将整个文件加载到自己的数据集中。

每个样本都需要单独标记，所以请使用 tf.data.Dataset.map 来为每个样本设定标签。这将迭代数据集中的每一个样本并且返回（ example, label ）对。
'''


# In[5]:


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


# In[6]:


labeld_data_sets = []
for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeld_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeld_data_sets.append(labeld_dataset)


# In[7]:


# 将这些标记的数据集合并到一个数据集中，然后对其进行随机化操作。
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 50000


# In[8]:


all_labeled_data = labeld_data_sets[0]
for labeld_dataset in labeld_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeld_dataset)


# In[9]:


all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)


# In[10]:


get_ipython().run_line_magic('pinfo', 'all_labeled_data.shuffle')


# In[11]:


# 你可以使用 tf.data.Dataset.take 与 print 来查看 (example, label) 对的外观。numpy 属性显示每个 Tensor 的值。
for ex in all_labeled_data.take(5):
    print(ex)


# In[12]:


'''
将文本编码成数字
机器学习基于的是数字而非文本，所以字符串需要被转化成数字列表。 为了达到此目的，我们需要构建文本与整数的一一映射。
'''


# In[13]:


'''
建立词汇表

首先，通过将文本标记为单独的单词集合来构建词汇表。在 TensorFlow 和 Python 中均有很多方法来达成这一目的。在本教程中:

1. 迭代每个样本的 numpy 值。
2. 使用 tfds.features.text.Tokenizer 来将其分割成 token。
3. 将这些 token 放入一个 Python 集合中，借此来清除重复项。
4. 获取该词汇表的大小以便于以后使用。
'''


# In[14]:


tokenizer = tfds.features.text.Tokenizer()


# In[15]:


vocabulary_set = set()


# In[17]:


for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)


# In[18]:


vocab_size = len(vocabulary_set)
print(vocab_size)


# In[20]:


'''
样本编码
通过传递 vocabulary_set 到 tfds.features.text.TokenTextEncoder 来构建一个编码器。编码器的 encode 方法传入一行文本，返回一个整数列表。
'''
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


# In[21]:


example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)


# In[22]:


encoded_example = encoder.encode(example_text)
print(encoded_example)


# In[23]:


# 现在，在数据集上运行编码器（通过将编码器打包到 tf.py_function 并且传参至数据集的 map 方法的方式来运行）。
def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

all_encoded_data = all_labeled_data.map(encode_map_fn)


# In[24]:


'''
将数据集分割为测试集和训练集且进行分支
使用 tf.data.Dataset.take 和 tf.data.Dataset.skip 来建立一个小一些的测试数据集和稍大一些的训练数据集。

在数据集被传入模型之前，数据集需要被分批。最典型的是，每个分支中的样本大小与格式需要一致。但是数据集中样本
并不全是相同大小的（每行文本字数并不相同）。因此，使用 tf.data.Dataset.padded_batch（而不是 batch ）将样
本填充到相同的大小。
'''


# In[25]:


train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)


# In[26]:


train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))


# In[27]:


test_data = all_encoded_data.take(TAKE_SIZE)


# In[28]:


test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))


# In[29]:


'''
现在，test_data 和 train_data 不是（ example, label ）对的集合，而是批次的集合。每个批次都是一对（多样本, 多标签 ），表示为数组。
'''


# In[31]:


sample_text, sample_labels = next(iter(test_data))
print(sample_text[0], sample_labels[0])


# In[32]:


# 由于我们引入了一个新的 token 来编码（填充零），因此词汇表大小增加了一个。
vocab_size += 1


# In[33]:


# 建立模型
model = tf.keras.Sequential()


# In[34]:


# 第一层将整数表示转换为密集矢量嵌入。
model.add(tf.keras.layers.Embedding(vocab_size, 64))


# In[35]:


# 下一层是 LSTM 层，它允许模型利用上下文中理解单词含义。 LSTM 上的双向包装器有助于模型理解当前数据点与其之前和之后的数据点的关系。
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))


# In[36]:


'''
最后，我们将获得一个或多个紧密连接的层，其中最后一层是输出层。输出层输出样本属于各个标签的概率，最后具有最高概率的分类标签即为最终预测结果。
'''


# In[44]:


# 一个或多个紧密连接的层
# 编辑 `for` 行的列表去检测层的大小
for units in [64, 64]:
#     model.add(tf.keras.layers.Dense(units, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units, activation='relu'))

# 输出层。第一个参数是标签个数。
model.add(tf.keras.layers.Dense(3, activation='softmax'))


# In[45]:


# 最后，编译这个模型。对于一个 softmax 分类模型来说，通常使用 sparse_categorical_crossentropy 作为其损失函数。
# 你可以尝试其他的优化器，但是 adam 是最常用的。
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#              loss=tf.keras.losses.sparse_categorical_crossentropy,
#              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[40]:


# 训练模型
# 利用提供的数据训练出的模型有着不错的精度（大约 83% ）。


# In[47]:


model.fit(train_data, epochs=3, validation_data=test_data)


# In[ ]:


eval_loss, eval_acc = model.evaluate(test_data)

