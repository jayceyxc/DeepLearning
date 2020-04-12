#!/usr/bin/env python3
# @Time    : 2018/10/7 8:00 AM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : L2_norm_regularization.py
# @Software: PyCharm
# @Description Ridge模型在4次多项式特征上的拟合表现

# 从sklearn.preproessing中导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures
# 从sklearn.linear_model中导入Lasso
from sklearn.linear_model import Ridge

import numpy as np

# 输入训练样本的特征以及目标值，分别存储在变量X_train与y_train之中。
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

# 准备测试数据。
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

# 初始化4次多项式特征生成器。
poly4 = PolynomialFeatures(degree=4)

X_train_poly4 = poly4.fit_transform(X_train)
X_test_poly4 = poly4.transform(X_test)
print(X_train_poly4)

# 使用默认配置初始化Lasso。
ridge_poly4 = Ridge()
# 使用Ridge对4次多项式特征进行拟合。
ridge_poly4.fit(X_train_poly4, y_train)

# 对Ridge模型在测试样本上的回归性能进行评估。
print(ridge_poly4.score(X_test_poly4, y_test))

# 输出Ridge模型的参数列表。
print(ridge_poly4.coef_)

# 计算Ridge模型拟合后参数的平方和。
print(np.sum(ridge_poly4.coef_ ** 2))


