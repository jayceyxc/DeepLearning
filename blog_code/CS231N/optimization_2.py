#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-12-01 10:30
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : optimization_2.py
# @Software: PyCharm
# @Description 代码来源：http://cs231n.github.io/optimization-2/

import math

# back propagation example. sigmoid function
w = [2, -3, -3]
x = [-1, -2]

# forward pass
dot = w[0] * x[0] + w[1] * x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot))  # sigmoid function


