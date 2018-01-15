# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:07:32 2018

@author: loesj
"""

import csv
import random
import numpy as np


x = []
y = []
with open('semeion.data.txt') as inputfile:
    for row in csv.reader(inputfile):
        i=0
        digit = []
        label = []
        for element in row[0].strip().split(' '):
            if i<256:
                digit.append(float(element))
            else:
                label.append(float(element))
            i+=1
        x.append(digit)
        y.append(label)

np.asarray(x)
np.asarray(y)

combined = list(zip(x, y))
random.shuffle(combined)

def convert_labels(data_labels):
    text_labels = []
    for digit in data_labels:
        i=0
        for value in digit:
            if value==1:
                text_labels.append(i)
            i+=1
    return text_labels

x, y = zip(*combined)
x_train = np.asarray(x[:1393])
y_train = np.asarray(y[:1393])
x_test  = np.asarray(x[200:])
y_test = np.asarray(y[200:])

y_train_ints = convert_labels(y_train)
y_test_ints = convert_labels(y_test)