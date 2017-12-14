#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:16:10 2017

@author: leanderheuvel
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import tensorflow as tf

data = []
data_labels = []
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
        data.append(digit)
        data_labels.append(label)

#[float(i) for i in data]
x=16
y=16
plt.imshow(np.reshape(data[170],(x,y)))
def convert_labels(data_labels):
    text_labels = []
    for digit in data_labels:
        i=0
        for value in digit:
            if value==1:
                text_labels.append(i)
            i+=1
    return text_labels
text_labels = convert_labels(data_labels)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

