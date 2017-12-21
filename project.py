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
import random

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

combined = list(zip(data, data_labels))
random.shuffle(combined)

data_x, data_y = zip(*combined)
data_x_train = np.asarray(data_x[:200])
data_y_train = np.asarray(data_y[:200])
data_x_test  = np.asarray(data_y[200:])
data_x_train = np.asarray(data_y[200:])
#plt.imshow(np.reshape(data[170],(x,y)))
def convert_labels(data_labels):
    text_labels = []
    for digit in data_labels:
        i=0
        for value in digit:
            if value==1:
                text_labels.append(i)
            i+=1
    return text_labels
#text_labels = convert_labels(data_labels)

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 256])

W = tf.Variable(tf.zeros([256, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
print('training model...')
for _ in range(1000):
  #batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: data_x, y_: data_y})
  
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))