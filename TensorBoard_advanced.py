#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:11:02 2018

@author: leanderheuvel
"""
import random
import Semeion_data_loader as data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data

#weight function
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#bias function
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#convolution
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#max pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,256],order='F'),keep_prob:1.0})
    plotNNFilter(units)
    


# pull a batch from the Semion data set
def next_batch(size):
    batch_x = np.empty([size,256])
    batch_y = np.empty([size,10])
    for s in range(size):
        index = random.randint(0,1392)
        batch_x[s,] = data.x_train[index]
        batch_y[s,] = data.y_train[index]
    return batch_x, batch_y
 
#monitoring  CNN using tensorboard api       
with tf.name_scope("input_x") as scope:
    x = tf.placeholder(tf.float32, shape=[None, 256])


with tf.name_scope("input_x_image") as scope:
    x_image = tf.reshape(x, [-1,16,16,1])
    tf.summary.image('input', x_image, 10)

with tf.name_scope("label_y_") as scope:
    y_ = tf.placeholder(tf.float32, shape=[None, 10])


with tf.name_scope("conv1") as scope:
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

with tf.name_scope("pool1") as scope:
    h_pool1 = max_pool_2x2(h_conv1)



with tf.name_scope("conv2") as scope:
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

with tf.name_scope("pool2") as scope:
    h_pool2 = max_pool_2x2(h_conv2)


###fully connected layer
with tf.name_scope("fc1") as scope:
    W_fc1 = weight_variable([4 * 4 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#dropout, prevent overfitting
with tf.name_scope("dropout") as scope:
    keep_prob = tf.placeholder(tf.float32)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


###fully connected layer 2
with tf.name_scope("output_y") as scope:
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10]) 
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

###training###
with tf.name_scope("cross_entropy") as scope:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #scalarの追加
    cross_entropy_summary = tf.summary.scalar("cross_entropy", cross_entropy)

#AdamOptimizer
with tf.name_scope('train'):    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


with tf.name_scope("accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

#creating histograms
b_conv1_hist = tf.summary.histogram("conv1_b", b_conv1)
w_conv1_hist = tf.summary.histogram("conv1_w", W_conv1)
acivaions_conv1_hist = tf.summary.histogram("conv1/activations", h_conv1)
w_conv2_hist = tf.summary.histogram("conv2/w", W_conv2)
b_conv2_hist = tf.summary.histogram("conv2/b", b_conv2)
activations_conv2_hist = tf.summary.histogram("conv2/activations", h_conv2)
w_fc1_hist = tf.summary.histogram("fc1/w", W_fc1)
b_fc1_hist = tf.summary.histogram("fc1/b", b_fc1)
activations_fc1_hist = tf.summary.histogram("fc1/activations", h_fc1)
w_fc2_hist = tf.summary.histogram("fc2/w", W_fc2)
b_fc2_hist = tf.summary.histogram("fc2/b", b_fc2)

y_hist = tf.summary.histogram("y", y)



#start Session
sess = tf.InteractiveSession()

#Merge all tensorboard parameters
merged = tf.summary.merge_all()
#log network

writer = tf.summary.FileWriter("/Users/leanderheuvel/Documents/Session1", sess.graph)
sess.run(tf.global_variables_initializer())

print('training model...')
accuracies=[]

for i in range(100):
    batch_xs, batch_ys = next_batch(50)
    if i % 100 == 0:
        result = sess.run([merged,accuracy],feed_dict={x: batch_xs, y_: batch_ys,keep_prob:1.0})     
        print("step %d, training accuracy %g" % (i, result[1]))
        
        writer.add_summary(result[0], i)
        accuracies.append(accuracy.eval(feed_dict={ x: data.x_test, y_: data.y_test, keep_prob: 1.0}))    
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

#run test
print("done training)
print("test accuracy %g" % accuracy.eval(feed_dict={x: data.x_test, y_: data.y_test, keep_prob: 1.0}))
plt.plot(accuracies)
plt.show()

print('done')
