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

####################関数####################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#バイアスの初期化
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#畳み込み層(ストライド=1,パディングの値=0)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#max-pooling(2*2サイズ)
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,256],order='F'),keep_prob:1.0})
    plotNNFilter(units)
    
def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

####################main####################



def next_batch(size):
    batch_x = np.empty([size,256])
    batch_y = np.empty([size,10])
    for s in range(size):
        index = random.randint(0,1392)
        batch_x[s,] = data.x_train[index]
        batch_y[s,] = data.y_train[index]
    return batch_x, batch_y
        
with tf.name_scope("input_x") as scope:
    x = tf.placeholder(tf.float32, shape=[None, 256])

##1次元から4次元に変換(-1,縦,横,チャンネル)
with tf.name_scope("input_x_image") as scope:
    x_image = tf.reshape(x, [-1,16,16,1])
    tf.summary.image('input', x_image, 10)

with tf.name_scope("label_y_") as scope:
    y_ = tf.placeholder(tf.float32, shape=[None, 10])


###第1畳み込み層&max-pooling###
with tf.name_scope("conv1") as scope:
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

with tf.name_scope("pool1") as scope:
    h_pool1 = max_pool_2x2(h_conv1)


###第2畳み込み層&max-pooling###
with tf.name_scope("conv2") as scope:
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

with tf.name_scope("pool2") as scope:
    h_pool2 = max_pool_2x2(h_conv2)


###fc1(全結合層)###
with tf.name_scope("fc1") as scope:
    W_fc1 = weight_variable([4 * 4 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    #４次元のh_poo2を1次元の[-1,7*7*64]に変換
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64])
    #内積を計算してバイアスを加算後、ReLU関数を適用
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


###drop out###
with tf.name_scope("dropout") as scope:
    keep_prob = tf.placeholder(tf.float32)
    #読み出し層の１つ前にドロップアウトを実装
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


###出力層###
with tf.name_scope("output_y") as scope:
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10]) 
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

###train###
with tf.name_scope("cross_entropy") as scope:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #scalarの追加
    cross_entropy_summary = tf.summary.scalar("cross_entropy", cross_entropy)

#AdamOptimizerを使用
with tf.name_scope('train'):    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


with tf.name_scope("accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #scalarの追加
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

#histgramの追加
w_conv1_hist = tf.summary.histogram("conv1/w", W_conv1)
b_conv1_hist = tf.summary.histogram("conv1/b", b_conv1)
acivaions_conv1_hist = tf.summary.histogram("conv1/activations", h_conv1)
w_conv2_hist = tf.summary.histogram("conv2/w", W_conv2)
b_conv2_hist = tf.summary.histogram("conv2/b", b_conv2)
acivaions_conv2_hist = tf.summary.histogram("conv2/activations", h_conv2)
w_fc1_hist = tf.summary.histogram("fc1/w", W_fc1)
b_fc1_hist = tf.summary.histogram("fc1/b", b_fc1)
acivaions_fc1_hist = tf.summary.histogram("fc1/activations", h_fc1)
w_fc2_hist = tf.summary.histogram("fc2/w", W_fc2)
b_fc2_hist = tf.summary.histogram("fc2/b", b_fc2)

y_hist = tf.summary.histogram("y", y)



#作成したモデルを開始
sess = tf.InteractiveSession()

#すべてのsummariesをmerge
merged = tf.summary.merge_all()
#log network

writer = tf.summary.FileWriter("/Users/leanderheuvel/Documents/Session1", sess.graph)
sess.run(tf.global_variables_initializer())


for i in range(100):
    batch_xs, batch_ys = next_batch(50)
    if i % 100 == 0:
        result = sess.run([merged,accuracy],feed_dict={x: batch_xs, y_: batch_ys,keep_prob:1.0})     
        print("step %d, training accuracy %g" % (i, result[1]))
        writer.add_summary(result[0], i)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

##########################test##########################

print("test accuracy %g" % accuracy.eval(feed_dict={x: data.x_test, y_: data.y_test, keep_prob: 1.0}))

imageToUse = data.x_test[1]
plt.imshow(np.reshape(imageToUse,[16,16]), interpolation="nearest", cmap="gray")
plt.show()
getActivations(W_conv2,imageToUse)

print('done')
