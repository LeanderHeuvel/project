"""
Created on Thu Dec 14 14:16:10 2017

@author: leanderheuvel
"""

import numpy as np
import tensorflow as tf
import random
import Semeion_data_loader as data

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 256])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
 
x = tf.placeholder(tf.float32, [None, 256])
    
W = tf.Variable(tf.zeros([256, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
print('training model...')

def next_batch(size):
    batch_x = np.empty([100,256])
    batch_y = np.empty([100,10])
    for s in range(size):
        index = random.randint(0,1392)
        batch_x[s,] = data.x_train[index]
        batch_y[s,] = data.y_train[index]
    return batch_x, batch_y
        

for _ in range(1500):
    batch_xs, batch_ys = next_batch(100)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: data.x_test, y_: data.y_test}))