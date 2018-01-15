#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:07:06 2018

@author: leanderheuvel
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv
import random
import matplotlib.pyplot as plt

data_x = []
data_y = []
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
        data_x.append(digit)
        data_y.append(label)

np.asarray(data_x)
np.asarray(data_y)

combined = list(zip(data_x, data_y))
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

data_x, data_y = zip(*combined)
data_x_train = np.asarray(data_x[:1393])
data_y_train = np.asarray(data_y[:1393])
data_x_test  = np.asarray(data_x[200:])
data_y_test = np.asarray(data_y[200:])

data_y_train_ints = convert_labels(data_y_train)
data_y_test_ints = convert_labels(data_y_test)

clf = RandomForestClassifier(max_depth=40, n_jobs=-1, random_state=0, n_estimators=250)
clf.fit(data_x_train, data_y_train_ints)
'''
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', 
                       max_depth=2, max_features='auto', max_leaf_nodes=None, 
                       min_impurity_decrease=0.0, min_impurity_split=None, 
                       min_samples_leaf=1, min_samples_split=8, 
                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, 
                       oob_score=False, random_state=0, verbose=0, warm_start=False)
'''
feature_importances = clf.feature_importances_
predicted_ys = clf.predict(data_x_test)
print (predicted_ys)
accuracies = []
depths = []
for i in range(1,200,5):
    print(i)
    clf = RandomForestClassifier(max_depth=None, n_estimators=i, max_features=10, n_jobs=-1) 
    clf.fit(pj.data_x_train, data_y_train_ints)

    feature_importances = clf.feature_importances_
    predicted_ys = clf.predict(pj.data_x_test)
    #print (predicted_ys)

    j = 0
    for x in range(0,1393):
        if predicted_ys[x] == data_y_test_ints[x]:
            j+=1
    print (j/1393)
    accuracies.append(j/1393)
    depths.append(i)
plt.plot(depths,accuracies)
    
