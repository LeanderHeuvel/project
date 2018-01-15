#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:07:06 2018

@author: leanderheuvel
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import Semeion_data_loader as data



clf = RandomForestClassifier(max_depth=40, n_jobs=-1, random_state=0, n_estimators=250)
clf.fit(data.x_train, data.y_train_ints)
'''
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', 
                       max_depth=2, max_features='auto', max_leaf_nodes=None, 
                       min_impurity_decrease=0.0, min_impurity_split=None, 
                       min_samples_leaf=1, min_samples_split=8, 
                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, 
                       oob_score=False, random_state=0, verbose=0, warm_start=False)
'''
#print (clf.feature_importances_)
predicted_ys = clf.predict(data.x_test)
accuracies = []
depths = []
for i in range(1,200,5):
    #print(i)
    clf = RandomForestClassifier(max_depth=None, n_estimators=i, max_features=10, n_jobs=-1) 
    clf.fit(data.x_train, data.y_train_ints)

    predicted_ys = clf.predict(data.x_test)
    #print (predicted_ys)

    j = 0
    for x in range(0,1393):
        if predicted_ys[x] == data.y_test_ints[x]:
            j+=1
    #print (j/1393)
    accuracies.append(j/1393)
    depths.append(i)
plt.plot(depths,accuracies)
    
