#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:07:06 2018

@author: leanderheuvel
"""

import project as pj
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

data_y_train_ints = pj.convert_labels(pj.data_y_train)
data_y_test_ints = pj.convert_labels(pj.data_y_test)

accuracies = []
depths = []
for i in range(1,200,5):
    print(i)
    clf = RandomForestClassifier(max_depth=None, n_estimators=i, max_features=10) 
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
    
