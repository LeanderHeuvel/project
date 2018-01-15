#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:07:06 2018

@author: leanderheuvel
"""

import project as pj
from sklearn.ensemble import RandomForestClassifier


data_y_train_ints = pj.convert_labels(pj.data_y_train)
data_y_test_ints = pj.convert_labels(pj.data_y_test)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(pj.data_x_train, data_y_train_ints)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', 
                       max_depth=2, max_features='auto', max_leaf_nodes=None, 
                       min_impurity_decrease=0.0, min_impurity_split=None, 
                       min_samples_leaf=1, min_samples_split=2, 
                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, 
                       oob_score=False, random_state=0, verbose=0, warm_start=False)
feature_importances = clf.feature_importances_
predicted_ys = clf.predict(pj.data_x_test)
print (predicted_ys)

i = 0
for x in range(0,1393):
    if predicted_ys[x] == data_y_test_ints[x]:
        i+=1
print (i/1393)
