#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:07:06 2018

@author: loeserven
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import Semeion_data_loader as data


NR_ITERS = 20 #one iteration is run entirely over the train and test sets
estimators = []
all_accs = [] # list of all accuracies over all iterations to calculate averages
for iteration in range (0,NR_ITERS):
    accuracies = [] # list of the accuracies of one iteration
    for i in range(1,200,5):
    	#for running over the number of trees:
        clf = RandomForestClassifier(max_depth=None, n_estimators=i, max_features=15, n_jobs=-1) 
        #for running over the maximum features:
        #clf = RandomForestClassifier(max_depth=None, n_estimators=120, max_features=i, n_jobs=-1)
        #use one of the "clf = [...]" lines
        clf.fit(data.x_train, data.y_train_ints)
        
        predicted_ys = clf.predict(data.x_test)
        #print (predicted_ys)

        j = 0.0 #to calculate accuracy
        for x in range(0,1393):
            if predicted_ys[x] == data.y_test_ints[x]:
                j+=1
                #print (j/1393)
        accuracies.append(j/1393)
        if len(estimators) < 40:
            estimators.append(i)
    print(iteration)
    
# for showing the grid of feature importances:    
#feature_importances = clf.feature_importances_
#plt.imshow(np.reshape(feature_importances,(16,16)))
#plt.show()

    all_accs.append(accuracies)
all_accs = np.array(all_accs)
all_accs = all_accs.T
all_accs = all_accs.tolist()
    
avg_accs = []
for i in range(0,40):
    avg_accs.append(sum(all_accs[i])/NR_ITERS)
avg_accs = np.array(avg_accs)
avg_accs = avg_accs.T
avg_accs = avg_accs.tolist()

plt.plot(estimators, avg_accs)