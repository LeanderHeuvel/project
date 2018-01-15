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
NR_ITERS = 20 #one iteration is run entirely over the train and test sets
depths = []
all_accs = [] # list of all accuracies over all iterations to calculate averages
#for _ in range (0,NR_ITERS):
for k in range(1,200,20):  
    accuracies = [] # list of the accuracies of one iteration
    for i in range(1,200,5):
    #print(i)
        clf = RandomForestClassifier(max_depth=None, n_estimators=k, max_features=i, n_jobs=-1) 
        clf.fit(data.x_train, data.y_train_ints)
        
        predicted_ys = clf.predict(data.x_test)
        #print (predicted_ys)

        j = 0 #to calculate accuracy
        for x in range(0,1393):
            if predicted_ys[x] == data.y_test_ints[x]:
                j+=1
            #print (j/1393)
        accuracies.append(j/1393)
        if len(depths) < 40:
            depths.append(i)
   # feature_importances = clf.feature_importances_
    #plt.imshow(np.reshape(feature_importances,(16,16)))
    #plt.show()
    '''
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

    plt.plot(depths, avg_accs)
    plt.x_label("nr. of trees")
    #todo, plot fatsoenlijk mooi maken
    '''

    plt.plot(depths, accuracies)
plt.show()