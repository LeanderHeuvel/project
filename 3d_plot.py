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
from mpl_toolkits.mplot3d import Axes3D



clf = RandomForestClassifier(max_depth=40, n_jobs=-1, random_state=0, n_estimators=250)
clf.fit(data.x_train, data.y_train_ints)
NR_ITERS = 20 #one iteration is run entirely over the train and test sets
estimators = [] #same as features
#for _ in range (0,NR_ITERS):
accuracies = np.empty((10,10)) # list of the accuracies of one iteration
for k in range(1,200,20):  
    #TODOO: doe op een of andere manier de accuracies los voor estimators en features, maar hoe dannn? en wat is dan op de Z as? idk m8
    for i in range(1,200,20): #grotere stappen want voor 3dplot moet alles dezelfde dimensie hebben
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
        #accuracies.append(j/1393)
        k_ind = int((k-1)/20)
        i_ind = int((i-1)/20)
        accuracies[k_ind][i_ind] = (j/1393)
        if len(estimators) < 10:
            estimators.append(i)
    
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

    plt.plot(estimators, avg_accs)
    plt.x_label("nr. of trees")
    #todo, plot fatsoenlijk mooi maken
    '''
'''
X = estimators,features
Y = features,estimators
Z = accuracies
'''

X = np.array([estimators, estimators, estimators, estimators, estimators, estimators, estimators, estimators, estimators, estimators]) #estimators array would be the same as the features array and is thus treated as one and the same codewise
Y = np.array([estimators, estimators, estimators, estimators, estimators, estimators, estimators, estimators, estimators, estimators])
Z = accuracies

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X,Y,Z, rstride=10, cstride=10)