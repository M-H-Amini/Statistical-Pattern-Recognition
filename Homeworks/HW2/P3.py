# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:46:31 2019

@author: MHA
"""

from MHBayesian import MHBayesian as mhb
from MHEstimator import MHEstimator as mhe
import numpy as np

##  Input featurs...
X=[[69,116],[88,132],[104,157],[25,22],[99,145],[208,30],[113,19],[159,11],[199,19],[249,141],[226,231],[133,131],[218,214],[186,176],[207,202]]
##  Classes...
y=[int(i/5) for i in range(15)]
classes={0: 'Sabzeh', 1: 'Ribbon', 2: 'Background'}

means=[]
for i in range(3):
    class_dataset=[X[j] for j in range(len(X)) if y[j]==i]
    class_dataset=np.array(class_dataset)
    class_dataset=class_dataset.T
    estimator=mhe(class_dataset)
    mean, covariance = estimator.gaussianEstimate()
    means.append(mean)

bayesian=mhb([0],0)
test=np.array([[196,34],[195,180],[255,88],[125,110],[110,146]])
test=test.T
for i in range(test.shape[1]):
    current_test=test[:,i]
    print('**********Test{}**********'.format(i+1))
    print('Input...')
    print(current_test)
    print('Predicted Class...')
    print(classes[bayesian.MDC(means,current_test)[0]])