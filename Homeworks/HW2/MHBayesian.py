# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:15:17 2019

@author: MHA
"""

import numpy as np

class MHBayesian:
    def __init__(self, likelihoods ,priors=0):
        ##  priors=0 means equiprobable case!...
        self.likelihoods=likelihoods
        if priors:
            self.priors=priors
        else:
            self.priors=[1/len(self.likelihoods) for i in range(len(self.likelihoods))]
        
    def classify(self,x,details=False):
        posteriors=[]
        for i in range(len(self.priors)):
            posteriors.append(self.likelihoods[i](x)*self.priors(i))
        if details:
            print('Poseteriors:\n',posteriors)
        return np.argmax(posteriors)

    def euclideanDistance(self,a,b):
        dimension=len(a)
        summation=0
        for i in range(dimension):
            summation+=(a[i]-b[i])*(a[i]-b[i])
        return np.sqrt(summation)
            
    def MDC(self,means,x):
        classes=len(means)
        minimum=1e12
        arg_min=-1
        for i in range(classes):
            temp_dist=self.euclideanDistance(means[i],x)
            if temp_dist<minimum:
                minimum=temp_dist
                arg_min=i
        return arg_min, minimum
