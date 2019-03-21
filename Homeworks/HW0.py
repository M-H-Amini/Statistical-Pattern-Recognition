# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:18:02 2019

@author: MHA
"""
import numpy as np

class ThreeValuedBayesianClassifier:
    def __init__(self, w1, w2, p1, p2): 
        self.w1=w1  #  prior probability of class 1
        self.w2=w2  #  prior probability of class 2
        self.p1=p1  #  conditional probability of X given it is of class 1
        self.p2=p2  #  conditional probability of X given it is of class 2
        self.threshold= np.log10(w2/w1)
        
    def m(self, x):
        return x*(x+1)*(x-0.5)
    def n(self, x):
        return (x+1)*(x-1)
    def o(self, x):
        return -x*(x-1)*(x+0.5)
    
    #  likelihood...
    def l(self, X):
        dimension=X.shape[0]
        summation=0
        for i in range(dimension):
            summation+=self.m(X[i])*np.log10(self.p1[0]/self.p2[0])\
                      +self.n(X[i])*np.log10(self.p1[1]/self.p2[1])\
                      +self.o(X[i])*np.log10(self.p1[2]/self.p2[2])
        return summation
    
    def classify(self,X):
        if self.l(X)>=self.threshold:
            return 1
        else:
            return 2
        
        
if __name__=='__main__':
    p1=[0.3, 0.3, 0.4]
    p2=[0.1, 0.7, 0.2]
    w1=0.2
    w2=0.8
    classifier=ThreeValuedBayesianClassifier(w1, w2, p1, p2)
    X=np.array([1, -1, 1, -1, 0])
    print(classifier.classify(X))
        
        
        