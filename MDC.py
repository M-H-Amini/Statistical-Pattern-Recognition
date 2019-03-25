# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:19:28 2019

@author: MHA
"""
import numpy as np

def euclideanDistance(a,b):
    dimension=len(a)
    summation=0
    for i in range(dimension):
        summation+=(a[i]-b[i])*(a[i]-b[i])
    return np.sqrt(summation)
        

def MDC(means,x):
    classes=len(means)
    minimum=1e12
    arg_min=-1
    for i in range(classes):
        temp_dist=euclideanDistance(means[i],x)
        if temp_dist<minimum:
            minimum=temp_dist
            arg_min=i
    return arg_min, minimum

if __name__=='__main__':
    means=[[1,0,-5],[0,5,2]]
    print(MDC(means,[2,0,7]))
    


    