# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:35:39 2019

@author: MHA
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from MHEstimator import MHEstimator as mhe

##  Turn image into windows...
def window(image,size):
    X=np.zeros((size[0]*size[1],1))
    for i in range(int(image.shape[0]/size[0])):
        for j in range(int(image.shape[1]/size[1])):
            window=image[size[0]*i:size[0]*(i+1),size[1]*j:size[1]*(j+1)]
            window=window.reshape((-1,1))
            X=np.concatenate((X,window),axis=1)
    X=X[:,1:]/255
    mean=np.sum(X,axis=1)
    mean=np.reshape(mean,(-1,1))
    mean/=X.shape[1]
    X-=mean
    return X, mean

##  Turn windows into image...
def decode(image, X, mean, size):
    counter=0
    for i in range(int(image.shape[0]/size[0])):
        for j in range(int(image.shape[1]/size[1])):
            window=X[:,counter:counter+1]
            window+=mean
            window=np.reshape(window,size)
            image[i*size[0]:(i+1)*size[0],j*size[1]:(j+1)*size[1]]=window*255
            counter+=1
    return image



##  Import image...            
image=cv2.imread('B.jpg')
##  Size of each window...
window_size=(8,8)
##  Turn image into windows...
X, mean= window(image[:,:,0], window_size)
##  Estimating covariance matrix of windows...
my_estimator=mhe(X)
est_mean, est_cov= my_estimator.gaussianEstimate()
##  Finding eigenvectors and eigenvalues of the covariance matrix...
eigvals, eigvecs=np.linalg.eig(est_cov)
sorted_index=np.argsort(eigvals)
sorted_index=sorted_index[::-1]
n=16  #  No of Principal Components to choose
##  Choosing PCAs...
chosen_eigs=np.zeros((window_size[0]*window_size[1],1))
for i in range(n):
    chosen_eigs=np.concatenate((chosen_eigs,eigvecs[:,sorted_index[i]:sorted_index[i]+1]),axis=1)
chosen_eigs=chosen_eigs[:,1:]
##  Projecting windows vectors into PCAs...
newX=np.transpose(np.dot(np.transpose(X),chosen_eigs))
##  Recovering image...
newX=np.dot(chosen_eigs,newX)
new_image=decode(image[:,:,0], newX, mean, window_size)
cv2.imshow('SPR_Method',new_image)