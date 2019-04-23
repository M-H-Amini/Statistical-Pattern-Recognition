# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 23:44:18 2019

@author: MHA
"""

import numpy as np
import cv2
from MHPCANN import MHPCANN as pca

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
            
image=cv2.imread('B.jpg')
window_size=(8,8)
max_iter=3e5
X0, mean=window(image[:,:,0],window_size)
my_pca=pca(X0,1e-3,64)
my_pca.train_thresh(1e-14,max_iter)
W0=my_pca.W
Y0=my_pca.output()
newX=(np.dot(W0.T,Y0))
compressed=np.zeros_like(image[:,:,0])
new0=decode(compressed, newX, mean, window_size)
###
X1, mean=window(image[:,:,1],window_size)
my_pca.X=X1
my_pca.train_thresh(1e-14,max_iter)
W1=my_pca.W
Y1=my_pca.output()
newX=(np.dot(W1.T,Y1))
compressed=np.zeros_like(image[:,:,1])
new1=decode(compressed, newX, mean, window_size)
###
X2, mean=window(image[:,:,2],window_size)
my_pca.X=X2
my_pca.train_thresh(1e-14,max_iter)
W2=my_pca.W
Y2=my_pca.output()
newX=(np.dot(W2.T,Y2))
compressed=np.zeros_like(image[:,:,2])
new2=decode(compressed, newX, mean, window_size)
###
new=np.zeros_like(image)
new[:,:,0]=new0
new[:,:,1]=new1
new[:,:,2]=new2
cv2.imshow('compressed1',new)
cv2.imwrite('1Result.jpg',new)
cv2.imshow('compressed2',new0)
cv2.imwrite('2Result.jpg',new)