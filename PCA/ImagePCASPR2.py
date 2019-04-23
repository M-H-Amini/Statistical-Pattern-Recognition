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

def PCADerivation(X, n, details=False):
    ##  n --> no of Principal Components to keep...
    ##  Estimating covariance matrix of windows...
    my_estimator=mhe(X)
    est_mean, est_cov= my_estimator.gaussianEstimate()
    ##  Finding eigenvectors and eigenvalues of the covariance matrix...
    eigvals, eigvecs=np.linalg.eig(est_cov)
    sorted_index=np.argsort(eigvals)
    sorted_index=sorted_index[::-1]
    ##  Choosing PCAs...
    chosen_eigs=np.zeros((X.shape[0],1))
    for i in range(n):
        chosen_eigs=np.concatenate((chosen_eigs,eigvecs[:,sorted_index[i]:sorted_index[i]+1]),axis=1)
    chosen_eigs=chosen_eigs[:,1:]
    ##  Projecting windows vectors into PCAs...
    newX=np.transpose(np.dot(np.transpose(X),chosen_eigs))
    ##  Recovering image...
    newX=np.dot(chosen_eigs,newX)    
    if details:
        return newX, [eigvals[i] for i in sorted_index]
    return newX
##  Plot eigenvalues of the covariance matrix...
def plotEigvals(eigvals, logscale=True):
    n=[i for i in range(len(eigvals))]
    plt.figure()
    if logscale:
        plt.plot(n,20*np.log10(eigvals),'rx')
    else:
        plt.plot(n,eigvals,'rx')

##  Import image...            
image=cv2.imread('B.jpg')
##  Size of each window...
window_size=(8,8)
##  Turn image into windows...
X0, mean0= window(image[:,:,0], window_size)
X1, mean1= window(image[:,:,1], window_size)
X2, mean2= window(image[:,:,2], window_size)
##  Derivation of PCAs...
plt.figure()
plt.title('Statistical Approach to PCA')
n=[1, 8, 16, 32]
for i in range(len(n)):
   plt.subplot(2,2,i+1)
   plt.xticks([])
   plt.yticks([])
   plt.title('n={} Principal Components'.format(n[i]))
   newX0, eigvals0=PCADerivation(X0, n[i], details=True)
   newX1, eigvals1=PCADerivation(X1, n[i], details=True)
   newX2, eigvals2=PCADerivation(X2, n[i], details=True)
   newX0=decode(image[:,:,0], newX0, mean0, window_size)
   newX1=decode(image[:,:,1], newX1, mean1, window_size)
   newX2=decode(image[:,:,2], newX2, mean2, window_size)
   newImage=np.zeros_like(image)
   newImage[:,:,0]=newX0
   newImage[:,:,1]=newX1
   newImage[:,:,2]=newX2
   plt.imshow(cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))
   cv2.imwrite('SPR-Result{}.png'.format(n[i]),newImage)

plotEigvals(eigvals0)