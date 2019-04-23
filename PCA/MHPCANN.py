# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:10:34 2019

@author: MHA
"""

import numpy as np

class MHPCANN:
    def __init__(self, X, learning_rate, output_dim):
        self.X=X
        self.etha=learning_rate
        self.W=np.random.randn(output_dim, X.shape[0])
        for i in range(self.W.shape[0]):
            mag=np.sqrt(np.dot(self.W[i:i+1,:],np.transpose(self.W[i:i+1,:])))
            self.W[i:i+1,:]/=mag
        self.output_dim=output_dim
        self.y=np.zeros((output_dim,1))
        
    def train_iter(self,no_of_iterations):
        for k in range(no_of_iterations):
            index=np.random.randint(0,self.X.shape[1])
            self.y=np.dot(self.W,self.X[:,index:index+1])
            dw=np.zeros_like(self.W)
            for j in range(self.W.shape[0]):
                dw[j:j+1,:]=-self.etha*self.y[j,0]*np.dot(np.transpose(self.y[:j+1,:]),(self.W[:j+1,:]))
            dw+=self.etha*np.dot(self.y,np.transpose(self.X[:,index:index+1]))
            self.W+=dw
        print('W',self.W)

    def train_thresh(self,thresh, max_iter):
        old_var=1000
        iterations=0
        while True:
            iterations+=1
            index=np.random.randint(0,self.X.shape[1])
            self.y=np.dot(self.W,self.X[:,index:index+1])
            dw=np.zeros_like(self.W)
            for j in range(self.W.shape[0]):
                dw[j:j+1,:]=-self.etha*self.y[j,0]*np.dot(np.transpose(self.y[:j+1,:]),(self.W[:j+1,:]))
            dw+=self.etha*np.dot(self.y,np.transpose(self.X[:,index:index+1]))
            self.W+=dw
            new_var=np.sum(dw)
            if abs(new_var-old_var)<thresh:
                break
            else:
                old_var=new_var
            if int(iterations/1e4)==iterations/1e4:
                print('{} iterations done...'.format(iterations))
            if iterations>max_iter:
                break
        print('iterations',iterations-1)
    
    def output(self):
        return np.dot(self.W,self.X)
        
if __name__=='__main__':
    X=np.array([[1,2,3],[4,5,6],[1,5,9]])
    obj=MHPCANN(X,1e-4,2)
    #obj.train_iter(10000)
    obj.train_thresh(1e-9,1e4)