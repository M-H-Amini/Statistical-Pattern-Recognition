# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:25:45 2019

@author: MHA
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class MHEstimator:
    def __init__(self, dataset, estimator_type='ML'):
        self.dataset=dataset
        self.no_of_features=self.dataset.shape[0]
        self.no_of_samples=self.dataset.shape[1]
        self.estimator_type=estimator_type
        self.mean=np.zeros((self.no_of_features,1))
        self.covariance=np.zeros((self.no_of_features,self.no_of_features))
        
    def gaussianEstimate(self,plot=False):
        ##  Estimating mean...
        for i in range(self.no_of_samples):
            self.mean+=self.dataset[:,i]
        self.mean/=self.no_of_samples
        
        print('Mean: \n',self.mean)
        
        ##  Estimating covariance...
        for i in range(self.no_of_samples):
            self.covariance+=(self.dataset[:,i]-self.mean)*np.transpose((self.dataset[:,i]-self.mean))
        self.covariance/=self.no_of_samples
        
        print('Covariance: \n',self.covariance)
        
        ##  Plotting the distribution...
        if plot:
            ##  1-d case...
            if not self.no_of_features-1:
                x=np.linspace(float(self.mean-3*np.sqrt(float(self.covariance)))
                , float(self.mean+3*np.sqrt(float(self.covariance))),100)
                plt.figure()
                plt.title(("Distribution of dataset\n $\mu={}$, "
                          "$\sigma^2={}$").format(float(self.mean),float(self.covariance)))
                plt.plot(x,stats.norm.pdf(x,float(self.mean)
                    ,np.sqrt(float(self.covariance))))
                for i in range(self.no_of_samples):
                    plt.plot(self.dataset[0,i],stats.norm.pdf(self.dataset[0,i]
                    ,float(self.mean),np.sqrt(float(self.covariance))),'rX')
        
        return self.mean, self.covariance
            
    def displayDataset(self,description='rx'):
        plt.figure()
        plt.title('Dataset')
        if self.no_of_features==1:
            for i in range(self.no_of_samples):
                plt.plot(self.dataset[0,i],0,description)
                plt.grid()
        elif self.no_of_features==2:
            for i in range(self.no_of_samples):
                plt.plot(self.dataset[0,i],self.dataset[1,i],description)
                plt.grid()
        else:
            print('No display for more than 2 features!')
            
if __name__=='__main__':
    dataset1=np.matrix('19,18,20,19.5,17',dtype=float)
    dataset2=np.matrix('19,18,20,19.5,17;40,31,25,49,10',dtype=float)
    
    my_estimator=MHEstimator(dataset1)
    #my_estimator.displayDataset()
    my_estimator.gaussianEstimate(True)
    