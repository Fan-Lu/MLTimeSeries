# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 11:12:16 2018

@author: fanlu
"""

#%%
#import needed libraries
import numpy as np
import matplotlib.pyplot as plt

#%%
#prepare data
x = np.loadtxt('D:\GitHub_Repositories\MLTimeSeries\RLS\input.txt')  #input data
y = np.loadtxt('D:\GitHub_Repositories\MLTimeSeries\RLS\output.txt') #unknown plant output data
n = np.random.normal(0, 0.1, 10000) #noise
d = y + n  #output with gaussian noise, used as desired of filter

size = np.size(x)
predict = np.zeros([size, 1])
error = np.zeros([size, 1])

#%%
M = 5  #Filter order
X = np.zeros([10000, M])  #IIR pre-processing
w = np.zeros([M, 1])
R_Inv = 0.01*np.identity(M)

for i in range(M):
    X[i:10000, i] = x[0:(10000-i)]

for i in range(10000):
    x_t = X[i, :].reshape(1, M)
    y_prio = x_t.dot(w)    #apriori output, predict value
    predict[i] = y_prio    #save predict value
    e_prio = d[i] - y_prio  #apriori error
    error[i] = e_prio**2
    z = R_Inv.dot(x_t.T)     #filter information vector
    q = x_t.dot(z)         #normalized power
    v = 1/(1+q)             #gain
    z_norm = v*z            #normalized z
    w += e_prio * z_norm    #update optimal weight vector
    R_Inv = R_Inv - z_norm * z.T #updata autocorelation matrix
    
#%%
plt.plot(y)
plt.plot(predict)