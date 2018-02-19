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
x = np.loadtxt('D:\MyGitHub\MLTimeSeries\RLS\input.txt')  #input data
y = np.loadtxt('D:\MyGitHub\MLTimeSeries\RLS\output.txt') #unknown plant output data
n = np.random.normal(0, 1.5, 10000) #noise
d = y + n  #output with gaussian noise, used as desired of filter

size = np.size(x)
predict = np.zeros([size, 1])
error = np.zeros([size, 1])

#%%
M = 40  #Filter order
X = np.zeros([10000, M])  #IIR pre-processing
w = np.zeros([M, 1])
sigma = np.cov(x[0:100])
R_Inv = 0.01*np.identity(M)
w_star = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
w_star = w_star.reshape([10 ,1])
w_track = np.zeros([10000, M])

for i in range(M):
    X[(M-i-1):10000, i] = x[0:(10000-(M-i-1))]


for i in range(10000):
    x_t = X[i, :].reshape(1, M)
    y_prio = x_t @ w    #apriori outpvalue
    predict[i] = y_prio    #save predict value
    e_prio = d[i] - y_prio  #apriori error
    error[i] = (y_prio - d[i])**2
    z = R_Inv @ (x_t.T)     #filter information vector
    q = x_t @ z        #normalized power
    v = 1/(1 + q)             #gain
    z_norm = v*z            #normalized z
    w = w + e_prio * z_norm    #update optimal weight vector
    w_track[i, :] = w.reshape(M);
    R_Inv = R_Inv - z_norm * z.T #updata autocorelation matrixut, predict 

if M < np.size(w_star):
    zero_pad = np.zeros([(10-M), 1])
    w_reshape = np.vstack((w, zero_pad))
    WSNR = 10*np.log(w_star.T@w_star/(w_star-w_reshape).T@(w_star-w_reshape))
else:
    zero_pad = np.zeros([(M-10), 1])
    w_reshape = np.vstack((w_star, zero_pad))
    WSNR = 10*np.log(w_star.T@w_star/(w-w_reshape).T@(w-w_reshape))
print(WSNR)
#%%
#plt.plot(y)
#plt.plot(error)
#plt.plot(w_track)
#plt.title("Weight Track")
#plt.xlabel("Epoch")
#plt.ylabel("Weights")

#%%
#Learning Curve
plt.plot(error)
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE")

#%%
#plt.plot(d)
#plt.plot(predict)