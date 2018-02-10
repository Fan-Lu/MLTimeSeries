# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 15:35:54 2018

@author: fanlu
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

#%%
speech = np.loadtxt('D:\GitHub_Repositories\MLTimeSeries\RLS\speech.txt')
size = np.size(speech)

M = 6  #Filter order
X = np.zeros([size, M])
R_Inv = 0.1*np.identity(M)
w = np.zeros([M, 1])
predict = np.zeros([size, 1])
error = np.zeros([size, 1])

for i in range(M):
    X[i:size, i] = speech[0:size-i]
    
for i in range(size):
    x_t = X[i, :].reshape(1, M)
    y_prio = x_t.dot(w)    #apriori output, predict value
    predict[i] = y_prio    #save predict value
    e_prio = speech[i] - y_prio  #apriori error
    error[i] = e_prio**2
    z = R_Inv.dot(x_t.T)     #filter information vector
    q = x_t.dot(z)         #normalized power
    v = 1/(1+q)             #gain
    z_norm = v*z            #normalized z
    w += e_prio * z_norm    #update optimal weight vector
    R_Inv = R_Inv - z_norm * z.T #updata autocorelation matrix

#%%
plt.plot(speech)
plt.plot(predict)

#sd.play(predict, 10000)