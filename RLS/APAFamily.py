# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 08:10:05 2018

@author: fanlu
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

#%%
speech = np.loadtxt('D:\MyGitHub\MLTimeSeries\RLS\speech.txt')
size = np.size(speech)
fs = 100000

#%%
k = 10;
x = np.zeros([size, k])
w = np.zeros([k, 1])
eta = 0.001 #learning rate
w_track = np.zeros([10000, M])


for i in range(k):
    x[i:size, i] = speech[0:(size-i)]
    
#%%
err_save = np.zeros([size, 1])
for i in range(size):
    x_t = x[i, :].reshape([1, k])
    pred = np.dot(x_t, w)
    err = speech[i] - pred
    err_save[i] = err**2
    w = w + eta * err * x_t.T
    
    
#%%
plt.plot(err_save)