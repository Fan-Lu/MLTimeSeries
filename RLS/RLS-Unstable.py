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
speech = np.loadtxt('D:\MyGitHub\MLTimeSeries\RLS\speech.txt')
size = np.size(speech)

M = 6  #Filter order
X = np.zeros([size, M])
sigma = np.cov(speech[0:100])
R_Inv = sigma*np.identity(M)
w = np.zeros([M, 1])
predict = np.zeros([size, 1])
error = np.zeros([size, 1])
lam = 0.99 #forgeting factor
mse = np.zeros([4, 1])
w_track = np.zeros([size, M])

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
    v = 1/(lam+q)             #gain
    z_norm = v*z            #normalized z
    w += e_prio * z_norm    #update optimal weight vector
    R_Inv = R_Inv/lam - z_norm * z.T/lam #updata autocorelation matrix
    w_track[i, :] = w.reshape(M);

MSE = np.mean(error) 
print(MSE)
#%%
#mse[0] = 1.08258825787e-06
#mse[1] = 1.21218641772e-06
#mse[2] = 1.55997661104e-06
#mse[3] = 3.69614513726e-06
#
#
#b = np.zeros([4, 1])
#b[0] = 0.96
#b[1] = 0.97
#b[2] = 0.98
#b[3] = 0.99

#plt.plot(b, mse)
#plt.title("MSE ovre different forgetting factor")
#plt.xlabel("Forgetting factor")
#plt.ylabel("MSE")
#plt.plot(speech)
#plt.title("Original Speech")
#plt.plot(speech)
plt.plot(w_track)
sd.play(predict, 10000)