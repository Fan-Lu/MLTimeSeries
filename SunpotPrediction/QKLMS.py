# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:49:22 2018

@author: fanlu
"""

#%%
#import needed libraries
import numpy as np
import matplotlib.pyplot as plt

#%%
def loadCSVfile(path):
    tmp = np.loadtxt(path, dtype = np.str, delimiter = ",")
    data = tmp[1:, 3].astype(np.float)
    return data

#%%
SN = loadCSVfile("D:\MyGitHub\MLTimeSeries\SunpotPrediction\SN_m_tot_V2.0.csv")
#D = SN.reshape(np.size(SN), 1)
#plt.plot(D)

M = 5  #Filter Order
D = 2  #delay
X_In = np.zeros([len(SN), M])
for j in range(M):
    for i in range(len(SN)-M+j+1):
        X_In[i+M-j-1][j] = SN[i]

#%%
def get_postion(array, dismin):
    for i in range(len(array)):
        if (array[i] == dismin):
            return i
        
def QKLMS(X, eta3, sigma, q_size):
    qpred = np.zeros(np.size(X, 0))
    qc = np.zeros([np.size(X, 0), M])
    qc[0, :] = X[0, :].reshape([M, 1]).reshape(5)
    qa = np.zeros(np.size(X, 0))
    qa[0] = eta3 * SN[0]
    netsize = np.zeros(np.size(X, 0))
    qe = 0
    e_qklms = np.zeros(np.size(X, 0))

    for i in range(1, np.size(X, 0)):
        x_tmp = X[i, :].reshape([M, 1])
        #Compute the output
        for j in range(i):
            x_pre = qc[j, :].reshape([M, 1])
            qpred[i] += qa[j] * np.exp(-(x_tmp-x_pre).T.dot((x_tmp-x_pre))/sigma)
        #compute the error
        e_tmp = SN[i-D] - qpred[i] 
        #compute the distance between u(i) and c(i-1)
        dis = np.zeros([i, 1])
        for j in range(i):
            x_pre = X[j, :].reshape([M, 1])
            dis[j] = (x_tmp-x_pre).T.dot(x_tmp-x_pre)
        dis_min = np.min(dis)
        seat = get_postion(dis, dis_min)
        
        if (dis_min <= q_size):
            qc[i, :] = qc[seat, :]
            qa[i] = eta3*e_tmp + qa[i-1]
            netsize[i] = netsize[i-1]
        else:
            qc[i, :] = X[i, :]
            qa[i] = eta3*e_tmp
            netsize[i] = netsize[i-1] + 1
            
        qe += e_tmp**2
        e_qklms[i] = qe/(i+1)
        
        print(i)
        
    return e_qklms, netsize
#%%
MSE, NetSize = QKLMS(X_In, 0.8, 40, 0)
#%%
#plt.plot(MSE)
plt.plot(MSE)