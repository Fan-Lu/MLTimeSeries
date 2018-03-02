# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:49:22 2018

@author: fanlu
"""

#%%
#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import time

#%%
def loadCSVfile(path):
    tmp = np.loadtxt(path, dtype = np.str, delimiter = ",")
    data = tmp[1:, 3].astype(np.float)
    return data

#%%
SN = loadCSVfile("D:\MyGitHub\MLTimeSeries\SunpotPrediction\SN_m_tot_V2.0.csv")
#D = SN.reshape(np.size(SN), 1)
#plt.plot(D)
SN = (SN-np.sum(SN))/np.sum(SN)
M = 6  #Filter Order， Taken's Theory
D = 10  #delay 10 or 20
N_tr = 3000
N_te = 228
X_tr = np.zeros([N_tr, M])
X_te = np.zeros([N_te, M])
D_tr = np.zeros([N_tr, 1])
D_te = np.zeros([N_te, 1])

D_tr = SN[0:N_tr]
D_te = SN[N_tr:(N_tr+N_te)]

for j in range(M):
    for i in range(N_tr-M+j+1):
        X_tr[i+M-j-1][j] = SN[i]
        
for j in range(M):
    for i in range(N_te-M+j+1):
        X_te[i+M-j-1][j] = SN[i]        

#%%
def get_postion(array, dismin):
    for i in range(len(array)):
        if (array[i] == dismin):
            return i

#%%
#######LMS######
def LMS(x_tr, d_tr, x_te, d_te, lr_k, m):
    w = np.zeros([m, 1])
    w_track = np.zeros([np.size(x_tr, 0), m])
    mse_tr = np.zeros(np.size(x_tr, 0))
    mse_te = np.zeros(np.size(x_te, 0))
    error_tr = 0
    error_te = 0
    pred_tr = np.zeros(np.size(x_tr, 0))
    pred_te = np.zeros(np.size(x_te, 0))
    
    ####Training####
    for i in range(np.size(x_tr, 0)):
        x_tmp = x_tr[i, :].reshape([M, 1])
        y_tmp = x_tmp.T.dot(w)
        pred_tr[i] = y_tmp
        e_tmp = d_tr[i] - y_tmp
        error_tr += e_tmp**2
        mse_tr[i] = error_tr/(i+1)
        w_track[i, :] = w.reshape(m)
        w = w + lr_k*x_tmp*e_tmp
        
        print('LMS Train Iter = ', i, 'MSE = ', mse_tr[i])

    for i in range(np.size(x_te, 0)):
        x_tmp = x_te[i, :].reshape([M, 1])
        pred_te[i] = x_tmp.T.dot(w)
        e_tmp = d_te[i] - pred_te[i]
        error_te += e_tmp**2
        mse_te[i] = error_te/(i+1)
        
        
        print('LMS Test Inter = ', i, 'MSE = ', mse_te[i])
    return mse_tr, mse_te, pred_tr, pred_te
#%%
####QKLMS Trianed with MSE        
def MSE(x_tr, x_te, lr_k, sigma, q_size):
    qpred_tr = np.zeros(np.size(x_tr, 0))
    qpred_te = np.zeros(np.size(x_te, 0))
    qc = np.zeros([np.size(x_tr, 0), M])
    qc[0, :] = x_tr[0, :].reshape([M, 1]).reshape(M)
    qa = np.zeros(np.size(x_tr, 0))
    qa[0] = lr_k * SN[0]
    netsize = np.zeros(np.size(x_tr, 0))
    qe_tr = 0
    qe_te = 0
    mse_tr = np.zeros(np.size(x_tr, 0))
    mse_te = np.zeros(np.size(x_te, 0))

    #######Training#######
    for i in range(1, np.size(x_tr, 0)-D):
        x_tmp = x_tr[i, :].reshape([M, 1])
        #Compute the output
        for j in range(i):
            x_pre = qc[j, :].reshape([M, 1])
            qpred_tr[i] += qa[j] * np.exp(-(x_tmp-x_pre).T.dot((x_tmp-x_pre))*sigma)
        #compute the error
        e_tmp = D_tr[i+D] - qpred_tr[i] 
        #compute the distance between u(i) and c(i-1)
        dis = np.zeros([i, 1])
        for j in range(i):
            x_pre = x_tr[j, :].reshape([M, 1])
            dis[j] = (x_tmp-x_pre).T.dot(x_tmp-x_pre)
        dis_min = np.min(dis)
        seat = get_postion(dis, dis_min)
        
        if (dis_min <= q_size):
            qc[i, :] = qc[seat, :]   #use old center
            qa[i] = lr_k*e_tmp + qa[i]  
            netsize[i] = netsize[i-1]
        else:
            qc[i, :] = x_tr[i, :]   #update center
            qa[i] = lr_k*e_tmp
            netsize[i] = netsize[i-1] + 1
            
        qe_tr += e_tmp**2
        mse_tr[i] = qe_tr/(i+1)
        
        print('Trian Inter =', i, 'MSE =', mse_tr[i])
    ######End of trian#######
        
    #Testing
    for i in range(np.size(x_te, 0)-D):
        x_tmp = x_te[i, :].reshape([M, 1])
        for j in range(np.size(x_tr, 0)):
            x_pre = qc[j, :].reshape([M, 1])
            qpred_te[i] += qa[j] * np.exp(-(x_tmp-x_pre).T.dot((x_tmp-x_pre))*sigma)
        e_tmp = D_te[i+D] - qpred_te[i]
        qe_te += e_tmp**2
        mse_te[i] = qe_te/(i+1)
        
        print('Test Inter =', i, 'MSE =', mse_te[i])
    #End of test

    return mse_tr, mse_te, netsize, qpred_tr, qpred_te

#%%QKLMS trained with MCC
def MCC(x_tr, x_te, lr_q, k, q_size):
    qc = np.zeros([np.size(x_tr, 0), M])
    qc[0, :] = x_tr[0, :].reshape([M, 1]).reshape(M)    
    qe_tr = np.zeros(np.size(x_tr, 0))
    qe_tr[0] = D_tr[0]
    qpred_tr = np.zeros(np.size(x_tr, 0))
    qpred_te = np.zeros(np.size(x_te, 0))
    qpred_tr[0] = lr_q*qe_tr[0]*np.exp(-qe_tr[0]**2/(2*k**2))
    mse_tr = np.zeros(np.size(x_tr, 0))
    mse_te = np.zeros(np.size(x_te, 0))
    e_tmp_tr = qe_tr[0]
    e_tmp_te = 0
    
    ###Start Trianing
    for i in range(1, np.size(x_tr, 0)):
        x_tmp = x_tr[i, :].reshape([M, 1])
        for j in range(i):
            x_pre = qc[j, :].reshape([M, 1])
            qpred_tr[i] += lr_q*np.exp(-qe_tr[j]**2/(2*k**2))*qe_tr[j]*np.exp(-(x_tmp-x_pre).T.dot((x_tmp-x_pre))/(2*k**2))
        qe_tr[i] = D_tr[i] - qpred_tr[i]
        
        #compute the distance between u(i) and c(i-1)
        dis = np.zeros([i, 1])
        for j in range(i):
            x_pre = x_tr[j, :].reshape([M, 1])
            dis[j] = (x_tmp-x_pre).T.dot(x_tmp-x_pre)
        dis_min = np.min(dis)
        seat = get_postion(dis, dis_min)
        if (dis_min <= q_size):
            qc[i, :] = qc[seat, :]   #use old center
        else:
            qc[i, :] = x_tr[i, :]   #update center 
            
        e_tmp_tr += qe_tr[i]**2
        mse_tr[i] = e_tmp_tr/(i+1)
        
        print('Trian Inter =', i, 'MSE =', mse_tr[i])
        
    ###Start Trianing
    for i in range(1, np.size(x_te, 0)):
        x_tmp = x_te[i, :].reshape([M, 1])
        for j in range(np.size(x_tr, 0)):
            x_pre = qc[j, :].reshape([M, 1])
            qpred_te[i] += lr_q*np.exp(-qe_tr[j]**2/(2*k**2))*qe_tr[j]*np.exp(-(x_tmp-x_pre).T.dot((x_tmp-x_pre))/(2*k**2))  
        e_tmp_te += (D_te[i] - qpred_te[i])**2
        mse_te[i] = e_tmp_te/(i+1)
        
        print('Test Inter =', i, 'MSE =', mse_te[i])
        
    return mse_tr, qpred_tr, mse_te, qpred_te      
    ###End Trianing###
    
#%%
if __name__ == "__main__":
    #MSE_Tr, MSE_Te, NetSize, QPred_Tr,  QPred_Te = MSE(X_tr, X_te, 0.9, 0.0000001, 20)
    #MSE_Tr, QPred_Tr, MSE_Te, QPred_Te = MCC(X_tr, X_te, 0.75, 100000, 25)
    MSE_Tr, MSE_te, Pred_Tr, Pred_Te = LMS(X_tr, D_tr, X_te, D_te, 0.1, M )
#%%
#plt.plot(QPred_Te)
#plt.plot(D_te)
plt.plot(MSE_Tr, 'r')
#plt.plot(D_te, 'b')