# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:50:58 2018

@author: fanlu
"""
#%%
#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import time

#%%
#Function
def awgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

#%%
#Prepare data
yn = np.random.normal(0, 1.0, 5000).reshape(5000 ,1) #input
tn = np.zeros([5000, 1])
qn = np.zeros([5000, 1])

tn[0] = -0.8*yn[0]
for i in range(1 , len(yn)):
    tn[i] = -0.8*yn[i] + 0.7*yn[i-1]    
    
for i in range(len(yn)):
    qn[i] = tn[i] + 0.25*tn[i]**2 + 0.11*tn[i]**3
    
noise = awgn(qn, 15).reshape(5000, 1)  #noise
xn = qn + noise  #Input of Adaptive filter

M = 5  #Filter Order
D = 2  #Delay

X = np.zeros([len(xn), M])
for j in range(M):
    for i in range(len(xn)-M+j+1):
        X[i+M-j-1][j] = xn[i]
        
        
#%%
#KLMS Train with mse
def KMSE(org, des, lr, ks):
    #Initialization
    pred = np.zeros([len(des)])
    a = np.zeros([len(des)])
    pred[0] = yn[-2]
    a[0] = lr*yn[0]
    mse = np.zeros([5000, 1])
    e = 0
    erle = np.zeros([len(des)])
    
    #start learning
    start_KLMS = time.clock()
    for i in range(1, len(xn)):
        x_tmp = org[i, :].reshape([M, 1])
        #Compute the output
        for j in range(i):
            x_pre = org[j, :].reshape([M, 1])
            pred[i] += a[j] * np.exp(-ks*(x_tmp-x_pre).T.dot((x_tmp-x_pre)))
        #Compute the error
        e_tmp = des[i-D] - pred[i]
        e += e_tmp**2
        mse[i] = e/(i+1)
        a[i] = lr*e_tmp
        
        erle[i] = 10*np.log(des[i]**2/(e_tmp**2))
        
        print('MSE ter = ', i, 'mse = ', mse[i])
        
    end_KLMS = time.clock()
    time_KLMS = end_KLMS - start_KLMS
    
    return mse, pred, erle, time_KLMS

#%%
#KMLS trained with MCC
def KMCC(org, des, lr, ks):
    #Initialization
    pred = np.zeros([len(des)])
    mse = np.zeros([5000, 1])
    error = np.zeros([5000, 1])
    e = 0
    error[0] = yn[-2]
    pred[0] = lr*error[0]*np.exp(-error[0]**2*ks)
    erle = np.zeros([len(des)])
    
    #start learning
    start_KLMS = time.clock()
    for i in range(1, len(xn)):
        x_tmp = org[i, :].reshape([M, 1])
        #Compute the output
        for j in range(i):
            x_pre = org[j, :].reshape([M, 1])
            pred[i] += lr*np.exp(-error[j]**2*ks)*error[j]*np.exp(-ks*(x_tmp-x_pre).T.dot(x_tmp-x_pre))
        #Compute the error
        error[i] = des[i-D] - pred[i]
        e += error[i]**2
        mse[i] = e/(i+1)
        
        erle[i] = 10*np.log(des[i]**2/(error[i]**2))
        
        print('MCC ter = ', i, 'mse = ', mse[i])
        
    end_KLMS = time.clock()
    time_KLMS = end_KLMS - start_KLMS
    
    return mse, pred, erle, time_KLMS  

#%%
def KMEE(org, des, lr, ks, kc): #L: error samples length
    #Initialization
    pred = np.zeros([len(des)])
    a = np.zeros([len(des)])
    pred[0] = yn[-2]
    a[0] = lr*yn[0]
    mse = np.zeros([5000, 1])
    e = 0
    erle = np.zeros([len(des)])
    
    #start learning
    start_KLMS = time.clock()
    for i in range(1, len(xn)):
        x_tmp = org[i, :].reshape([M, 1])
        #Compute the output
        for j in range(i):
            x_pre = org[j, :].reshape([M, 1])
            pred[i] += a[j] * np.exp(-ks*(x_tmp-x_pre).T.dot((x_tmp-x_pre)))
        #Compute the error
        e_old = des[i-D-1] - pred[i-1] - np.mean(des)
        e_tmp = des[i-D] - pred[i] - np.mean(des)
        a[i] = lr*np.exp(-kc*(e_tmp-e_old)**2)*(e_tmp-e_old)*(2*kc)
        
        e += e_tmp**2
        mse[i] = e/(i+1)
        
        erle[i] = 10*np.log(des[i]**2/(e_tmp**2))

        print('MEE ter = ', i, 'mse = ', mse[i])
        
    end_KLMS = time.clock()
    time_KLMS = end_KLMS - start_KLMS
            
    return mse, pred, erle, time_KLMS
#%%
if __name__ == "__main__":
    mse_MSE, pre_MSE, ERLE_MSE, t_kmse_MSE = KMSE(X, yn, .8, .01)  #0.01 best
    mse_MCC, pre_MCC, ERLE_MCC, t_kmse_MCC = KMCC(X, yn, .8, .02)   #0.01 best
    mse_MEE, pre_MEE, ERLE_MEE, t_kmse_MEE = KMEE(X, yn, 0.8, .1, .3)
#%%
#p1 = plt.plot(mse001, 'r')
#p2 = plt.plot(mse005, 'g')
#p3 = plt.plot(mse01, 'b')
#p4 = plt.plot(mse05, 'y')
#plt.legend((p1[0], p2[0], p3[0], p4[0]), ('kernel size = 0.01', 'kernel size = 0.05', 'kernel size = 0.1', 'kernel size = 0.5'))
#plt.xlabel('iteration')
#plt.ylabel('MSE')
##%%
#p1 = plt.plot(mse_MSE, 'r')
#p2 = plt.plot(mse_MCC, 'y')
#p3 = plt.plot(mse_MEE, 'b')
#plt.legend((p1[0], p2[0], p3[0]), ('KMEE', 'KMCC', 'KLMS'))
#plt.xlabel('iteration')
#plt.ylabel('MSE')
#
#plt.plot(mse3, 'r') #r g b
#
##%%
erle_MSE = np.zeros([50])
erle_MCC = np.zeros([50])
erle_MEE = np.zeros([50])
for i in range(50):
    erle_MSE[i] = np.sum(ERLE_MSE[100*(i):100*(i+1)])/100 
 
for i in range(50):
    erle_MCC[i] = np.sum(ERLE_MCC[100*(i):100*(i+1)])/100 

for i in range(50):
    erle_MEE[i] = np.sum(ERLE_MEE[100*(i):100*(i+1)])/100     

#%%
p1 = plt.plot(erle_MSE, 'r')
p2 = plt.plot(erle_MCC, 'y')
p3 = plt.plot(erle_MEE, 'b')
plt.legend((p1[0], p2[0], p3[0]), ('KMEE', 'KMCC', 'KLMS'))
plt.xlabel('iteration')
plt.ylabel('ERLE (dB)')