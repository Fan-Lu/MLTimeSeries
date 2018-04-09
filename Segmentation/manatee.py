# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:10:54 2018

@author: fanlu
"""
#%%
import sounddevice as sd
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

#%%
#   load data
#fs_train, train = wavfile.read('D:/MyGitHub/MLTimeSeries/Segmentation/data/train_signal.wav')
#fs_test, test = wavfile.read('D:/MyGitHub/MLTimeSeries/Segmentation/data/test_signal.wav')
#fs_noise, noise = wavfile.read('D:/MyGitHub/MLTimeSeries/Segmentation/data/noise_signal.wav')

train = np.loadtxt('D:/MyGitHub/MLTimeSeries/Segmentation/data/train.txt')
test = np.loadtxt('D:/MyGitHub/MLTimeSeries/Segmentation/data/test.txt')
noise = np.loadtxt('D:/MyGitHub/MLTimeSeries/Segmentation/data/noise.txt')

train_fs = 48000  # sample rate for train
fs = 44100 # sample rate for test and noise

#%%
### Data Prune ####
train_pru = []
train_pru.append(train[96000:109158])
#train_pru.append(train[205158:218994])
#train_pru.append(train[314994:328155])
#train_pru.append(train[424155:441225])
#train_pru.append(train[537225:551764])
#train_pru.append(train[647764:661664])
#train_pru.append(train[757664:771693])
#train_pru.append(train[867693:888366])
#train_pru.append(train[984366:1000043])
#train_pru.append(train[1096043:1107446])

train_pru = np.hstack(train_pru)

#start = 0
#end = 0
#
#for i in range(1107446, len(train)):
#    if np.abs(train[i]) > 5e-5:
#        start = i
#        break
#for j in range(start, len(train)):
#    if np.abs(train[j]) < 5e-5 and np.abs(train[j+4]) < 5e-5:
#        end = j
#        break
#print(start)
#print(end)

#%%
#plt.plot(noise)
sd.play(test, fs)

#%%
plt.figure('Train_Signal')
plt.plot(train)
plt.figure('Test_Signal')
plt.plot(test)
plt.figure('Noise_Signal')
plt.plot(noise)

#%%
#   DATA Preparation
def DataPre(xn, m):
    X = np.zeros([len(xn), m])
    for j in range(m):
        for i in range(len(xn)-m+j+1):
            X[i+m-j-1][j] = xn[i]
    return X

#%%
M = 3   #Model Order
pro_train = DataPre(train_pru, M)
pro_test = DataPre(test, M)
pro_noise = DataPre(noise, M)

#%%
#   KLMS Train with mse
def KLMS(org, des, lr, ks):
    #Initialization
    pred = np.zeros([len(des)])
    a = np.zeros([len(des)])
    pred[0] = des[0]
    a[0] = lr*des[0]
    mse = np.zeros([len(des)])
    e = 0
    
    #start learning
    for i in range(1, len(des)):
        x_tmp = org[i, :].reshape([M, 1])
        #Compute the output
#        x_pre = org[:i, :].reshape([M, i])
#        coe = a[:i].reshape([i, 1])
#        dif =  (x_pre - x_tmp).T.dot(x_pre - x_tmp)
#        pred[i] = np.trace(np.multiply(dif, coe))
        for j in range(i):
            x_pre = org[j, :].reshape([M, 1])
            pred[i] += a[j] * np.exp(-ks*(x_tmp-x_pre).T.dot((x_tmp-x_pre)))
        #Compute the error
        e_tmp = des[i] - pred[i]
        e += e_tmp**2
        mse[i] = e/(i+1)
        a[i] = lr*e_tmp
    
        print('MSE ter = ', i, 'mse = ', mse[i])
    return mse, pred, a


#%%
#   Model Training
MSE, Pred_Train, Coe = KLMS(pro_train, train, 0.1, 0.01)
np.savetxt('MSE.csv', MSE, delimiter=',')
np.savetxt('Pred_Train.csv', MSE, delimiter=',')
np.savetxt('Coe.csv', MSE, delimiter=',')


