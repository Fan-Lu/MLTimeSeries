# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:56:06 2018

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
yn = np.random.normal(0, 1, 5000).reshape(5000 ,1) #input
tn = np.zeros([5000, 1])
qn = np.zeros([5000, 1])

tn[0] = -0.8*yn[0]
for i in range(1 , len(yn)):
    tn[i] = -0.8*yn[i] + 0.7*yn[i-1]    
    
for i in range(len(yn)):
    qn[i] = tn[i] + 0.25*tn[i]**2 + 0.11*tn[i]**3
    
n = awgn(qn, 15).reshape(5000, 1)  #noise
xn = qn + n  #Input of Adaptive filter

M = 5  #Filter Order
D = 2  #Delay

X = np.zeros([len(xn), M])
for j in range(M):
    for i in range(len(xn)-M+j+1):
        X[i+M-j-1][j] = xn[i]

#%%
plt.plot(xn)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Output signal')
#%%
#LMS
w = np.zeros([M, 1])
e_lms = np.zeros([5000, 1])
eta1 = 0.005 #learning rate
w_save = np.zeros([5000, M])
pred_save = np.zeros([5000, 1])
error = 0

start_LMS = time.clock()
for i in range(5000):
    x_tmp = X[i, :].reshape([M, 1])
    y_tmp = x_tmp.T.dot(w)
    e_tmp = yn[i-D] - y_tmp
    error += e_tmp**2
    e_lms[i] = error/(i+1)
    w = w + eta1*x_tmp*e_tmp
    w_save[i, :] = w.reshape(M)
end_LMS = time.clock()
time_LMS = end_LMS - start_LMS
    
#%%
#Plot
#p1 = plt.plot(w_save)
##plt.legend(['Learning Rate = 0.001', 'Learning Rate = 0.005', 'Learning Rate = 0.01'])
#plt.xlabel('Iteration')
#plt.ylabel('Weghts')
#plt.title('Weight track')
#%%
#KLMS
#Initialization
eta2 = 0.6
h = 0.04  #kernel size 0.06
pred = np.zeros([len(xn)])
a = np.zeros([len(xn)])
pred[0] = yn[-2]
a[0] = eta2*yn[0]
e_klms = np.zeros([5000, 1])
e = 0

#kernel function
def kernel(x_tmp, cur):
    f_tmp = 0
    for j in range(cur):
        x_pre = X[j, :].reshape([M, 1])
        f_tmp +=  a[j] * np.exp(-h*(x_tmp-x_pre).T.dot((x_tmp-x_pre)))
    return f_tmp

start_KLMS = time.clock()
for i in range(1, len(xn)):
    x_tmp = X[i, :].reshape([M, 1])
    #Compute the output
    for j in range(i):
        x_pre = X[j, :].reshape([M, 1])
        pred[i] += a[j] * np.exp(-h*(x_tmp-x_pre).T.dot((x_tmp-x_pre)))
    #pred[i] = cernel(x_tmp, i)
    #Compute the error
    e_tmp = yn[i-D] - pred[i]
    e += e_tmp**2
    e_klms[i] = e/(i+1)
    a[i] = eta2*e_tmp
    print(i)
end_KLMS = time.clock()
time_KLMS = end_KLMS - start_KLMS

#p1 = plt.plot(e_klms, 'b')
##plt.legend(['Kernel size = 0.06', 'Kernel size = 0.04', 'Kernel size = 0.02', 'Kernel size = 0.005'])
#plt.xlabel('Iteration')
#plt.ylabel('Mean Square Error')
#plt.title('Learning Curve')
#%%
#QKLMS
#Initializetion
netsize = np.zeros(len(xn))
eta3 = 0.6 #step size
sigma = 0.02 #kernel size
q_size = 0.2 #quantization size 0.1
qc = np.zeros([len(xn), M])
qc[0, :] = X[0, :].reshape([M, 1]).reshape(5)
qa = np.zeros(len(xn))
qa[0] = eta3 * yn[0]
qpred = np.zeros(len(xn))
qe = 0
e_qklms = np.zeros(len(xn))

def get_postion(array, dismin):
    for i in range(len(array)):
        if (array[i] == dismin):
            return i

start_QKLMS = time.clock()
for i in range(1, len(xn)):
    x_tmp = X[i, :].reshape([M, 1])
    #Compute the output
    for j in range(i):
        x_pre = qc[j, :].reshape([M, 1])
        qpred[i] += qa[j] * np.exp(-sigma*(x_tmp-x_pre).T.dot((x_tmp-x_pre)))
    #compute the error
    e_tmp = yn[i-D] - qpred[i] 
    #compute the distance between u(i) and c(i-1)
    dis = np.zeros([i, 1])
    for j in range(i):
        x_pre = X[j, :].reshape([M, 1])
        dis[j] = (x_tmp-x_pre).T.dot(x_tmp-x_pre)
    dis_min = np.min(dis)
    seat = get_postion(dis, dis_min)
    
    if (dis_min <= q_size):
        qc[i, :] = qc[i-1, :]
        qa[i] = eta3*e_tmp + qa[i-1]
        netsize[i] = netsize[i-1]
    else:
        qc[i, :] = X[i, :]
        qa[i] = eta3*e_tmp
        netsize[i] = netsize[i-1] + 1
        
    qe += e_tmp**2
    e_qklms[i] = qe/(i+1)
 
    print(i)
end_QKLMS = time.clock()
time_QKLMS = end_QKLMS - start_QKLMS

#p1 = plt.plot(e_qklms, 'r')
##plt.legend(['Kernel size = 0.06', 'Kernel size = 0.04', 'Kernel size = 0.02', 'Kernel size = 0.005'])
#plt.legend(['quantization size = 0.4', 'quantization size = 0.3', 'quantization size = 0.2', 'quantization size = 0.1', 'quantization size = 0.05'])
#plt.xlabel('Iteration')
#plt.ylabel('Mean Square Error')
#plt.title('Learning Curve')
#%%
plt.figure()
p1 = plt.plot(e_lms, 'r')
p2 = plt.plot(e_klms, 'g')
p3 = plt.plot(e_qklms, 'y')
plt.legend((p1[0], p2[0], p3[0]), ('LMS', 'KLMS', 'QKLMS'))
plt.xlabel("Iteration")
plt.ylabel("Mean Square Error")
plt.title("Learning Curve")
#plt.plot(pred)

#%%
y = np.arange(5000)
p1 = plt.plot(y, netsize, 'r')
p2 = plt.plot(y, y, 'b')
plt.legend((p1[0], p2[0]), ('KLMS', 'QKLMS'))
plt.xlabel("Iteration")
plt.ylabel("Network size")