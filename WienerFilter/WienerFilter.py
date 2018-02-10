# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:33:54 2018

@author: fanlu
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
#Prepare Data
alpha = 1.5
t = np.random.randn(10000)
alpha_stable_noise = np.exp(-np.abs(t)**alpha)
white_noise = np.random.normal(0, 0.1, 10000)
output = np.random.rand(10000)
plt.plot(t,alpha_stable_noise)


