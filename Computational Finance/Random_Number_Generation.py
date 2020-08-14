# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:23:49 2020

@author: Yingxin Wang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
import time


#### 1

# a
def LGM_Random(X0, N):
    '''
    LGM method: X_n+1 = (7^5 * X_n) modulo (2^31-1)
    '''
    Xn = [X0]
    for i in range(0, int(N-1)):
      Xn.append((7**5 * Xn[-1]) % (2**31 - 1)) 
    Un = np.transpose(np.array(np.mat(Xn))) / (2**31 - 1)
    
    return Un

Un_1 = LGM_Random(time.time(), 10000)

plt.figure(), plt.hist(Un_1, bins=10)
plt.xlabel("Value"), plt.ylabel("Frequency"), plt.title("Uniform Distribution [0,1]")
Un_mean_1 = np.mean(Un_1)
Un_stdDev_1 = np.std(Un_1)


# b
# Using build-in function
Un_2 = np.random.uniform(0, 1, 10000)
Un_mean_2 = np.mean(Un_2)
Un_stdDev_2 = np.std(Un_2)
# Theoretical value
# print(1/2*(1-0))
# print(np.sqrt(1/12))


#### 2
# a
def Transform(u, Probs=[0.3, 0.35, 0.2, 0.15]):
   
    CumProbs = np.cumsum(Probs)
    
    if u < CumProbs[0]:
        x = -1
    elif u >= CumProbs[0] and u < CumProbs[1]:
        x = 0
    elif u >= CumProbs[1] and u < CumProbs[2]:
        x = 1
    else:
        x = 2
        
    return x

Xn_2 = np.array(list(map(Transform, Un_1)))


# b
plt.figure()
plt.bar(np.unique(Xn_2), pd.value_counts(Xn_2).sort_index())
plt.xlabel("Value"), plt.ylabel("Frequency"), plt.title("Distribution in Question 2")
Xn_2_mean = np.mean(Xn_2)
Xn_2_stdDev = np.std(Xn_2)


#### 3

# a
Xn_3 = [1 if u < 0.64 else 0 for u in LGM_Random(time.time(), 44*1000)]
Xn_3_1 = sum(np.array(Xn_3).reshape((44,-1)))


# b
plt.figure(), plt.hist(Xn_3_1)
plt.xlabel("Value"), plt.ylabel("Frequency"), plt.title("Binomial distribution with n=44 and p=0.64")
p_3_the = sum([comb(44, k) * (0.64**k) * (0.36**(44-k)) for k in range(40, 45)])
p_3_emp = sum(Xn_3_1 > 40) / 1000


#### 4

# a
Xn_4 = -1.5* np.log(LGM_Random(time.time(), 10000))

# b
p_4_1_the = 1 - (1 - np.exp(-1/1.5))
p_4_1_emp = sum(Xn_4 > 1) / 10000
p_4_2_the = 1 - (1 - np.exp(-4/1.5))
p_4_2_emp = sum(Xn_4 > 4) / 10000

# c
Xn_4_mean = np.mean(Xn_4)
Xn_4_stdDev = np.std(Xn_4)
plt.figure(), plt.hist(Xn_4, bins=50)
plt.xlabel("Value"), plt.ylabel("Frequency"), plt.title(r"Exponential Distribution with $\lambda$=1.5")

#### 5

# a
def Normrv_BM(N):
    '''
    Generate r.v. of NORMAL distribution by using Box-Muller method
    '''
    U = np.hstack((LGM_Random(time.time(), N/2), LGM_Random(time.time()+922, N/2)))
    Z1 = [np.sqrt(-2* np.log(u[0])) * np.cos(2* np.pi * u[1]) for u in U]
    Z2 = [np.sqrt(-2* np.log(u[0])) * np.sin(2* np.pi * u[1]) for u in U]
    Zn = np.array(Z1 + Z2)
    
    return Zn

Zn_1 = Normrv_BM(5000)
plt.figure(), plt.hist(Zn_1, bins=50)
plt.xlabel("Value"), plt.ylabel("Frequency")
plt.title(r"Normal Distribution with $\mu$=0, $\sigma$=1")


# b 
def Normrv_PM(N):
    '''
    Generate r.v. of NORMAL distribution by using Polar-Marsaglia method
    '''
    U1 = LGM_Random(time.time(), N)
    U2 = LGM_Random(time.time()+922, N)
    V1 = 2*U1 - 1
    V2 = 2*U2 - 1
    W = V1**2 + V2**2
    
    i, Z1, Z2 = 0, [], []
    while(len(Z1+Z2) < N):
        
        if W[i] <= 1:
            Z1.append(V1[i] * np.sqrt(-2 * np.log(W[i]) / W[i]))
            Z2.append(V2[i] * np.sqrt(-2 * np.log(W[i]) / W[i]))
            
        i += 1
        
    Zn = np.array(Z1 + Z2)
    
    return Zn        

Zn_2 = Normrv_PM(5000)
plt.figure(), plt.hist(Zn_2, bins=50)
plt.xlabel("Value"), plt.ylabel("Frequency")
plt.title(r"Normal Distribution with $\mu$=0, $\sigma$=1")

# c
start_time = time.time()
Zn_1 = Normrv_BM(5000)
print("---Execution time of Box-Muller method is %s seconds ---" % (time.time() - start_time))

start_time = time.time()
Zn_2 = Normrv_PM(5000)
print("---Execution time of Polar-Marsaglia method is %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# Zn_1 = Normrv_BM(10000)
# print("---Execution time of Box-Muller method is %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# Zn_2 = Normrv_PM(10000)
# print("---Execution time of Polar-Marsaglia method is %s seconds ---" % (time.time() - start_time))
