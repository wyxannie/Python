# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 08:34:05 2020

@author: Yingxin Wang
"""

############## Computational Methods in Finance ##################
##############           Homework 2             ##################

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

########################################### Question 1 ###########################################

def Q_1(a, N=1):
    
    np.random.seed(123)
    CholDecom = np.linalg.cholesky(np.array([[3, a], [a, 5]]))
    n = 1000
    Z = np.random.normal(size=(2, N*n))
    Mu = np.zeros(Z.shape)
    BiNormal = Mu + np.dot(CholDecom, Z)
    # print(BiNormal)
    # SampleMonment = np.cov(BiNormal[0, ], BiNormal[1, ], ddof = 1)
    # p = SampleMonment[0, 1] / (np.sqrt(SampleMonment[0, 0]) * np.sqrt(SampleMonment[1, 1]))
    
    p_0 = [np.corrcoef(BiNormal[0, i:(i+1)*n], BiNormal[1, i:(i+1)*n])[0, 1] for i in range(0, N)]
    # print(p_0)
    
    return np.mean(p_0)

p = Q_1(a = -0.7, N=10000)
(-0.7)/(np.sqrt(3) * np.sqrt(5))


########################################### Question 2 ###########################################

def Q_2(p, N=1):
    
    # np.random.seed(0)
    
    # N=10
    # p = 0.6
    CholDecom = np.linalg.cholesky(np.array([[1, p], [p, 1]]))
    Z = np.random.normal(size=(2, N))
    Mu = np.zeros(shape=(2, N))
    [X, Y] = Mu + np.dot(CholDecom, Z)

    E = np.max(np.vstack((np.zeros(shape=(N, )), (X**3 + np.sin(Y) + (X**2) * Y))), axis = 0)
    print(E.shape)
    return np.mean(E)

Q_2(p=0.6, N=1000)


########################################### Question 3 ###########################################

def Q_3(t, N = 10000, VarianceReduction = False):
    np.random.seed(0)
    # W_t = np.sqrt(t) * np.random.randn(N)
    W_t = np.cumsum(np.random.randn(t, N), axis=0)[-1, ] # dt = 1
    
    if VarianceReduction:
        A_t = 1/2 * (W_t**2 + np.sin(W_t) + (-W_t)**2 + np.sin(-W_t))
        B_t = 1/2 * (np.exp(t/2) * np.cos(W_t) + np.exp(t/2) * np.cos(-W_t))
    else:
        A_t = W_t**2 + np.sin(W_t)
        B_t = np.exp(t/2) * np.cos(W_t)
        
    if t == 5:       
        return np.mean(A_t), np.mean(B_t), np.var(A_t), np.var(B_t)
    else:
        return np.mean(A_t), np.mean(B_t)

A_1, B_1 = Q_3(t=1)
A_3, B_3 = Q_3(t=3)
A_5, B_5, A_t_var, B_t_var = Q_3(t=5)
print(A_1, A_3, A_5)
print(B_1, B_3, B_5) # B is martingale ?

A_5, B_5, A_t_var_new, B_t_var_new = Q_3(t=5, VarianceReduction = True)
print([A_t_var, A_t_var_new], [B_t_var, B_t_var_new])
# no improvement, since the B_t is an even function of W_t


########################################### Question 4 ###########################################

def StockSim(T, S_0, r, sigma, N, VarianceReduction = False):
#### simulate the whole path? or just the final stock price? ########################33###########3    
    
    np.random.seed(0)
    
    # Z_t = np.random.randn(100, N) # dt = T/100
    
    # Stock = np.zeros(Z_t.shape)
    # Stock[0, ]= S_0
    # Stock_1 = Stock.copy()
    
    # for i in range(1, 100):
    #     Stock[i, ] = Stock[i-1, ] + r*Stock[i-1, ]*T/100 + sigma*Stock[i-1, ] * Z_t[i-1, ]*np.sqrt(T/100)
    
    # if VarianceReduction:
    #     for i in range(1, 100):
    #         Stock_1[i, ] = Stock_1[i-1, ] + r*Stock_1[i-1, ]*T/100 + sigma*Stock_1[i-1, ] * (-Z_t[i-1, ])*np.sqrt(T/100)    
    
    #     return Stock, Stock_1
    # else:
    #     return Stock
    
    # T=5
    # S_0 = 88
    # r=0.04
    # sigma=0.2
    # N=10
    
    Z_t = np.random.randn(1, N)
    Stock = S_0 * np.exp((r - (sigma**2)/2) * T + sigma * np.sqrt(T) * Z_t)
    if VarianceReduction:
        Stock_1 = S_0 * np.exp((r - (sigma**2)/2) * T + sigma * np.sqrt(T) * (-Z_t))
        return Stock, Stock_1
    else:
        return Stock


def EuropeanCall(X, T, S_0, r, sigma, N, VarianceReduction = False):
    
    if VarianceReduction:
        StockPrice_plus,  StockPrice_minus = StockSim(T, S_0, r, sigma, N, VarianceReduction)
        # print(StockPrice_plus[-1, ].shape)
        Payoff_plus = np.max(np.vstack((np.zeros(StockPrice_plus[-1, ].shape), (StockPrice_plus[-1, ] - X))), axis = 0)
        Payoff_minus = np.max(np.vstack((np.zeros(StockPrice_minus[-1, ].shape), (StockPrice_minus[-1, ] - X))), axis = 0)
        
        Payoff = 1/2 * (Payoff_plus + Payoff_minus)
        price = 1/2 * (np.exp(-r*T) * np.mean(Payoff_plus) + np.exp(-r*T) * np.mean(Payoff_minus))
    
    else:
        StockPrice = StockSim(T, S_0, r, sigma, N)
        # print(StockPrice[-1, ].shape)
        Payoff = np.max(np.vstack((np.zeros(StockPrice[-1, ].shape), (StockPrice[-1, ] - X))), axis = 0)
        price = np.exp(-r*T) * np.mean(Payoff)
        
    return price, ((np.exp(-r*T))**2) * np.var(Payoff)


def BSOptionPrice(S_0, K, T, r, sigma):
    
    d1 = (np.log(S_0/K) + (r + (1/2)*(sigma**2))*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    CallPrice = S_0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    return CallPrice

c_sim, c_sim_var = EuropeanCall(X=100, T=5, S_0 = 88, r=0.04, sigma=0.2, N=100000)
c_bs = BSOptionPrice(S_0=88, K=100, T=5, r=0.04, sigma=0.2)
c_sim_1,  c_sim_var_1= EuropeanCall(X=100, T=5, S_0 = 88, r=0.04, sigma=0.2, N=100000, VarianceReduction=True)


########################################### Question 5 ###########################################

#### a

ESn = []
for i in range(10):
    S = StockSim(T=i+1, S_0=88, r=0.04, sigma=0.18, N=1000)
    ESn.append(np.mean(S))
    
plt.figure(), plt.plot(range(1, 11), ESn)



#### b
S_0 = 88
Z_t = np.random.randn(1000, 6) # dt = T/1000

Stock = np.zeros(Z_t.shape)
Stock[0, ], dt = S_0, 10/1000

for i in range(1, 1000):
    Stock[i, ] = Stock[i-1, ] + 0.04*Stock[i-1, ]*dt + 0.18*Stock[i-1, ] * Z_t[i-1, ]*np.sqrt(dt)


ESn_1 = Stock[np.array(range(1, 11))*100 - 1,] 
plt.figure() 
plt.plot(range(1, 11), ESn)
plt.plot(range(1, 11), ESn_1)
###################### put them together? pick up corresponding point and then combine and redraw?

#### c
ESn_new = []
for i in range(10):
    S_new = StockSim(T=i+1, S_0=88, r=0.04, sigma=0.35, N=1000)
    ESn_new.append(np.mean(S_new))
    
plt.figure(), plt.plot(range(1, 11), ESn_new)

Z_t = np.random.randn(1000, 6) # dt = T/1000

Stock_new = np.zeros(Z_t.shape)
Stock_new[0, ], dt = 88, 10/1000

for i in range(1, 1000):
    Stock_new[i, ] = Stock_new[i-1, ] + 0.04*Stock_new[i-1, ]*dt + 0.35*Stock_new[i-1, ] * Z_t[i-1, ]*np.sqrt(dt)
 
plt.plot(Stock_new)


########################################### Question 6 ###########################################

#### a
# using Euler's discretization
Ia = 4 * sum([np.sqrt(1 - (i/1000)**2) * (1/1000) for i in range(1, 1001)])


#### b
# monte-carlo simulation
n = 10000
x = np.random.uniform(0, 1, n)
Ib = 4 * np.mean(np.sqrt(1 - x**2))

np.var(np.sqrt(1 - x**2))

#### c
a = 0.74
t_x = (1 - a * (x**2)) / (1 - a/3)
# x is U[0,1]
Ic = 4 * np.mean(np.sqrt(1 - x**2) /t_x)

np.var(np.sqrt(1 - x**2) /t_x)
