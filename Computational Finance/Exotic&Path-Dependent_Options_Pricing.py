# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:33:33 2020

@author: Yingxin Wang
"""


############## Computational Methods in Finance ##################
##############           Homework 6            ##################


import numpy as np
import matplotlib.pyplot as plt


########################################### Question 1 ###########################################

class Lookback_Option():
    
    def __init__(self, S0, T, r, N):
        self.S0 = S0
        self.T = T
        self.r = r
        self.dt = T / int(T*np.sqrt(N))
        self.N = N
        
        
    def StockPrice(self, sigma):
        Ndt = int(self.T/self.dt)+1
        
        np.random.seed(42)
        W = np.random.randn(Ndt, self.N)
        
        StockPrice = np.zeros(shape=(Ndt, self.N))
        StockPrice[0, :] = self.S0
        for i in range(1, Ndt): # row-by-row 
            StockPrice[i, :] = StockPrice[i-1, :] + self.r*StockPrice[i-1, :]*self.dt + sigma*StockPrice[i-1, :]*W[i, :]*np.sqrt(self.dt)  
        
        return StockPrice
    
    
    def CallOption(self, sigma, X):
        StockPrice = self.StockPrice(sigma)
        Payoff = np.maximum(StockPrice.max(axis = 0) - X, 0)
        OptionPrice = np.exp(- self.r * self.T) * np.mean(Payoff)
        
        return OptionPrice   
    
    
    def PutOption(self, sigma, X):
        StockPrice = self.StockPrice(sigma)
        Payoff = np.maximum(X - StockPrice.min(axis = 0), 0)
        OptionPrice = np.exp(- self.r * self.T) * np.mean(Payoff)
        
        return OptionPrice    
        
    
S0, X, T, r, N = 98, 100, 1, 0.03, 10000
Proj6_Q1 = Lookback_Option(S0, T, r, N)

sigmaValues = np.arange(0.12, 0.48+0.04, step=0.04)
CallOption_Q1 = [Proj6_Q1.CallOption(sigma, X) for sigma in sigmaValues]
PutOption_Q1 = [Proj6_Q1.PutOption(sigma, X) for sigma in sigmaValues]

plt.figure(), plt.plot(sigmaValues, CallOption_Q1, marker="o", markersize=4)
plt.xlabel("sigma"), plt.ylabel("Call Option Price")
plt.figure(), plt.plot(sigmaValues, PutOption_Q1, marker="o", markersize=4)
plt.xlabel("sigma"), plt.ylabel("Put Option Price")



########################################### Question 2 ###########################################

def CollateralPrice(V0, mu, sigma, gamma, lambda_1, Ndt, N, dt): 
    '''
    simulate V process
    '''
    W = np.random.randn(Ndt, N)
    n_possion = np.random.poisson(lam = lambda_1*dt, size = (Ndt, N))
    J = gamma*n_possion
                         
    V = np.zeros(shape=(Ndt+1, N))
    V[0, :] = V0
    for i in range(1, Ndt+1):
        V[i, :] = V[i-1, :] + mu*dt*V[i-1, :] + sigma*W[i-1, :]*np.sqrt(dt)*V[i-1, :] + J[i-1, :]*V[i-1, :]
    
    return V


def Loan(a, b, c, Ndt, N, dt):
    '''
    simulate loan value
    '''
    Loan = np.zeros(shape = (Ndt+1, N))
    for i in range(Ndt+1):
        Loan[i, :] = a - b * (c**(12*dt*i))
        
    return Loan


def StopTime_Q(alpha, beta, Ndt, N, dt, V, Loan):
    qt = np.zeros(shape = (Ndt+1, N))
    for i in range(Ndt+1):
        qt[i, :] = alpha + beta*(dt*i)
        
    StopTime_Q = [float("inf")]*N ## no stopping time Q -> inf
    for n in range(N):
        for i in range(Ndt+1):
            if V[i][n] <= (qt[i][n]*Loan[i][n]):
                StopTime_Q[n] = dt*i
                break
    
    return StopTime_Q            
    
    
def StopTime_S(lambda_2, Ndt, N, dt):
    Nt = np.random.poisson(lam=lambda_2*dt, size = (Ndt+1, N)) 
    StopTime_S = [float("inf")]*N ## no stopping time S -> inf
    for n in range(N):
        for i in range(Ndt+1):
            if Nt[i][n] > 0:
                StopTime_S[n] = dt*i
                break
    
    return StopTime_S
 

def ExerciseTime(Q, S, T):
    ExTime = np.minimum(Q, S)
    Expected_ExTime = np.mean(ExTime[ExTime <= T])
    
    return Expected_ExTime


def DefaultOption(V, L, epsilon, Q, S, T, r0, N, dt):
    Prices, NoDefault = [], 0
    for i in range(len(Q)):
        if Q[i] < S[i]:
            ExTime = int(Q[i]/dt) # row number
            DiscountPayoff = np.exp(-r0*Q[i]) * max(L[ExTime][i] - epsilon*V[ExTime][i], 0)
            
        elif Q[i] > S[i]:
            ExTime = int(S[i]/dt) # row number
            DiscountPayoff = np.exp(-r0*S[i]) * abs(L[ExTime][i] - epsilon*V[ExTime][i])
            
        else:
            if Q[i] == float("inf") and S[i] == float("inf"): # no default option exercise
                NoDefault += 1
                DiscountPayoff = 0
            else:
                ExTime = int(S[i]/dt) # row number
                DiscountPayoff = np.exp(-r0*S[i]) * abs(L[ExTime][i] - epsilon*V[ExTime][i])
        
        Prices.append(DiscountPayoff)
 
    OptionPrice = np.mean(Prices)
    DefaultProb = 1 - NoDefault/N
    
    return OptionPrice, DefaultProb
                 
                  
def Proj6_2func(lambda1 = 0.2, lambda2 = 0.4, T = 5,
                N = 10000,
                V0 = 20000, mu = -0.1, sigma = 0.2, gamma = -0.4, 
                L0 = 22000, r0 = 0.02, delta = 0.25,
                alpha = 0.7, epsilon = 0.95):
    '''
    Parameters
    ----------
    1. Colaateral price process params
    V0, mu, sigma, gamma = 20000, -0.1, 0.2, -0.4
    lambda1 (The default is 0.2) # Jt Poisson process
    
    2. Loan process params: Lt = a - bc^(12t)
    L0 = 22000, 5
    r0, delta,  = 0.02, 0.25, 0.4 
    lambda2 (The default is 0.4)
    R = r0 + delta*lambda2 # APR of the loan R = r0 + delta*lambda_2
    
    Compute PMT, a, b, c
    r, n = R/12, T*12
    PMT = (L0*r) / (1 - (1 / ((1+r)**n)) )
    a, b, c = PMT/r, PMT/(r*((1+r)**n)), 1+r

    3. Stopping time Q params
    alpha, epsilon = 0.7, 0.95
    beta = (epsilon - alpha)/T # qt = alpha + beta*t (beta = (epsilon - alpha)/T)
    
    4. Stopping time S params
    lambda2 (The default is 0.4) # Poisson process

    5. Simulation params
    N = 10000

    Returns
    -------
    OptionPrice
    DefaultProb
    Expected_ExTime

    '''
    np.random.seed(42)
    
    ## Loan process params
    R = r0 + delta*lambda2 
    r, n = R/12, T*12
    PMT = (L0*r) / (1 - (1 / ((1+r)**n)) )
    a, b, c = PMT/r, PMT/(r*((1+r)**n)), 1+r
    ## stopping time Q params
    beta = (epsilon - alpha)/T 

    dt = 1/np.sqrt(N)
    Ndt = int(T/dt)
    dt = T/Ndt
    
    V = CollateralPrice(V0, mu, sigma, gamma, lambda1, Ndt, N, dt) 
    L = Loan(a, b, c, Ndt, N, dt)
    Q = StopTime_Q(alpha, beta, Ndt, N, dt, V, L)
    S = StopTime_S(lambda2, Ndt, N, dt)
     
    Expected_ExTime = ExerciseTime(Q, S, T)
    OptionPrice, DefaultProb = DefaultOption(V, L, epsilon, Q, S, T, r0, N, dt) 
    
    return OptionPrice, DefaultProb, Expected_ExTime
    

## part 1
D, Prob, Et = Proj6_2func(lambda1 = 0.2, lambda2 = 0.4, T = 5) 

## part 2
OptionPrice_Q2_1 = np.zeros(shape=(6, 9))
DefaultProb_Q2_1 = np.zeros(shape=(6, 9))
Expected_ExTime_Q2_1 = np.zeros(shape=(6, 9))
for i in range(6):
    T_2 = 3+i
    for j in range(9):
        lambda_2 = 0.1*j
        price, prob, time = Proj6_2func(0.2, lambda_2, T_2)
        OptionPrice_Q2_1[i][j] = price
        DefaultProb_Q2_1[i][j] = prob
        Expected_ExTime_Q2_1[i][j] = time


## Option Price        
plt.figure(), plt.plot(range(3, 9), OptionPrice_Q2_1, marker="o", markersize=4)
plt.xlabel("T"), plt.ylabel("Default Option Price")
plt.title("Default Option Price with Lambda1=0.2")
plt.legend(["Lambda2=" + str(round(lambda2, 1)) for lambda2 in np.arange(0, 0.8+0.1, 0.1)])
## Default Probability
plt.figure(), plt.plot(range(3, 9), DefaultProb_Q2_1, marker="o", markersize=4)
plt.xlabel("T"), plt.ylabel("Default Probability")
plt.title("Default Probability with Lambda1=0.2")
plt.legend(["Lambda2=" + str(round(lambda2, 1)) for lambda2 in np.arange(0, 0.8+0.1, 0.1)])
## Expected Exercise Time
plt.figure(), plt.plot(range(3, 9), Expected_ExTime_Q2_1, marker="o", markersize=4)
plt.xlabel("T"), plt.ylabel("Expected Exercise Time")
plt.title("Expected Exercise Time with Lambda1=0.2")
plt.legend(["Lambda2=" + str(round(lambda2, 1)) for lambda2 in np.arange(0, 0.8+0.1, 0.1)])


OptionPrice_Q2_2 = np.zeros(shape=(6, 9))
DefaultProb_Q2_2 = np.zeros(shape=(6, 9))
Expected_ExTime_Q2_2 = np.zeros(shape=(6, 9))
for i in range(6):
    T_2 = 3+i
    for j in range(9):
        lambda_1 = 0.05+0.05*j
        price, prob, time = Proj6_2func(lambda_1, 0.4, T_2)
        OptionPrice_Q2_2[i][j] = price
        DefaultProb_Q2_2[i][j] = prob
        Expected_ExTime_Q2_2[i][j] = time
      
        
## Option Price 
plt.figure(), plt.plot(range(3, 9), OptionPrice_Q2_2, marker="o", markersize=4)
plt.xlabel("T"), plt.ylabel("Default Option Price")
plt.title("Default Option Price with Lambda2=0.4")
plt.legend(["Lambda1=" + str(round(lambda1, 2)) for lambda1 in np.arange(0.05, 0.4+0.05, 0.05)])   
## Default Probability   
plt.figure(), plt.plot(range(3, 9), DefaultProb_Q2_2, marker="o", markersize=4)
plt.xlabel("T"), plt.ylabel("Default Probability")
plt.title("Default Probability with Lambda2=0.4")
plt.legend(["Lambda1=" + str(round(lambda1, 2)) for lambda1 in np.arange(0.05, 0.4+0.05, 0.05)])
## Expected Exercise Time
plt.figure(), plt.plot(range(3, 9), Expected_ExTime_Q2_2, marker="o", markersize=4)
plt.xlabel("T"), plt.ylabel("Expected Exercise Time")
plt.title("Expected Exercise Time with Lambda2=0.4")
plt.legend(["Lambda1=" + str(round(lambda1, 2)) for lambda1 in np.arange(0.05, 0.4+0.05, 0.05)]) 