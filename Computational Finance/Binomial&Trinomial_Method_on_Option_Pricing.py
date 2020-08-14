# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:19:50 2020

@author: Yingxin Wang
"""

############## Computational Methods in Finance ##################
##############           Homework 4            ##################


import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import os
import pandas as pd
import datetime
from scipy.optimize import brentq


########################################### Question 1 ###########################################


T, r, sigma = 0.5, 0.05, 0.24
nvalues = np.array([10, 20, 40, 80, 100, 200, 500])
dt = T/nvalues

def Q_1(n, u, d, p, S0 = 32, K = 30, T = 0.5, r = 0.05):
    
    ExpectedPayoff = 0
    for i in range(n+1):
        payoff = comb(n, i) * (p**i) * ((1-p)**(n-i)) * max(S0*(u**i)*(d**(n-i)) - K, 0)
        ExpectedPayoff += payoff
    OptionPrice = np.exp(-r*T) * ExpectedPayoff
    
    return OptionPrice


#### a
c = 1/2 * (np.exp(-r*dt) + np.exp((r + sigma**2) * dt))
d = c - np.sqrt(c**2 - 1)
u = 1/d
p = (np.exp(r*dt) - d)/(u - d)
Price_Q1_a = [Q_1(nvalues[i], u[i], d[i], p[i]) for i in range(len(nvalues))]
    
#### b
u = np.exp(r*dt) * (1 + np.sqrt(np.exp(sigma**2 * dt) - 1))
d = np.exp(r*dt) * (1 - np.sqrt(np.exp(sigma**2 * dt) - 1))
p = 1/2
Price_Q1_b = [Q_1(nvalues[i], u[i], d[i], p) for i in range(len(nvalues))]

#### c
u = np.exp((r - (sigma**2)/2)*dt + sigma*np.sqrt(dt))
d = np.exp((r - (sigma**2)/2)*dt - sigma*np.sqrt(dt))
p = 1/2
Price_Q1_c = [Q_1(nvalues[i], u[i], d[i], p) for i in range(len(nvalues))]

#### d
u = np.exp(sigma*np.sqrt(dt))
d = np.exp(-sigma*np.sqrt(dt))
p = 1/2 + 1/2*(((r - (sigma**2)/2) * np.sqrt(dt)) / sigma)
Price_Q1_d = [Q_1(nvalues[i], u[i], d[i], p[i]) for i in range(len(nvalues))]

Prices_Q1 = np.array([Price_Q1_a, Price_Q1_b, Price_Q1_c, Price_Q1_d]).T
plt.figure()
plt.plot(nvalues, Prices_Q1, marker="o", markersize=4)
plt.xlabel("n"), plt.ylabel("Option Price")
plt.legend(["(a)", "(b)", "(c)", "(d)"])
    
  
# ########################################### Question 2 ###########################################


os.chdir("D:/UCLA MFE/Spring 2020/A Computational Method in Finance/HW/HW4")
data_amzn = pd.read_csv("AMZN.csv")
data_amzn['Date'] = pd.to_datetime(data_amzn['Date'])
data_amzn = data_amzn.set_index(["Date"])
data_amzn['Return'] = data_amzn['Adj Close'].pct_change()
sigma_Q2 = np.std(data_amzn['Return'])* np.sqrt(252)# data_amzn.iloc[:,5]


r, S0_Q2 = 0.02, data_amzn['Adj Close'][-1]
K_Q2 = round(data_amzn['Adj Close'][-1] * 1.1 / 10) * 10
T_Q2 = (datetime.datetime(2021, 1, 31) - data_amzn.index[-1]).days / 365

def Q_2(sigma, S0, K, T, r = 0.02, n=100):
    dt = T/n
    c = 1/2 * (np.exp(-r*dt) + np.exp((r + sigma**2) * dt))
    d = c - np.sqrt(c**2 - 1)
    u = 1/d
    p = (np.exp(r*dt) - d)/(u - d)
    return Q_1(n, u, d, p, S0, K, T, r)

OptionPrice_Q2_BM = Q_2(sigma = sigma_Q2, S0 = S0_Q2, K = K_Q2, T = T_Q2, r = 0.02, n=100)
OptionPrice_Q2_actual = 229


def ImpliedVol(sigma, target = OptionPrice_Q2_actual):
    return Q_2(sigma, S0 = S0_Q2, K = K_Q2, T = T_Q2) - target

sigma_implied = brentq(ImpliedVol, 0.1, 0.5)
Q_2(sigma_implied, S0 = S0_Q2, K = K_Q2, T = T_Q2) - OptionPrice_Q2_actual


# ########################################### Question 3 ###########################################


class BinomialTree():
    
    def __init__(self, n):
        self.nodes = n
        
        
    def BinomialTree_Method(self, r, sigma, T):
        
        self.dt = T/self.nodes
        c = 1/2 * (np.exp(-r*self.dt) + np.exp((r + sigma**2) * self.dt))
        d = c - np.sqrt(c**2 - 1)
        u = 1/d
        self.p = (np.exp(r*self.dt) - d)/(u - d)
        
        return u, d

    
    def StockPrice_Tree(self, S0, r, sigma, T):
        
        u, d = self.BinomialTree_Method(r, sigma, T)
        
        StockPrice = np.zeros(shape=(self.nodes+1, self.nodes+1))
        for i in range(self.nodes+1):
            for j in range(i, self.nodes+1):
                    StockPrice[i, j] = S0 * (u**(j-i)) * (d**i)
        
        return StockPrice
    
    
    def EuropeanOption_Tree(self, S0, r, sigma, T, K=50, OptionType = "Call"):
        
        StockPrice = self.StockPrice_Tree(S0, r, sigma, T)
        
        RepliPort_Price = np.zeros(StockPrice.shape)
        if OptionType == "Put":
            RepliPort_Price[:, -1] = np.max(np.vstack((np.zeros(StockPrice[:, -1].shape), (K - StockPrice[:, -1]))), axis = 0)
        else:
            RepliPort_Price[:, -1] = np.max(np.vstack((np.zeros(StockPrice[:, -1].shape), (StockPrice[:, -1] - K))), axis = 0)
        
        for j in np.arange(self.nodes-1, -1, -1):
            for i in range(j+1):
                RepliPort_Price[i, j] = np.exp(-r*self.dt)*(self.p*RepliPort_Price[i, j+1] + (1-self.p)*RepliPort_Price[i+1, j+1])
        
        return RepliPort_Price
    
    
    def EuropeanOption_Greeks(self, GreekType, S0, r = 0.03, sigma = 0.2, T = 0.3846, K = 50):
        
        if GreekType == "Delta":
            epsilon = S0*0.01
            OptionPrice_plus = self.EuropeanOption_Tree(S0 + epsilon, r, sigma, T)[0,0]
            OptionPrice_minus = self.EuropeanOption_Tree(S0 - epsilon, r, sigma, T)[0,0]
            
        if GreekType == "Gamma":
            epsilon = 1.2
            OptionPrice_plus = self.EuropeanOption_Greeks("Delta", S0 + epsilon)
            OptionPrice_minus = self.EuropeanOption_Greeks("Delta", S0 - epsilon)
        
        if GreekType == "Theta":
            epsilon = T * 0.001
            OptionPrice_plus = -self.EuropeanOption_Tree(S0, r, sigma, T+epsilon)[0,0]
            OptionPrice_minus = -self.EuropeanOption_Tree(S0, r, sigma, T-epsilon)[0,0]
            
        if GreekType == "Vega":
            epsilon = sigma * 0.001
            OptionPrice_plus = self.EuropeanOption_Tree(S0, r, sigma+epsilon, T)[0,0]
            OptionPrice_minus = self.EuropeanOption_Tree(S0, r, sigma-epsilon, T)[0,0]
        
        if GreekType == "Rho":
            epsilon = r * 0.001
            OptionPrice_plus = self.EuropeanOption_Tree(S0, r+epsilon, sigma, T)[0,0]
            OptionPrice_minus = self.EuropeanOption_Tree(S0, r-epsilon, sigma, T)[0,0]
            
        return (OptionPrice_plus - OptionPrice_minus) / (2*epsilon)
    
    
    
    def AmericanOption_Tree(self, S0, r, sigma, T, K, OptionType = "Call"):    
    
        StockPrice = self.StockPrice_Tree(S0, r, sigma, T) 
        
        OptionPrice_Tree = np.zeros(StockPrice.shape)
        if OptionType == "Put":
            OptionPrice_Tree[:, -1] = np.maximum(K - StockPrice[:, -1], 0)
            EV_Tree = K - StockPrice
        else:
            OptionPrice_Tree[:, -1] = np.maximum(StockPrice[:, -1] - K, 0)
            EV_Tree = StockPrice - K
        
        for j in np.arange(self.nodes-1, -1, -1):
            for i in range(j+1):
                OptionPrice_Tree[i, j] = np.exp(-r*self.dt)*(self.p*OptionPrice_Tree[i, j+1] + (1-self.p)*OptionPrice_Tree[i+1, j+1])
                OptionPrice_Tree[i, j] = max(OptionPrice_Tree[i, j], EV_Tree[i, j])
        
        return OptionPrice_Tree
        
    
                
Tree_Q3 = BinomialTree(n=100)     
S0_values_Q3 = np.arange(20, 82, 2)
D_s = [Tree_Q3.EuropeanOption_Greeks("Delta", S0=s0) for s0 in S0_values_Q3]
D_t = [Tree_Q3.EuropeanOption_Greeks("Delta", S0=49, T = t) for t in np.arange(0, 0.3846+0.01, 0.01)]
T = [Tree_Q3.EuropeanOption_Greeks("Theta", S0=s0) for s0 in S0_values_Q3]
G = [Tree_Q3.EuropeanOption_Greeks("Gamma", S0=s0) for s0 in S0_values_Q3]
V = [Tree_Q3.EuropeanOption_Greeks("Vega", S0=s0) for s0 in S0_values_Q3]
R = [Tree_Q3.EuropeanOption_Greeks("Rho", S0=s0) for s0 in S0_values_Q3]

# plt.figure(), plt.plot(S0_values_Q3, D_s, marker="o", markersize=4)
# plt.xlabel("S0"), plt.ylabel("Delta")
# plt.figure(), plt.plot(S0_values_Q3, T, marker="o", markersize=4)
# plt.xlabel("S0"), plt.ylabel("Theta")
# plt.figure(), plt.plot(S0_values_Q3, G, marker="o", markersize=4)
# plt.xlabel("S0"), plt.ylabel("Gamma")
# plt.figure(), plt.plot(S0_values_Q3, V, marker="o", markersize=4)
# plt.xlabel("S0"), plt.ylabel("Vega")
# plt.figure(), plt.plot(S0_values_Q3, R, marker="o", markersize=4)
# plt.xlabel("S0"), plt.ylabel("Rho")

plt.figure(), plt.plot(np.arange(0, 0.3846+0.01, 0.01), D_t, marker="o", markersize=4)
plt.xlabel("T"), plt.ylabel("Delta")

GreeksMatrix = np.matrix(D_s + T + G + V + R, ).reshape(-1, len(D_s)).T
plt.figure()
plt.plot(S0_values_Q3, GreeksMatrix, marker="o", markersize=4)
plt.legend(["Delta", "Theta", "Gamma", "Vega", "Rho"])


# ########################################### Question 4 ###########################################


T, r, sigma, K = 1, 0.05, 0.3, 100
S0 = 100

Tree_Q4 = BinomialTree(n=100)

EuropeanOption_Q4 = Tree_Q4.EuropeanOption_Tree(S0, r, sigma, T, K, OptionType = "Put")[0, 0]
AmericanOption_Q4 = Tree_Q4.AmericanOption_Tree(S0, r, sigma, T, K, OptionType = "Put")[0, 0]

EuropeanOption_Q4 = [Tree_Q4.EuropeanOption_Tree(s0, r, sigma, T, K, OptionType = "Put")[0, 0] for s0 in np.arange(80, 120+4, 4)]
AmericanOption_Q4 = [Tree_Q4.AmericanOption_Tree(s0, r, sigma, T, K, OptionType = "Put")[0, 0] for s0 in np.arange(80, 120+4, 4)]

plt.figure()
plt.plot(np.arange(80, 120+4, 4), EuropeanOption_Q4, marker="o", markersize=4)
plt.plot(np.arange(80, 120+4, 4), AmericanOption_Q4, marker="o", markersize=4)
plt.xlabel("Current Stock Price"), plt.ylabel("Option Price")
plt.legend(["European Put Option", "American Put Option"])


########################################### Question 5 ###########################################


class TrinomialTree():
    
    def __init__(self, n, T = 0.5):
        self.n = n
        self.dt = T/n
        
    
    def TrinomialTree_Method_1(self, r, sigma):
        '''
        Applied to stock price process
        '''        
        d = np.exp(-sigma * np.sqrt(3*self.dt))
        u = 1/d
        p_d = (r*self.dt*(1-u) + (r*self.dt)**2 + (sigma**2)*self.dt) / ((u-d) * (1-d))
        p_u = (r*self.dt*(1-d) + (r*self.dt)**2 + (sigma**2)*self.dt) / ((u-d) * (u-1))
        p_m = 1 - p_d - p_u
        
        return u, d, p_u, p_m, p_d
    
    
    def TrinomialTree_Method_2(self, r, sigma):
        '''
        Applied to LOG stock price process
        '''        
        x_u, x_d = sigma*np.sqrt(3*self.dt), -sigma*np.sqrt(3*self.dt)
        p_d = 1/2 * ( (
            ((sigma**2)*self.dt + ((r - sigma**2/2)**2)*(self.dt**2)) / (x_u**2) ) - (
                ((r - (sigma**2)/2)*self.dt) / x_u) )
        p_u = 1/2 * ( (
            ((sigma**2)*self.dt + ((r - sigma**2/2)**2)*(self.dt**2)) / (x_u**2) ) + (
                ((r - (sigma**2)/2)*self.dt) / x_u) )
        p_m = 1 - p_d - p_u
        
        return x_u, x_d, p_u, p_m, p_d
        
    
    def OptionPrice_Tree(self, S0, r, sigma, K, TrinomialTree_Method = 1):
        
        u_index = np.arange(self.n, 0, -1).tolist() + [0] * (self.n+1)
        d_index = np.matrix(sorted(u_index))
        u_index = np.matrix(u_index)
        
        if TrinomialTree_Method == 1:
            u, d, p_u, p_m, p_d = self.TrinomialTree_Method_1(r, sigma)
            StockPrice_T = S0 * np.multiply((u**u_index), (d**d_index))
            
        if TrinomialTree_Method == 2:
            x_u, x_d, p_u, p_m, p_d = self.TrinomialTree_Method_2(r, sigma)    
            LogStockPrice_T = np.log(S0) + (x_u * u_index) + (x_d * d_index)
            StockPrice_T = np.exp(LogStockPrice_T)
        
        OptionPrice = np.zeros(shape=((2*self.n)+1, self.n + 1))
        OptionPrice[:, -1] = np.maximum(StockPrice_T - K, 0)
        for j in np.arange(self.n - 1, -1, -1):
            for i in range(self.n - j, self.n + j+ 1):
                OptionPrice[i, j] = np.exp(-r*self.dt) * (p_u*OptionPrice[i-1, j+1] + p_m*OptionPrice[i, j+1] + p_d*OptionPrice[i+1, j+1])
        
        return  OptionPrice[self.n, 0]      


nvalues_Q5 = [10, 15, 20, 40, 70, 80, 100, 200, 500]
S0_Q5, r_Q5, sigma_Q5, K_Q5 = 32, 0.05, 0.24, 30    

CallOption_Q5_a, CallOption_Q5_b = [], []
for n in nvalues_Q5:
    Tree_Q5 = TrinomialTree(n)
    CallOption_Q5_a.append(Tree_Q5.OptionPrice_Tree(S0_Q5, r_Q5, sigma_Q5, K_Q5, TrinomialTree_Method = 1))
    CallOption_Q5_b.append(Tree_Q5.OptionPrice_Tree(S0_Q5, r_Q5, sigma_Q5, K_Q5, TrinomialTree_Method = 2))

plt.figure()
plt.plot(nvalues_Q5, CallOption_Q5_a, marker="o", markersize=4)
plt.plot(nvalues_Q5, CallOption_Q5_b, marker="o", markersize=4)
plt.xlabel("n"), plt.ylabel("Call Option Price")
plt.legend(["Trinomial Tree using stock price", "Trinomial Tree using log stock price"])


# ########################################### Question 6 ###########################################


class LDS_SimulationOption():
    '''
    Using LDS-Halton seq to simulate option price
    '''
    
    def __init__(self, N, b1, b2):
        '''
        Default setting: Halton seq with base [b1, b2]
        '''
        self.N = N
        self.Bases = [b1, b2]       
        Halton_seq = []
        for base in self.Bases:
            for n in range(1, N+1):
                Halton_n_base = []
                while n // base > 0:
                    Halton_n_base.append(n % base)
                    n //= base   
                Halton_n_base.append(n % base)
                
                Halton_nth = sum([Halton_n_base[i]/(base**(i+1)) for i in range(len(Halton_n_base))])
                Halton_seq.append(Halton_nth)    
            
        self.Halton_seq = np.reshape(np.array(Halton_seq), (N, -1), order='F')                        
    
    
    def Normrv_BM(self, Halton = True):
        '''
        Generate r.v. of NORMAL distribution by using Box-Muller method [1-TO-1 MATCH]
        '''
        U = self.Halton_seq
        # print(U)
        Z1 = [np.sqrt(-2* np.log(u[0])) * np.cos(2* np.pi * u[1]) for u in U]
        Z2 = [np.sqrt(-2* np.log(u[0])) * np.sin(2* np.pi * u[1]) for u in U]
        Zn = np.array(Z1 + Z2)
        
        return Zn
    
    
    def EuropeanOption(self, S0, r, sigma, T, K, OptionType = "Call"):
        
        W_T = np.sqrt(T) * self.Normrv_BM(Halton = True)
        S_T = S0* np.exp((r - (sigma**2)/2) * T + sigma * W_T)

        Payoff = np.maximum(S_T - K, 0)
        OptionPrice = np.exp(-r*T) * np.mean(Payoff)
        
        return OptionPrice

    
N, b1, b2 = 100, 2, 3
S0, r, sigma, T, K = 88, 0.05, 0.3, 1, 100

PriceSim = LDS_SimulationOption(N, b1, b2)
Z = PriceSim.Normrv_BM()
C = PriceSim.EuropeanOption(S0, r, sigma, T, K)

