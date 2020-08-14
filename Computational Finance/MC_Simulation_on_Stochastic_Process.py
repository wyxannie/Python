# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:53:52 2020

@author: Yingxin Wang
"""

############## Computational Methods in Finance ##################
##############           Homework 3            ##################


import numpy as np
import matplotlib.pyplot as plt
import time


########################################### Question 1 ###########################################


def Q_1(X0, Y0, N = 10000, dt = 0.001):
    
    np.random.seed(0)

    W = np.random.normal(0, 1, size=(int(2/dt), N))
    Z = np.random.normal(0, 1, size=(int(3/dt), N))
    
    X, Y = np.zeros(W.shape), np.zeros(Z.shape)
    X[0, ], Y[0, ] = X0, Y0
    for i in range(1, int(3/dt)):    
        t = (i-1)* dt
        if i < 2/dt:
            X[i, ] = X[i-1, ] + ((1/5 - 1/2* X[i-1, ]) *dt + 2/3* W[i-1, ] * np.sqrt(dt))
        Y[i, ] = Y[i-1, ] + ((2/(1+t)* Y[i-1, ] + (1+t**3)/3) *dt + (1+t**3)/3* Z[i-1, ] * np.sqrt(dt))
    
    p1 = np.mean(Y[int(2/dt)-1, ] > 5)
    e1 = np.mean([-(abs(x)**(1/3)) if x < 0 else x**(1/3) for x in X[-1, ]])
    e2 = np.mean(Y[-1, ])
    e3 = np.mean(X[-1, ] * Y[int(2/dt)-1, ] * (X[-1, ] > 1))
    
    return p1, e1, e2, e3


X0, Y0 = 1, 3/4
p1, e1, e2, e3 = Q_1(X0, Y0)


########################################### Question 2 ###########################################


def Q_2(X0, N = 10000, dt = 0.001):
    
    np.random.seed(0)

    W = np.random.normal(0, 1, size=(int(3/dt), N))
    Z = np.random.normal(0, 1, size=(int(3/dt), N))
    
    X = np.zeros(W.shape)
    X[0, ] = X0
    for i in range(1, int(3/dt)):
        X[i, ] = X[i-1, ] + (
            1/4*X[i-1, ]*dt + 1/3*X[i-1, ]* W[i-1, ]*np.sqrt(dt) - 3/4*X[i-1, ]* Z[i-1, ]*np.sqrt(dt))
    
    W1 = sum(W[0:int(1/dt), ] * np.sqrt(dt))
    Z1 = sum(Z[0:int(1/dt), ] * np.sqrt(dt))
    Y1 = np.exp(-0.08*1 + 1/3*W1 + 3/4*Z1)
        
    e1 = np.mean([-(abs(x)**(1/3)) if x < 0 else x**(1/3) for x in (X[-1, ]+1)])
    e2 = np.mean(X[int(1/dt)-1, ] * Y1)
    
    return e1, e2


X0 = 1
e1, e2 = Q_2(X0)


########################################### Question 3 ###########################################


#### a

S0, T, X, r, sigma = 88, 0.5, 100, 0.04, 0.2

def EuropeanOption_MC(S0, T, X, r, sigma, N = 100000, dt = 0.004):
    
    np.random.seed(0)
    
    W = np.random.normal(0, 1, size=(int(T/dt), N))
    Stock = np.zeros(W.shape)
    Stock[0, ] = S0
    for i in range(1, int(T/dt)):
        Stock[i, ] = Stock[i-1, ] + r*Stock[i-1, ]*dt + sigma*Stock[i-1, ]*W[i-1, ]*np.sqrt(dt)
        
    Payoff = np.max(np.vstack((np.zeros(Stock[-1, ].shape), (Stock[-1, ] - X))), axis = 0)
    Price = np.exp(-r*T) * np.mean(Payoff)
    
    return Price

Price_MC = EuropeanOption_MC(S0, T, X, r, sigma)


#### b

def CDF_Normal(x):
    coefs = [1, 0.0498673470, 0.0211410061, 0.0032776263,
             0.0000380036, 0.0000488906, 0.0000053830]
    if x < 0:
        return 1 - CDF_Normal(-x)
    else:
        return 1 - 1/2 * (sum([coefs[i]*(x**i) for i in range(7)]) ** (-16))
    
    
def EuropeanOption_BS(S0, T, X, r, sigma):
    
    d1 = (np.log(S0/X) + (r + (1/2)*(sigma**2))*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Price = S0 * CDF_Normal(d1) - X * np.exp(-r*T) * CDF_Normal(d2)
    
    return Price

Price_BS = EuropeanOption_BS(S0, T, X, r, sigma)


#### C

def Option_Greeks(Greek, S0, T = 0.5, X = 20, r = 0.04, sigma = 0.25, dt = 0.004):
    
    if Greek == 'Delta':
        epsilon = S0 * 0.001
        price_plus = EuropeanOption_MC(S0+epsilon, T, X, r, sigma)
        price_minus = EuropeanOption_MC(S0-epsilon, T, X, r, sigma)
        delta = (price_plus - price_minus) / (2*epsilon)
        return delta

    if Greek == 'Gamma':
        epsilon = S0 * 0.001
        price_plus = Option_Greeks('Delta', S0+epsilon)
        price_minus = Option_Greeks('Delta', S0-epsilon)
        gamma = (price_plus - price_minus) / (2*epsilon)
        
        return gamma
    
    if Greek == 'Theta':
        epsilon = dt
        price_plus = EuropeanOption_MC(S0, T+epsilon, X, r, sigma)
        price_minus = EuropeanOption_MC(S0, T-epsilon, X, r, sigma)
        theta = -(price_plus - price_minus) / (2*epsilon)
        return theta
    
    if Greek == 'Vega':
        epsilon = sigma * 0.001
        price_plus = EuropeanOption_MC(S0, T, X, r, sigma+epsilon)
        price_minus = EuropeanOption_MC(S0, T, X, r, sigma-epsilon)
        vega = (price_plus - price_minus) / (2*epsilon)
        return vega
    
    if Greek == 'Rho':
        epsilon = r * 0.001
        price_plus = EuropeanOption_MC(S0, T, X, r+epsilon, sigma)
        price_minus = EuropeanOption_MC(S0, T, X, r-epsilon, sigma)
        rho = (price_plus - price_minus) / (2*epsilon)
        return rho
    
 
D = [Option_Greeks('Delta', S0) for S0 in range(15, 25+1)]
G = [Option_Greeks('Gamma', S0) for S0 in range(15, 25+1)]
T = [Option_Greeks('Theta', S0) for S0 in range(15, 25+1)]
V = [Option_Greeks('Vega', S0) for S0 in range(15, 25+1)]
R = [Option_Greeks('Rho', S0) for S0 in range(15, 25+1)]

plt.figure(), plt.plot(range(15, 25+1), D), plt.xlabel("S0"), plt.ylabel("Greek Delta")
plt.figure(), plt.plot(range(15, 25+1), G), plt.xlabel("S0"), plt.ylabel("Greek Gamma")
plt.figure(), plt.plot(range(15, 25+1), T), plt.xlabel("S0"), plt.ylabel("Greek Theta")
plt.figure(), plt.plot(range(15, 25+1), V), plt.xlabel("S0"), plt.ylabel("Greek Vega")        
plt.figure(), plt.plot(range(15, 25+1), R), plt.xlabel("S0"), plt.ylabel("Greek Rho")
        

########################################### Question 4 ###########################################        


def Truncation_Zero(x):
    '''
    x: matrix
    '''
    return np.max(np.vstack((np.zeros(x.shape), x)), axis = 0)


def Q_4(Method, rho, r, S0, K, V0, sigma, alpha, beta, T=1, dt=0.004, N=10000):
    
    np.random.seed(0)
    
    Z = np.random.normal(0, 1, size=(2, int(T/dt), N))
    W1 = 0 + 1* Z[0, ]
    W2 = 0 + 1* rho * Z[0, ] + 1* np.sqrt(1 - rho**2) * Z[1, ]
    
    Stock, Volatility = np.zeros(W1.shape), np.zeros(W2.shape)
    Stock[0, ], Volatility[0, ] = S0, V0
    
    if Method == 'Full Truncation':
    #### Full Truncation
        for i in range(1, int(T/dt)):
            
            Stock[i, ] = Stock[i-1, ] + (
                r*Stock[i-1, ]*dt ) + (
                    np.sqrt(Truncation_Zero(Volatility[i-1, ]))*Stock[i-1, ]*W1[i-1, ]*np.sqrt(dt))
            Volatility[i, ] = Volatility[i-1, ] + (
                alpha*(beta-Truncation_Zero(Volatility[i-1, ]))*dt) + (
                    sigma*np.sqrt(Truncation_Zero(Volatility[i-1, ]))*W2[i-1, ]*np.sqrt(dt))
    
                    
    if Method == 'Partial Truncation':    
    #### Partial Truncation
        for i in range(1, int(T/dt)):
            
            Stock[i, ] = Stock[i-1, ] + (
                r*Stock[i-1, ]*dt ) + (
                    np.sqrt(Truncation_Zero(Volatility[i-1, ]))*Stock[i-1, ]*W1[i-1, ]*np.sqrt(dt))
            Volatility[i, ] = Volatility[i-1, ] + (
                alpha * (beta - Volatility[i-1, ]) * dt) + (
                    sigma*np.sqrt(Truncation_Zero(Volatility[i-1, ]))*W2[i-1, ]*np.sqrt(dt))
                
    
    if Method == 'Reflection Method':
    #### Reflection Method
        for i in range(1, int(T/dt)):
            
            Stock[i, ] = Stock[i-1, ] + (
                r*Stock[i-1, ]*dt ) + (
                    np.sqrt(abs(Volatility[i-1, ]))*Stock[i-1, ]*W1[i-1, ]*np.sqrt(dt))
            Volatility[i, ] = abs(Volatility[i-1, ]) + (
                alpha*(beta-abs(Volatility[i-1, ]))*dt) + (
                    sigma*np.sqrt(abs(Volatility[i-1, ]))*W2[i-1, ]*np.sqrt(dt))
    
                
    Payoff = np.max(np.vstack((np.zeros(Stock[-1, ].shape), (Stock[-1, ] - K))), axis = 0)
    Price = np.exp(-r*T) * np.mean(Payoff)
    
    return Price


rho, r, S0, K, V0, sigma, alpha, beta = -0.6, 0.03, 48, 50, 0.05, 0.42, 5.8, 0.0625
C1 = Q_4('Full Truncation', rho, r, S0, K, V0, sigma, alpha, beta, T=1, dt=0.004, N=100)
C2 = Q_4('Partial Truncation', rho, r, S0, K, V0, sigma, alpha, beta)
C3 = Q_4('Reflection Method', rho, r, S0, K, V0, sigma, alpha, beta)

    
########################################### Question 5 ###########################################            
    

#### a

def LGM_Random(X0, N):
    '''
    LGM method: X_n+1 = (7^5 * X_n) modulo (2^31-1)
    '''
    Xn = [X0]
    for i in range(0, int(N-1)):
      Xn.append((7**5 * Xn[-1]) % (2**31 - 1)) 
    Un = np.transpose(np.array(np.mat(Xn))) / (2**31 - 1)
    
    return Un 


X_a = np.hstack((LGM_Random(time.time(), N=100), LGM_Random(time.time()+922, N=100)))   


#### b
    
def Get_nd_HaltonSeq(N, Bases):
    
    Halton_seq = []
    for base in Bases:
        for n in range(1, N+1):
            Halton_n_base = []
            while n // base > 0:
                Halton_n_base.append(n % base)
                n //= base   
            Halton_n_base.append(n % base)
            
            Halton_nth = sum([Halton_n_base[i]/(base**(i+1)) for i in range(len(Halton_n_base))])
            Halton_seq.append(Halton_nth)    
        
    Result_seq = np.reshape(np.array(Halton_seq), (N, -1), order='F')
                    
    return Result_seq

# Halton_seq_5 = Get_nd_HaltonSeq(N=10000, Bases=[5])
H_b = Get_nd_HaltonSeq(100, Bases=[2, 7])


#### c
   
H_c = Get_nd_HaltonSeq(100, Bases=[2, 4])   


#### d

plt.figure(), plt.scatter(X_a[:, 0], X_a[:,1])
plt.figure(), plt.scatter(H_b[:, 0], H_b[:,1]) 
plt.figure(), plt.scatter(H_c[:, 0], H_c[:,1])  

    
#### e   

def CubicRoot(x):
    return np.sign(x) * (np.abs(x)) ** (1/3)

def Q5_e(Bases, N = 10000):
    '''
    Bases: type=list; n-dimension need n bases
    '''
    nd_seq = Get_nd_HaltonSeq(N, Bases)
    X, Y = nd_seq[:, 0], nd_seq[:, 1]
    
    integral = np.exp(-X*Y) * (np.sin(6*np.pi * X) + CubicRoot(np.cos(2*np.pi * Y)))
    
    return np.mean(integral)


I1 = Q5_e(Bases = [2, 4]) 
I2 = Q5_e(Bases = [2, 7]) 
I3 = Q5_e(Bases = [5, 7])    
    
    
    
    