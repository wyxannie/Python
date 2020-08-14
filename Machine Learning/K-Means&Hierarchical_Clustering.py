# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:08:15 2020

@author: Yingxin Wang
"""

import os
import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing
import matplotlib.pyplot as plt
#import scipy.cluster.hierarchy as sch

#import data
os.chdir("C:/Users/72487/Desktop/Code Killer/Kaggle_Project/Cluster")
data_train = pd.read_csv("Final_Train_Data_24-Mar-2020.csv")
data_test = pd.read_csv("Final_Test_Data_24-Mar-2020.csv")

target = data_train['target']
data_combine = pd.concat([data_train, data_test], sort=False)
data_combine = data_combine.drop(['Id', 'target', 'LotFrontage_Flag'], axis=1)

##standardize
data_1 = preprocessing.scale(data_combine)
data_2 = np.transpose(data_1)

   
X = data_2
wss = np.zeros([113, 1])
for k in range(1, 113):
    wss[k-1] = cluster.KMeans(n_clusters=k).fit(X).inertia_
    
plt.plot(range(1, 113), wss[1:113])
plt.title("Elbow Method") 
plt.xlabel("Number of Cluster") 
plt.ylabel("Within-cluster Sum of Squares")


#Hierarchical Clustering
NumOfCluster = 40
Hcluster = cluster.AgglomerativeClustering(NumOfCluster)
Hcluster.fit(data_2)
label = Hcluster.labels_
#leaves = Hcluster.n_leaves_
#children = Hcluster.children_

VarName = data_combine.columns.values.tolist()
ClusterResult = dict(zip(VarName, label))
ClusterResult_1 = sorted(ClusterResult.items(), key=lambda item:item[1])   
ClusterResult_2 = pd.DataFrame(ClusterResult_1)
ClusterResult_2.columns = ['VarName', 'Cluster']
                        
# Correlation 1: with target
CorWithTarget = np.zeros([len(ClusterResult_1), 1])
for i in range(0, len(ClusterResult_1)):
    CorWithTarget[i] = np.corrcoef(data_train[ClusterResult_1[i][0]], target)[0][1]
ClusterResult_2['CorWithTarget'] = CorWithTarget
 
# Correlation 2: within cluster
MeanCorVarName = []
MeanCorValue = []
for i in range(0, NumOfCluster):
    subcluster = ClusterResult_2.loc[ClusterResult_2['Cluster'] == i]['VarName']
    MeanCor = dict(np.mean(data_train[subcluster].corr()))
    MeanCorVarName += list(MeanCor)
    MeanCorValue += list(MeanCor.values())


#sum(ClusterResult_2['VarName'] == MeanCorVarName)
ClusterResult_2['CorWithinCluster'] = MeanCorValue
ClusterResult_2.to_csv("ClusterResult_2.csv")  

# Select out final variables
FinalVariables = []
for i in range(0, NumOfCluster):
    subcluster = ClusterResult_2.loc[ClusterResult_2['Cluster'] == i]
    if len(subcluster) == 2:
        FinalVariables += subcluster.loc[subcluster['CorWithTarget'] == 
                               max(abs(subcluster['CorWithTarget']))]['VarName'].tolist()
    elif len(subcluster) == 1:
        FinalVariables += subcluster['VarName'].tolist()
    else:
        FinalVariables += subcluster.loc[subcluster['CorWithTarget'] == 
                               max(abs(subcluster['CorWithTarget']))]['VarName'].tolist()
        FinalVariables += subcluster.loc[subcluster['CorWithinCluster'] == 
                               max(abs(subcluster['CorWithinCluster']))]['VarName'].tolist()

FinalVariables_1 = list(set(FinalVariables))
FinalVariables_1 = ['Neighborhood' if x=='OverallCond_Qual' else x for x in FinalVariables_1]
# replace 'OverallCon_Qual' with 'Neighborhood'

data_train_1 = data_train[FinalVariables_1]
data_train_1['target'] = target
data_train_1 = data_train_1.set_index(data_train['Id'])

data_test_1 = data_test[FinalVariables_1]
data_test_1 = data_test_1.set_index(data_test['Id'])
#data_train_1.columns.values == data_test_1.columns.values

data_train_1.to_csv("FinalData_Train.csv")  
data_test_1.to_csv("FinalData_Test.csv")  


