# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:06:19 2020

@author: stellajin
"""

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

np.random.seed(0)

data = np.array(pd.read_csv('data/white.csv'))
data = np.array(list(set([tuple(t) for t in data])))


X = data[:,0:11]
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)
Y = data[:,-1]
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.75)

def calc_mse(y1,y2):
    summation = 0
    n = len(y1)
    for i in range (0,n):
        difference = y1[i] - y2[i]
        squared_difference = difference**2
        summation = summation + squared_difference
    MSE = summation/n
    return MSE

clf = MLPRegressor(activation='logistic',solver='lbfgs', alpha=0.001,hidden_layer_sizes=(5,3),max_iter=500)
clf.fit(Xtr, Ytr)
print("training scroe: ", clf.score(Xtr, Ytr))
Yhat = clf.predict(Xte)
print("MSE of predicted label: ", calc_mse(Yte,Yhat))
Yhat = np.floor(Yhat)
print("precision of predicted label", np.sum(Yte==Yhat)/Yte.shape[0])



kfold = model_selection.KFold(n_splits=5,random_state=2)
cv_results =model_selection.cross_val_score(SVR(),Xtr, Ytr,cv= kfold, scoring = 'neg_mean_absolute_error')
print("Cross Validation Score:", -1*(cv_results).mean())