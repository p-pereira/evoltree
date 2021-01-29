# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:27:15 2021

@author: pedro
"""

from mgedt import MGEDT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt

mgedt = MGEDT()
mgedt.params
mgedt.param_list

d = pd.read_csv("../datasets/Promos/TEST2/Train-IDF-1.csv", sep=";")
d.columns

dtrain, dtest = train_test_split(d, test_size=0.3, stratify=d['target'])
dtrain, dval = train_test_split(dtrain, test_size=0.3, stratify=dtrain['target'])
X = dtrain.drop('target', axis=1)
y = dtrain['target']
X_val = dval.drop('target', axis=1)
y_val = dval['target']
X_test = dtest.drop('target', axis=1)
y_test = dtest['target']
mgedt.fit(X, y, X_val, y_val)
mgedt.params
mgedt.predict(X_test)
pred = mgedt.predict(X_test)

roc_auc_score(y_test, pred)

ROC = pd.DataFrame(roc_curve(y_test, pred, pos_label='NoSale')).T
ROC.columns = ["FPR", "TPR", "TH"]
ROC

plt.plot(ROC['FPR'], ROC['TPR'], color='darkorange')
plt.show()
plt.figure()