# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:27:15 2021

@author: pedro
"""

if __name__ == "__main__":
    from mgedt import MGEDT
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve
    #from sklearn.metrics import plot_roc_curve
    import matplotlib.pyplot as plt
    import matplotlib
    
    
    
    dtr = pd.read_csv("../datasets/Promos/TEST2/Train-IDF-1.csv", sep=";")
    dtest = pd.read_csv("../datasets/Promos/TEST2/Test-IDF-1.csv", sep=";")
    
    dtrain, dval = train_test_split(dtr, test_size=0.1, stratify=dtr['target'])
    X = dtrain.drop('target', axis=1)
    y = dtrain['target']
    X_val = dval.drop('target', axis=1)
    y_val = dval['target']
    X_test = dtest.drop('target', axis=1)
    y_test = dtest['target']
    mgedt = MGEDT()
    mgedt.fit(X, y, X_val, y_val, pop=50, gen=50, lamarck=False, 
              folder_name="PROMOS_1")
    mgedt.predict(X_test)
    pred = mgedt.predict(X_test)
    
    auc = roc_auc_score(y_test, pred)*100
    print(auc)
    
    ROC = pd.DataFrame(roc_curve(y_test, pred, pos_label='Sale')).T
    ROC.columns = ["FPR", "TPR", "TH"]
    
    matplotlib.use('module://ipykernel.pylab.backend_inline', force=True)
    plt.plot(ROC['FPR'], ROC['TPR'], color='darkorange')
    plt.show()
    
    mgedt.refit(gen=10)
    mgedt.predict(X_test)
    pred = mgedt.predict(X_test)
    
    auc = roc_auc_score(y_test, pred)*100
    print(auc)
    
    ROC = pd.DataFrame(roc_curve(y_test, pred, pos_label='Sale')).T
    ROC.columns = ["FPR", "TPR", "TH"]
    
    matplotlib.use('module://ipykernel.pylab.backend_inline', force=True)
    plt.plot(ROC['FPR'], ROC['TPR'], color='darkorange')
    plt.show()
    
    dtr = pd.read_csv("../datasets/Promos/TEST2/Train-IDF-2.csv", sep=";")
    dtest = pd.read_csv("../datasets/Promos/TEST2/Test-IDF-2.csv", sep=";")
    
    dtrain, dval = train_test_split(dtr, test_size=0.1, stratify=dtr['target'])
    X = dtrain.drop('target', axis=1)
    y = dtrain['target']
    X_val = dval.drop('target', axis=1)
    y_val = dval['target']
    X_test = dtest.drop('target', axis=1)
    y_test = dtest['target']
    mgedt = MGEDT()
    mgedt.fit(X, y, X_val, y_val, pop=50, gen=10, lamarck=True, 
              folder_name="PROMOS_2")
    
    mgedt.predict(X_test)
    pred = mgedt.predict(X_test)
    
    auc = roc_auc_score(y_test, pred)*100
    print(auc)
    
    ROC = pd.DataFrame(roc_curve(y_test, pred, pos_label='Sale')).T
    ROC.columns = ["FPR", "TPR", "TH"]
    
    matplotlib.use('module://ipykernel.pylab.backend_inline', force=True)
    plt.plot(ROC['FPR'], ROC['TPR'], color='darkorange')
    plt.show()