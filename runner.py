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
    import matplotlib.pyplot as plt
    import matplotlib
    import os
    import re
    import pickle
    from time import time
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from math import log10
    MODES = ['BEST2', 'TEST2']
    RES = []
    
    for mode in MODES:
        DATA_DIR = "datasets/Promos/{}".format(mode)
        
        tr_files = [f for f in os.listdir(DATA_DIR) 
                    if re.match(r'Train+.*\.csv', f)]
        ts_files = [f for f in os.listdir(DATA_DIR) 
                    if re.match(r'Test+.*\.csv', f)]
        mgedt = MGEDT()
        mgedtl = MGEDT()
        DT = DecisionTreeClassifier()
        RF = RandomForestClassifier(n_jobs=-1)
        for n, (tr_filename, ts_filename) in enumerate(zip(tr_files, ts_files)):
            dtr = pd.read_csv("{0}/{1}".format(DATA_DIR, tr_filename), sep=";")
            dtest = pd.read_csv("{0}/{1}".format(DATA_DIR, ts_filename), sep=";")
            
            dtrain, dval = train_test_split(dtr, test_size=0.1, stratify=dtr['target'])
            X = dtrain.drop('target', axis=1)
            y = dtrain['target']
            X_val = dval.drop('target', axis=1)
            y_val = dval['target']
            X_test = dtest.drop('target', axis=1)
            y_test = dtest['target']
            
            #"""
            s = time()
            if not mgedt.fitted:
                mgedt.fit(X, y, X_val, y_val, pop=100, gen=100, lamarck=False, 
                          experiment_name="PROMOS",
                          folder_name="MGEDT_{0}_{1}".format(mode, n),
                          target_seed_folder="MGEDT_PROMOS_{0}".format(mode))
            else:
                mgedt.fit_new_data(X, y, X_val, y_val, pop=100, gen=25, 
                                   lamarck=False, experiment_name="PROMOS", 
                                   folder_name="MGEDT_{0}_{1}".format(mode, n), 
                                   target_seed_folder="MGEDT_PROMOS_{0}".format(mode))
            e = time()
            tr_t1 = e - s
            
            pickle.dump(mgedt.population, 
                        open("{0}/pop.pkl".format(mgedt.params['FILE_PATH']),
                             "wb"))
            
            s = time()
            pred1 = mgedt.predict(X_test)
            e = time()
            ts_t1 = (e - s) / X_test.shape[0]
            
            auc1 = roc_auc_score(y_test, pred1)*100
            ROC1 = pd.DataFrame(roc_curve(y_test, pred1, pos_label='Sale')).T
            ROC1.columns = ["FPR", "TPR", "TH"]
            ROC1.to_csv("{}/ROC.csv".format(mgedt.params['FILE_PATH']), sep=";")
            #"""
            s = time()
            if not mgedtl.fitted:
                mgedtl.fit(X, y, X_val, y_val, pop=100, gen=100, lamarck=True, 
                          experiment_name="PROMOS",
                          folder_name="MGEDTL_{0}_{1}".format(mode, n), 
                          target_seed_folder="MGEDTL_PROMOS_{0}".format(mode))
            else:
                mgedtl.fit_new_data(X, y, X_val, y_val, pop=100, gen=25, 
                                    lamarck=True, experiment_name="PROMOS", 
                                    folder_name="MGEDTL_{0}_{1}".format(mode, n), 
                                    target_seed_folder="MGEDTL_PROMOS_{0}".format(mode))
            
            e = time()
            tr_t2 = e - s
            
            pickle.dump(mgedtl.population, 
                        open("{0}/pop.pkl".format(mgedtl.params['FILE_PATH']),
                             "wb"))
            s = time()
            pred2 = mgedtl.predict(X_test)
            e = time()
            ts_t2 = (e - s) / X_test.shape[0]
            
            auc2 = roc_auc_score(y_test, pred2)*100
            
            ROC2 = pd.DataFrame(roc_curve(y_test, pred2, pos_label='Sale')).T
            ROC2.columns = ["FPR", "TPR", "TH"]
            ROC2.to_csv("{}/ROC.csv".format(mgedtl.params['FILE_PATH']), sep=";")
            
            s = time()
            DT.fit(X, y)
            e = time()
            tr_t3 = e - s
            s = time()
            pred3 = DT.predict_proba(X_test)[:,1]
            e = time()
            ts_t3 = (e - s) / X_test.shape[0]
            auc3 = roc_auc_score(y_test, pred3)*100
            ROC3 = pd.DataFrame(roc_curve(y_test, pred3, pos_label='Sale')).T
            ROC3.columns = ["FPR", "TPR", "TH"]
            
            s = time()
            RF.fit(X, y)
            e = time()
            tr_t4 = e - s
            s = time()
            pred4 = RF.predict_proba(X_test)[:,1]
            e = time()
            ts_t4 = (e - s) / X_test.shape[0]
            auc4 = roc_auc_score(y_test, pred4)*100
            ROC4 = pd.DataFrame(roc_curve(y_test, pred4, pos_label='Sale')).T
            ROC4.columns = ["FPR", "TPR", "TH"]
            
            #"""
            matplotlib.use('module://ipykernel.pylab.backend_inline', force=True)
            fig, ax = plt.subplots(1,1, figsize=(5.5,5))
            plt.plot(ROC2['FPR'], ROC2['TPR'], color='royalblue', ls="--", lw=2,
                     label="MGEDTL={}%".format(round(auc2, 2)))
            plt.plot(ROC1['FPR'], ROC1['TPR'], color='darkorange', ls="-", lw=2,
                     label="MGEDT={}%".format(round(auc1, 2)))
            plt.plot(ROC3['FPR'], ROC3['TPR'], color='firebrick', ls="-.", lw=2,
                     label="DT={}%".format(round(auc3, 2)))
            plt.plot(ROC4['FPR'], ROC4['TPR'], color='forestgreen', ls=":", lw=2,
                     label="RF={}%".format(round(auc4, 2)))
            plt.plot([0,1], [0,1], color="black", ls='--', label="baseline=50%")
            plt.legend(loc=4)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            legend = ax.legend(frameon=True, framealpha=1, ncol=1, loc=4,
                           fancybox=True, shadow=False, bbox_to_anchor=(1, 0.04))
            _ = legend.get_frame().set_facecolor('white')
            ax.grid(linestyle='--', linewidth=0.6, color='gray', alpha=0.5)
            plt.subplots_adjust(top = 0.99, bottom = 0.08, right = 0.99, left = 0.1,
                            hspace = 0, wspace = 0.01)
            ax.set_facecolor('white')
            plt.savefig("results/PROMOS/AUC_{0}_{1}.pdf".format(mode, n))
            plt.show()
            
            evals1 = mgedt.evaluate_all(X_test, y_test)
            evals1[1] = [log10(node) for node in evals1[1]]
            evals2 = mgedtl.evaluate_all(X_test, y_test)
            evals2[1] = [log10(node) for node in evals2[1]]
            evals3 = [auc3, mgedt.__get_tree_complexity__(DT, X_test.columns)]
            evals3[1] = log10(evals3[1])
            evals4 = [auc4, mgedt.__get_tree_complexity__(RF, X_test.columns)]
            evals4[1] = log10(evals4[1])
            
            fig, ax = plt.subplots(1,1, figsize=(5.5,5))
            plt.plot(evals2[0], evals2[1], color='royalblue', ls="--", lw=2,
                     label="MGEDTL", marker="D", ms=5)
            plt.plot(evals1[0], evals1[1], color='darkorange', ls="-", lw=2,
                     label="MGEDT", marker="o")
            plt.plot(evals3[0], evals3[1], color='firebrick', ls='None', ms=5,
                     marker="v", label="DT")
            plt.plot(evals4[0], evals4[1], color='forestgreen', ls='None', 
                     ms=5, marker="s", label="RF")
            plt.legend(loc=4)
            plt.xlabel("AUC")
            plt.ylabel("Complexity (log10)")
            legend = ax.legend(frameon=True, framealpha=1, ncol=1, loc=4,
                           fancybox=True, shadow=False, bbox_to_anchor=(1, 0.04))
            _ = legend.get_frame().set_facecolor('white')
            ax.grid(linestyle='--', linewidth=0.6, color='gray', alpha=0.5)
            plt.subplots_adjust(top = 0.99, bottom = 0.08, right = 0.99, left = 0.1,
                            hspace = 0, wspace = 0.01)
            ax.set_facecolor('white')
            plt.savefig("results/PROMOS/Pareto_{0}_{1}.pdf".format(mode, n))
            plt.show()
            
            RES.append(['MGEDT', mode, n, auc1, tr_t1, ts_t1])
            RES.append(['MGEDTL', mode, n, auc2, tr_t2, ts_t2])
            RES.append(['DT', mode, n, auc3, tr_t3, ts_t3])
            RES.append(['RF', mode, n, auc4, tr_t4, ts_t4])
            #"""
            break
    
    res_pd = pd.DataFrame(RES)
    res_pd.columns = ['Model', 'Mode', 'RW iter', 'AUC', 
                      'Train time', 'Pred time']
    res_pd.to_csv("../results/PROMOS/results.csv")
    
    """
    mgedt.refit(gen=2)
    mgedt.predict(X_test)
    pred = mgedt.predict(X_test)
    
    auc = roc_auc_score(y_test, pred)*100
    print(auc)
    
    ROC = pd.DataFrame(roc_curve(y_test, pred, pos_label='Sale')).T
    ROC.columns = ["FPR", "TPR", "TH"]
    
    matplotlib.use('module://ipykernel.pylab.backend_inline', force=True)
    plt.plot(ROC['FPR'], ROC['TPR'], color='darkorange')
    plt.show()
    #"""
    """
    dtr = pd.read_csv("../datasets/Promos/BEST2/Train-IDF-2.csv", sep=";")
    dtest = pd.read_csv("../datasets/Promos/BEST2/Test-IDF-2.csv", sep=";")
    
    dtrain, dval = train_test_split(dtr, test_size=0.1, stratify=dtr['target'])
    X = dtrain.drop('target', axis=1)
    y = dtrain['target']
    X_val = dval.drop('target', axis=1)
    y_val = dval['target']
    X_test = dtest.drop('target', axis=1)
    y_test = dtest['target']
    #"""
    