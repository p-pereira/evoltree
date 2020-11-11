# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:39:16 2020

@author: pedro

Multi-objective graphics
"""
import csv
import pandas as pd
import numpy as np
from glob import glob
from itertools import product
import matplotlib.pyplot as plt
import math
from math import log, isnan
from sklearn import metrics
from pygmo import hypervolume
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
base = importr('base')
utils = importr('utils')
robjects.r('''
    # create a function `f`
    wilcoxon <- function(a, verbose=FALSE) {
        w = wilcox.test(a, conf.level = 0.95, conf.int = T, 
                        alternative = 'two.sided', correct = T)
        CI=as.numeric(w$conf.int)
        CM=as.numeric(w$estimate)
        c(CM, CI[1], CI[2])
    }
    ''')
wilcox = robjects.globalenv['wilcoxon']

C1 = "k"
C2 = "b"
C3 = "r"
C4 = "g"
C5 = "m"
RFM = "H"
A1 = 0.05
A2 = 0.5
WMS = 10
WLW = 0.5
TMS = 6
WMW = 1
ALPHA = 0.7
LW = 1

def myround(x, base=1):
    return base * round(x/base)

def aux_estimated(complexities):
    rComplexities = robjects.FloatVector(complexities)
    e, _, _ = wilcox(rComplexities)
    if isnan(e):
        return 0
    else:
        return e

def aux_lower(complexities):
    rComplexities = robjects.FloatVector(complexities)
    e, _, l = wilcox(rComplexities)
    if isnan(l):
        if isnan(e):
            return 0
        else:
            return e
    else:
        return l

def aux_upper(complexities):
    rComplexities = robjects.FloatVector(complexities)
    e, u, _ = wilcox(rComplexities)
    if isnan(u):
        if isnan(e):
            return 0
        else:
            return e
    else:
        return u

def MOGraph(text=True, opac=True, trad=True, wilcoxon=True):
    WIL=""
    TEX=""
    OPAC=""
    TRD=""
    if wilcoxon:
        WIL = "-wilcox"
    if text:
        TEX="-txt"
    if opac:
        OPAC="-opac"
    if trad:
        TRD="-trd"
    method = "MOGEDT"
    mode = "BEST2"
    
    # read the results
    res = pd.read_csv("../results/"+method+"-"+mode+"/all_results.csv",
                      delimiter=";")
    a = res.loc[:,('AUCS','COMPLEXITY')]
    a['COMPLEXITY'] = [log(x,10) for x in a['COMPLEXITY']]
    a['AUCS'] = [-1*myround(x) for x in a['AUCS']]
    if trad:
        res2 = pd.read_csv("../DT/"+mode+"/IDF/TEST_metrics-RW.csv",
                           delimiter=";")
        dtAUC = np.median(res2['AUC'])*-1
        dtComplexity = log(np.median(res2['COMPLEXITY']),10)
    
    grouped = a.groupby('AUCS')
    medians = grouped['COMPLEXITY'].median()

    estimated = []
    uppers = []
    lowers = []
    
    for key in medians.index:
        complexities = list(grouped.get_group(key)['COMPLEXITY'])
        rComplexities = robjects.FloatVector(complexities)
        e, u, l = wilcox(rComplexities)
        estimated.append(e)
        uppers.append(u)
        lowers.append(l)
    
    print(mode, method, "- ", np.min(a['AUCS']), " - ",np.max(a['AUCS']))
    print(mode, "DT", "- ", round(np.median(res2['AUC']*-1),2))
    
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5,4), 
                                   sharex=True, sharey=True)
    
    #plt.figure(figsize=[6,5])
    if opac:
        ax1.plot(a['AUCS'], a['COMPLEXITY'], color=C1, marker='s', 
                 markerfacecolor='None', ms=4, alpha=A1, 
                 linestyle="None", label="")
    if wilcoxon:
        ax1.plot(medians.index, uppers, color=C1, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
        ax1.plot(medians.index, lowers, color=C1, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
    
    if text:
        ax1.text(min(medians.index)-1, max(estimated)+0.1, 
                 "["+ str(round(min(medians.index),2)) + 
                 ", " + str(round(max(estimated), 2)) + "]",
                 style='italic', fontsize=6, color=C3)    
        
        ax1.text(max(medians.index)+0.007, min(estimated)+0.15,
                 "["+ str(round(max(medians.index),2)) + 
                 ", " + str(round(min(estimated), 2)) + "]",
                 style='italic', fontsize=6, color=C3)    
    
    if wilcoxon:
        for i in range(len(uppers)):
            if math.isnan(uppers[i]):
                ax1.plot(medians.index[i], estimated[i], color=C1, marker='_',
                         ms=10, linestyle='None', alpha=ALPHA)
                ax1.plot((medians.index[i], medians.index[i]), 
                         (estimated[i],estimated[i]), color=C1, linewidth=WLW,
                         alpha=ALPHA)
            else:
                ax1.plot((medians.index[i], medians.index[i]), 
                         (lowers[i], uppers[i]), color=C1, linewidth=WLW,
                         alpha=ALPHA)
    
    ax1.plot(medians.index, estimated, color=C1, marker='s', ms=3, 
             linestyle="--", label="MGEDT", linewidth=LW)
    
    method = "MOGEDTLM"
    
    # read the results
    res = pd.read_csv("../results/"+method+"-"+mode+"/all_results.csv",
                      delimiter=";")
    a = res.loc[:,('AUCS','COMPLEXITY')]
    a['COMPLEXITY'] = [log(x,10) for x in a['COMPLEXITY']]
    a['AUCS'] = [-1*myround(x) for x in a['AUCS']]
    
    
    grouped = a.groupby('AUCS')
    medians = grouped['COMPLEXITY'].median()
    
    estimated = []
    uppers = []
    lowers = []
    
    for key in medians.index:
        complexities = list(grouped.get_group(key)['COMPLEXITY'])
        rComplexities = robjects.FloatVector(complexities)
        e, u, l = wilcox(rComplexities)
        estimated.append(e)
        uppers.append(u)
        lowers.append(l)
    
    if opac:
        ax1.plot(a['AUCS'], a['COMPLEXITY'], color=C2, marker='s', 
                 markerfacecolor='None', ms=4, alpha=A2, linestyle="None", 
                 label="")
    if wilcoxon:
        ax1.plot(medians.index, uppers, color=C2, marker='_', ms=WMS, 
                 linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
        ax1.plot(medians.index, lowers, color=C2, marker='_', ms=WMS, 
                 linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
    
    
    if text:
        ax1.text(min(medians.index)-4, max(estimated)-0.25, 
                 "["+ str(round(min(medians.index),2)) + ", " + 
                 str(round(max(estimated), 2)) + "]", style='italic', 
                 fontsize=6, color=C2)    
        
        ax1.text(max(medians.index)-7, min(estimated)-0.2, 
                 "["+ str(round(max(medians.index),2)) + ", " + 
                 str(round(min(estimated), 2)) + "]", style='italic', 
                 fontsize=6, color=C2)    
    
    if wilcoxon:
        for i in range(len(uppers)):
            if math.isnan(uppers[i]):
                ax1.plot(medians.index[i], estimated[i], color=C2, 
                         marker='_', ms=10, linestyle='None', alpha=ALPHA)
                ax1.plot((medians.index[i], medians.index[i]), 
                         (estimated[i], estimated[i]), color=C2, linewidth=WLW,
                         alpha=ALPHA)
            else:
                ax1.plot((medians.index[i], medians.index[i]), 
                         (lowers[i], uppers[i]), color=C2, linewidth=WLW,
                         alpha=ALPHA)
    
    ax1.plot(medians.index, estimated, color=C2, marker='s', ms=3,
             linestyle="-", label="MGEDTL", linewidth=LW)
    
    if trad:
        dlAUC = np.median([-92.5, -87.1, -79.7, -79.1, -89.9, -91.4, -93.2])
        dlComp = log(np.median([963076, 957956, 950788, 945668, 
                                941572, 942596, 938500]), 10)
        
        rfAUC = np.median([-91.98, -87.5, -92.2, -91.9, -88.6, -91.2, -92.8])
        rfComp = log(np.median([25858974, 26550434, 22494846, 23797714, 
                                20793008, 23950305, 22084884]), 10)
        
        ax1.plot(dlAUC, dlComp, color=C3, marker='o', ms=TMS, 
                 linestyle='None', label="DL")
        
        ax1.plot(dtAUC, dtComplexity, color=C4, marker='D', ms=TMS, 
                 linestyle='None', label="DT")
        
        ax1.plot(rfAUC, rfComp, color=C5, marker=RFM, ms=TMS, 
                 linestyle='None', label="RF")
        if opac:
            dtOpAUC = [x*-1 for x in res2['AUC']]
            dtOpComp = [log(x,10) for x in res2['COMPLEXITY']]
            ax1.plot(dtOpAUC, dtOpComp, color=C1, marker='D', 
                     markerfacecolor='None', ms=4, alpha=A1, 
                     linestyle="None", label="")
        if text:
            ax1.text(dtAUC+1, dtComplexity-0.05, "["+ str(round(dtAUC,2)) + 
                     ", " + str(round(dtComplexity, 2)) + "]", style='italic', 
                     fontsize=6, color=C3)
    print(mode, method, "- ", np.min(a['AUCS']), " - ",np.max(a['AUCS']))
    print(mode, "DL", "- ", dlAUC)
    method = "MOGEDT"
    mode = "TEST2"
    
    # read the results
    res = pd.read_csv("../results/"+method+"-"+mode+"/all_results.csv",
                      delimiter=";")
    
    a = res.loc[:,('AUCS','COMPLEXITY')]
    a['COMPLEXITY'] = [log(x,10) for x in a['COMPLEXITY']]
    a['AUCS'] = [-1*myround(x) for x in a['AUCS']]
    if trad:
        res2 = pd.read_csv("../DT/"+mode+"/IDF/TEST_metrics-RW.csv",
                           delimiter=";")
        dtAUC = np.median(res2['AUC'])*-1
        dtComplexity = log(np.median(res2['COMPLEXITY']),10)
    
    grouped = a.groupby('AUCS')
    medians = grouped['COMPLEXITY'].median()
    
    print(mode, method, "- ", np.min(a['AUCS']), " - ",np.max(a['AUCS']))
    print(mode, "DT", "- ", round(dtAUC,2))
    
    estimated = []
    uppers = []
    lowers = []
    
    for key in medians.index:
        complexities = list(grouped.get_group(key)['COMPLEXITY'])
        rComplexities = robjects.FloatVector(complexities)
        e, u, l = wilcox(rComplexities)
        estimated.append(e)
        uppers.append(u)
        lowers.append(l)
    
    if opac:
        ax2.plot(a['AUCS'], a['COMPLEXITY'], color=C1, marker='s', 
                 markerfacecolor='None', ms=4, alpha=A1, 
                 linestyle="None", label="")
    if wilcoxon:
        ax2.plot(medians.index, uppers, color=C1, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
        ax2.plot(medians.index, lowers, color=C1, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
    
    if text:
        ax2.text(min(medians.index)-1, max(estimated)+0.1, 
                 "["+ str(round(min(medians.index),2)) + ", " + 
                 str(round(max(estimated), 2)) + "]", style='italic', 
                 fontsize=6, color=C3)    
        
        ax2.text(max(medians.index)-1, min(estimated)+0.15, 
                 "["+ str(round(max(medians.index),2)) + ", " + 
                 str(round(min(estimated), 2)) + "]", style='italic', 
                 fontsize=6, color=C3)    
    if wilcoxon:
        for i in range(len(uppers)):
            if math.isnan(uppers[i]):
                ax2.plot(medians.index[i], estimated[i], color=C1, 
                         marker='_', ms=10, linestyle='None', linewidth=WLW,
                         alpha=ALPHA)
                ax2.plot((medians.index[i], medians.index[i]), 
                         (estimated[i], estimated[i]), color=C1, alpha=ALPHA)
            else:
                ax2.plot((medians.index[i], medians.index[i]), 
                         (lowers[i], uppers[i]), color=C1, linewidth=WLW,
                         alpha=ALPHA)
    
    ax2.plot(medians.index, estimated, color=C1, marker='s', ms=3, 
             linestyle="--", label="MGEDT", linewidth=LW)
    
    method = "MOGEDTLM"
    
    # read the results
    res = pd.read_csv("../results/"+method+"-"+mode+"/all_results.csv",
                      delimiter=";")
    a = res.loc[:,('AUCS','COMPLEXITY')]
    a['COMPLEXITY'] = [log(x,10) for x in a['COMPLEXITY']]
    a['AUCS'] = [-1*myround(x) for x in a['AUCS']]
    
    grouped = a.groupby('AUCS')
    medians = grouped['COMPLEXITY'].median()
    
    estimated = []
    uppers = []
    lowers = []
    
    for key in medians.index:
        complexities = list(grouped.get_group(key)['COMPLEXITY'])
        rComplexities = robjects.FloatVector(complexities)
        e, u, l = wilcox(rComplexities)
        estimated.append(e)
        uppers.append(u)
        lowers.append(l)
        
    if opac:
        ax2.plot(a['AUCS'], a['COMPLEXITY'], color=C2, marker='o', 
                 markerfacecolor='None', ms=4, alpha=A2, 
                 linestyle="None", label="")
    if wilcoxon:
        ax2.plot(medians.index, uppers, color=C2, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
        ax2.plot(medians.index, lowers, color=C2, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)    
    
    if wilcoxon:
        for i in range(len(uppers)):
            if math.isnan(uppers[i]):
                ax2.plot(medians.index[i], estimated[i], color=C2, 
                         marker='_', ms=10, linestyle='None', alpha=ALPHA)
                
                ax2.plot((medians.index[i], medians.index[i]), 
                         (estimated[i], estimated[i]), color=C2, linewidth=WLW,
                         alpha=ALPHA)
            else:
                ax2.plot((medians.index[i], medians.index[i]), 
                         (lowers[i], uppers[i]), color=C2, linewidth=WLW,
                         alpha=ALPHA)
    
    ax2.plot(medians.index, estimated, color=C2, marker='o', ms=3, 
             linestyle="-", label="MGEDTL", linewidth=LW)
    
    if trad:
        dlAUC = np.median([-93.1, -91.7, -93, -91.9, -94.1, -94, -94.6])
        dlComp = log(np.median([1108484, 1098244, 1091076, 1086980, 
                                1082884, 1082884, 1083908]),10)
        
        rfAUC = np.median([-87.7, -87.6, -88.6, -88.3, -88.0, -88.2, -89.6])
        rfComp = log(np.median([20604206, 20447039, 20885944, 20447481, 
                                18979757, 20452716, 19252509]),10)
        
        ax2.plot(dlAUC, dlComp, color=C3, marker='o', ms=TMS, 
                 linestyle='None', label="DL")
        
        ax2.plot(dtAUC, dtComplexity, color=C4, marker='D', ms=TMS, 
                 linestyle='None', label="DT")
        
        ax2.plot(rfAUC, rfComp, color=C5, marker=RFM, ms=TMS, 
                 linestyle='None', label="RF")
        
        if opac:
            dtOpAUC = [x*-1 for x in res2['AUC']]
            dtOpComp = [log(x,10) for x in res2['COMPLEXITY']]
            ax2.plot(dtOpAUC, dtOpComp, color=C3, marker='D', 
                     markerfacecolor='None', ms=4, alpha=A1, 
                     linestyle="None", label="")
        if text:
            ax2.text(dtAUC+1, dtComplexity-0.05, "["+ str(round(dtAUC,2)) + 
                     ", " + str(round(dtComplexity, 2)) + "]", style='italic', 
                     fontsize=6, color=C3)
        
    if text:
        ax2.text(min(medians.index)-4, max(estimated)-0.27, 
                 "["+ str(round(min(medians.index),2)) + ", " + 
                 str(round(max(estimated), 2)) + "]", style='italic', 
                 fontsize=6, color=C2)    
        
        ax2.text(max(medians.index)-7, min(estimated)-0.2, 
                 "["+ str(round(max(medians.index),2)) + ", " + 
                 str(round(min(estimated), 2)) + "]", style='italic', 
                 fontsize=6, color=C2)    
    
    plt.legend(loc='upper center', frameon=False, framealpha=1, ncol=5, 
               fancybox=True, shadow=True, bbox_to_anchor=(0.01, 1.1))
    
    ax1.set_ylabel('Complexity', fontsize=10)
    plt.xlim([-100, -45])
    plt.ylim([0, 8])
    fig.text(0.52, 0.01, "-AUC", ha="center", fontsize=10)
    
    print(mode, method, "- ", np.min(a['AUCS']), " - ",np.max(a['AUCS']))
    print(mode, "DL", "- ", dlAUC)
    
    #fig.text(0.3, 0.89, "BEST", ha="left", fontsize=10)
    #fig.text(0.75, 0.89, "TEST", ha="left", fontsize=10)
    
    ax1.grid(color='black', linestyle='--', linewidth=0.2)
    ax2.grid(color='black', linestyle='--', linewidth=0.2)
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    plt.subplots_adjust(top = 0.92, bottom = 0.15, right = 0.99, left = 0.064,
                        hspace = 0, wspace = 0.02)
    plt.savefig("MO_compar"+WIL+OPAC+TRD+TEX+".pdf")
    plt.close()

#MOGraph(text=False, opac=False, wilcoxon=True)

def unfinishedAUCcompare():
    """
    MOGraph(text=True, opac=True, trad=True, wilcoxon=True)
    MOGraph(text=True, opac=False, trad=True, wilcoxon=True)
    MOGraph(text=False, opac=False, trad=True, wilcoxon=True)
    MOGraph(text=False, opac=False, trad=True, wilcoxon=False)
    MOGraph(text=False, opac=True, trad=True, wilcoxon=False)
    MOGraph(text=True, opac=False, trad=False, wilcoxon=False)
    """
    
    
    model = "MOGEDTLM"
    mode = "TEST2"
    plt.figure(0).clf()
    
    for i in range(1,8):
        if i == 1:
            gen = 100
        else:
            gen = 25
        d = pd.read_csv("../results/" + model + "-" + mode + "/iteration" + 
                        str(i) + "/" + str(gen) + "_front/0_predictions.csv",
                        delimiter=";")
        label = d['Y']
        pred = d['pred']
        
        fpr, tpr, thresh = metrics.roc_curve(label, pred, pos_label='Sale')
        auc = metrics.roc_auc_score(label, pred)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        print(auc)
        
        d2 = pd.read_csv("../DT/"+mode+"/IDF/iteration"+str(i)+"/preds.csv", delimiter=";")
        label = d2['Y']
        pred = d2['pred']
        
        fpr, tpr, thresh = metrics.roc_curve(label, pred, pos_label='Sale')
        auc = metrics.roc_auc_score(label, pred)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        print(auc)
    
    plt.show()

def readData():
    dBEST = pd.read_json('../datasets/ColetasPromosBestUpdated.JSON', 
                         lines=True) #read JSON!!!
    # convert date from string to datetime
    dBEST['date_added_utc'] = pd.to_datetime(dBEST['date_added_utc']) 
    
    dTEST = pd.read_json('../datasets/ColetasPromosTestUpdated.JSON', 
                         lines=True) #read JSON!!!
    # convert date from string to datetime
    dTEST['date_added_utc'] = pd.to_datetime(dTEST['date_added_utc']) 
    
    return dBEST, dTEST

def sale_vs_nosale():
    labels = ['No Sale', 'Sale']
    valBEST = [3.963843, 0.112532]
    valTEST = [2.283725, 0.022769]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=(7,4.5), sharex=True, sharey=True)
    
    rects1 = ax.bar(x - width/2, valBEST, width, label='BEST', color=C1, alpha=0.9)
    rects2 = ax.bar(x + width/2, valTEST, width, label='TEST', color=C2, alpha=0.9)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of records (million)')
    #ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.legend()
    plt.legend(loc='upper right', frameon=True, framealpha=1, ncol=2, fancybox=True, shadow=True)
    
    fig.tight_layout()
    
    
    plt.savefig("sale-vs-nosale.pdf")
    plt.close()

def compare_by_iter():
    
    all_results = pd.read_csv("../results/all_results.csv", sep=";")
    all_results['macF1'] = all_results['macF1'] * -1
    
    by_iter = all_results.groupby(['method','mode','rw_iter'])\
        .agg({'macF1': np.max}).reset_index()
    
    agg_mgedt = by_iter[by_iter['method']=='MGEDT']
    agg_mgedt_b = agg_mgedt[agg_mgedt['mode']=='BEST_new']
    agg_mgedt_t = agg_mgedt[agg_mgedt['mode']=='TEST_new']
    
    agg_mgedtl = by_iter[by_iter['method']=='MGEDTL']
    agg_mgedtl_b = agg_mgedtl[agg_mgedtl['mode']=='BEST_new']
    agg_mgedtl_t = agg_mgedtl[agg_mgedtl['mode']=='TEST_new']
    #"""
    agg_dt = by_iter[by_iter['method']=='DT']
    agg_dt_b = agg_dt[agg_dt['mode']=='BEST_new']
    agg_dt_t = agg_dt[agg_dt['mode']=='TEST_new']
    
    agg_rf = by_iter[by_iter['method']=='RF']
    agg_rf_b = agg_rf[agg_rf['mode']=='BEST_new']
    agg_rf_t = agg_rf[agg_rf['mode']=='TEST_new']
    
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3), sharex=True, 
                                   sharey=True)
    ax1.plot(agg_mgedt_b['rw_iter'], agg_mgedt_b['macF1'], color=C1, 
             marker='s', ms=3, linestyle="--", label="MGEDT")
    #"""
    ax1.plot(agg_mgedtl_b['rw_iter'], agg_mgedtl_b['macF1'], color=C2, 
             marker='o', ms=3, linestyle="-", label="MGEDTL")
    """
    ax1.plot(by_iter.index, DL_BEST, color=C3, marker='o', ms=3, 
             linestyle=":", label="DL")
    #"""
    ax1.plot(agg_dt_b['rw_iter'], agg_dt_b['macF1'], color=C4, 
             marker='s', ms=3, linestyle="-.", label="DT")
    
    ax1.plot(agg_rf_b['rw_iter'], agg_rf_b['macF1'], color=C5, 
             marker=RFM, ms=3, linestyle="-.", label="RF")
    
    ax2.plot(agg_mgedt_t['rw_iter'], agg_mgedt_t['macF1'], color=C1, 
             marker='s', ms=3, linestyle="--", label="MGEDT")
    #"""
    ax2.plot(agg_mgedtl_t['rw_iter'], agg_mgedtl_t['macF1'], color=C2, 
             marker='o', ms=3, linestyle="-", label="MGEDTL")
    """
    ax2.plot(agg_mgedtl_t['rw_iter'], DL_TEST, color=C3, marker='o', ms=3, 
             linestyle=":", label="DL")
    #"""
    ax2.plot(agg_dt_t['rw_iter'], agg_dt_t['macF1'], color=C4, 
             marker='s', ms=3, linestyle="-.", label="DT")
    
    ax2.plot(agg_rf_t['rw_iter'], agg_rf_t['macF1'], color=C5, 
             marker=RFM, ms=3, linestyle="-.", label="RF")
    
    ax1.legend(loc='upper center', frameon=False, framealpha=0.4, ncol=5, 
               fancybox=True, shadow=False, bbox_to_anchor=(1, 1.12))
    
    ax1.set_ylabel('macF1', fontsize=10)
    #plt.ylim([70, 100])
    fig.text(0.52, 0.01, "RW Iteration", ha="center", fontsize=10)
    
    #fig.text(0.3, 0.89, "BEST", ha="left", fontsize=10)
    #fig.text(0.75, 0.89, "TEST", ha="left", fontsize=10)
    
    ax1.grid(color='black', linestyle='--', linewidth=0.2)
    ax2.grid(color='black', linestyle='--', linewidth=0.2)
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    plt.subplots_adjust(top = 0.93, bottom = 0.13, right = 0.99, left = 0.075, 
                        hspace = 0, wspace = 0.01)
    plt.savefig("macF1.pdf")
    plt.close()

compare_by_iter()
#from pygmo import hypervolume

def hypervol():
    x = -50
    y = 5
    
    hMB = []
    hMLB = []
    hMT = []
    hMLT = []
    
    for i in range(1,8):
        method = "MOGEDT"
        mode = "BEST2"
        # read the results
        res = pd.read_csv("../results/"+method+"-"+mode+"/all_results.csv",
                          delimiter=";")
        res = res.loc[res['ITER'] == i]
        a = res.loc[:,('AUCS','COMPLEXITY')]
        a['COMPLEXITY'] = [log(x,10) for x in a['COMPLEXITY']]
        a['AUCS'] = [-1*x for x in a['AUCS']]
    
        h = hypervolume(a.to_numpy())
        hMB.append(h.compute([x,y]))
        
        mode = "TEST2"
        # read the results
        res = pd.read_csv("../results/"+method+"-"+mode+"/all_results.csv",
                          delimiter=";")
        res = res.loc[res['ITER'] == i]
        a = res.loc[:,('AUCS','COMPLEXITY')]
        a['COMPLEXITY'] = [log(x,10) for x in a['COMPLEXITY']]
        a['AUCS'] = [-1*x for x in a['AUCS']]
    
        h = hypervolume(a.to_numpy())
        hMT.append(h.compute([x,y]))
        
        method = "MOGEDTLM"
        mode = "BEST2"
        # read the results
        res = pd.read_csv("../results/"+method+"-"+mode+"/all_results.csv",
                          delimiter=";")
        res = res.loc[res['ITER'] == i]
        a = res.loc[:,('AUCS','COMPLEXITY')]
        a['COMPLEXITY'] = [log(x,10) for x in a['COMPLEXITY']]
        a['AUCS'] = [-1*x for x in a['AUCS']]
    
        h = hypervolume(a.to_numpy())
        hMLB.append(h.compute([x,y]))
        
        mode = "TEST2"
        # read the results
        res = pd.read_csv("../results/"+method+"-"+mode+"/all_results.csv",
                          delimiter=";")
        res = res.loc[res['ITER'] == i]
        a = res.loc[:,('AUCS','COMPLEXITY')]
        a['COMPLEXITY'] = [log(x,10) for x in a['COMPLEXITY']]
        a['AUCS'] = [-1*x for x in a['AUCS']]
    
        h = hypervolume(a.to_numpy())
        hMLT.append(h.compute([x,y]))
    
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3), 
                                   sharex=True, sharey=True)
    
    ax1.plot(range(1, len(hMB)+1), hMB, color=C1, marker='s', ms=3,
             linestyle="-", label="", linewidth=LW)
    
    ax1.plot(range(1, len(hMB)+1), hMLB, color=C2, marker='o', ms=3,
             linestyle="--", label="", linewidth=LW)
    
    ax2.plot(range(1, len(hMB)+1), hMT, color=C1, marker='s', ms=3,
             linestyle="-", label="MGEDT", linewidth=LW)
    
    ax2.plot(range(1, len(hMB)+1), hMLT, color=C2, marker='o', ms=3,
             linestyle="--", label="MGEDTL", linewidth=LW)
    ax1.grid(color='black', linestyle='--', linewidth=0.2)
    ax2.grid(color='black', linestyle='--', linewidth=0.2)
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    fig.text(0.52, 0.01, "RW iteration", ha="center", fontsize=10)
    ax1.set_ylabel('Hypervolume', fontsize=10)
    
    plt.legend(loc='upper center', frameon=False, framealpha=1, ncol=4, 
                   fancybox=True, shadow=True, bbox_to_anchor=(0.01, 1.1))
    
    plt.subplots_adjust(top = 0.92, bottom = 0.15, right = 0.99, left = 0.08,
                        hspace = 0, wspace = 0.02)
    plt.savefig("Hypervol.pdf")
    plt.close()

'''
## Get training time and append with remaining info

for method in methods:
    for mode in modes:
        all_times = []
        for iteration in range(1,8):
            file = ("../results/"+method+"-"+mode+"/iteration"+
                    str(iteration)+"/stats.tsv")
            time = max(pd.read_csv(file, sep="\t")['total_time'])
            all_times.append(time)
        file2 = ("../results/"+method+"-"+mode+"/results.csv")
        res = pd.read_csv(file2, sep=";")
        res['TRAIN_TIME'] = all_times
        res.to_csv(file2, sep=";")


methods = ['MOGEDT', 'MOGEDTLM']
modes = ['BEST2','TEST2']

### Training and testing times
info = pd.DataFrame({'Model':[], 'Mode':[], 'ITER': [], 'Train':[],
                     'Pred':[], 'Max_Complexity' : []})
for method in methods:
    for mode in modes:
        file1 = ("../results/"+method+"-"+mode+"/all_results.csv")
        file2 = ("../results/"+method+"-"+mode+"/results.csv")
        
        res1 = pd.read_csv(file1, sep=";")
        res2 = pd.read_csv(file2, sep=";")
        
        by_iter = res1.groupby('ITER').agg(
            mean_time = pd.NamedAgg(column='TIME', aggfunc=np.mean),
            max_complexity = pd.NamedAgg(column='COMPLEXITY', aggfunc=max)
        )
        
        mes = np.repeat(method, len(by_iter))
        mos = np.repeat(mode, len(by_iter))
        
        d = pd.DataFrame({'Model' : mos,
                          'Mode' : mes, 
                          'ITER' : [ i for i in range(1,8)], 
                          'Train' : res2['TRAIN_TIME'], 
                          'Pred': by_iter['mean_time'].to_numpy()* 1000, #microsseconds
                          'Max_Complexity' : by_iter['max_complexity'].to_numpy()})
        info = pd.concat([info, d], ignore_index=True)
info.to_csv("info.csv", sep=";")


info = pd.read_csv("info.csv", sep=";")
byModel = info.groupby(['Model','Mode']).agg(
    train_time = pd.NamedAgg(column='Train', aggfunc=np.mean),
    pred_time = pd.NamedAgg(column='Pred', aggfunc=np.mean) 
    )

DLBT = [40.221, 41.955, 47.526, 53.069, 51.123, 51.575, 51.855]
DLBP = [0.042, 0.042, 0.043, 0.043, 0.045, 0.046, 0.047]
DLTT = [25.519, 29.810, 33.961, 37.642, 38.013, 38.304, 39.784]
DLTP = [0.053, 0.053, 0.055, 0.057, 0.058, 0.059, 0.057]

DTBT = [7.1, 8.05, 8.8, 9.93, 9.77, 9.6, 9.59]
DTBP = [0.00025, 0.00025, 0.00025, 0.00026000000000000003, 0.00023, 0.00022, 0.00023]
DTTT = [2.87, 3.42, 4.1, 4.65, 5.05, 4.6, 4.67]
DTTP = [0.00026000000000000003, 0.00022, 0.00024, 0.00022, 0.00024, 0.00024, 0.00023]


DNMB = [5, 4, 3, 3, 5, 3, 3]
DNMT = [1, 1, 2, 1, 1, 1, 1]

DNMLB = [328, 318, 692, 368, 364, 367, 769]
DNMLT = [864, 519, 621, 883, 1003, 598, 941]

DNDTB = [10487, 10611, 10666, 10303, 10169, 10087, 9999]
DNDTT = [8384, 8593, 8617, 8504, 8413, 8406, 8276]

'''

def seed_no_seed():
    iters=25
    AUCs_no = []
    mean_AUCs_no = []
    folder_no = "../results/review_new/no_seed/MGEDT-BEST2/iteration3/"
    AUCs_seed = []
    mean_AUCs_seed = []
    folder_seed = "../results/review_new/seed/MGEDT-BEST2/iteration3/"
    
    for it in range(iters+1):
        filename_no = "{0}/{1}_front/{2}".format(folder_no, str(it),
                                                 "pareto.csv")
        aucs_no = pd.read_csv(filename_no, sep=";")['m1']
        
        AUCs_no.append(min(aucs_no))
        mean_AUCs_no.append(np.mean(aucs_no))
        
        filename_seed = "{0}/{1}_front/{2}".format(folder_seed, str(it),
                                                 "pareto.csv")
        aucs_seed = pd.read_csv(filename_seed, sep=";")['m1']
        
        AUCs_seed.append(min(aucs_seed))
        mean_AUCs_seed.append(np.mean(aucs_seed))
    
    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(1, 1, figsize=(5,3))
    ax1.plot(range(iters+1), AUCs_no, color="steelblue", marker='o', ms=3,
             linestyle=":", label="No seed", linewidth=1)
    ax1.plot(range(iters+1), AUCs_seed, color="seagreen", marker='s', ms=3,
             linestyle="-.", label="Seeding", linewidth=1)
    
    ax1.set_ylabel('-AUC', fontsize=10)
    ax1.set_xlabel('Generations', fontsize=10)
    
    ax1.grid(color='black', linestyle='--', linewidth=0.2)
    ax1.set_facecolor('white')
    
    plt.legend(loc='upper center', frameon=False, framealpha=0.4, ncol=2, 
               fancybox=True, shadow=False, bbox_to_anchor=(0.5, 1.12))
    plt.subplots_adjust(top = 0.93, bottom = 0.13, right = 0.99, left = 0.13, 
                        hspace = 0, wspace = 0.01)
    plt.savefig("Seed_vs_noSeed.pdf")
    plt.close()
    
    fig, ax1 = plt.subplots(1, 1, figsize=(5,3))
    ax1.plot(range(iters+1), mean_AUCs_no, color="steelblue", marker='o', ms=3,
             linestyle=":", label="No seed", linewidth=1)
    ax1.plot(range(iters+1), mean_AUCs_seed, color="seagreen", marker='s', ms=3,
             linestyle="-.", label="Seeding", linewidth=1)
    
    ax1.set_ylabel('-AUC', fontsize=10)
    ax1.set_xlabel('Generations', fontsize=10)
    
    ax1.grid(color='black', linestyle='--', linewidth=0.2)
    ax1.set_facecolor('white')
    
    plt.legend(loc='upper center', frameon=False, framealpha=0.4, ncol=2, 
               fancybox=True, shadow=False, bbox_to_anchor=(0.5, 1.12))
    plt.subplots_adjust(top = 0.93, bottom = 0.13, right = 0.99, left = 0.13, 
                        hspace = 0, wspace = 0.01)
    plt.savefig("review/Seed_vs_noSeed_mean.pdf")
    plt.close()
    
#seed_no_seed()

def convertSolutions():
    from tqdm import tqdm
    METHODS = ["MGEDT", "MGEDTL"]
    MODES = ["BEST_new", "TEST_new"]
    RW_ITERS = 7
    GENS1 = 100
    GENS2 = 25
    
    all_ = [METHODS, MODES]
    all_comb = list(product(*all_))
    FOLDER = "../results"
    
    all_info = pd.DataFrame({"method" : [],
                             "mode" : [],
                             "rw_iter" : [],
                             #"phenotype" : [],
                             "macF1" : [],
                             "complexity" : [],
                             "decision_nodes" : [],
                             "leaf_nodes" : [],
                             "total_nodes" : []})
    for iteration in tqdm(range(1, RW_ITERS+1)):
        test_data = pd.read_csv("../datasets/Promos/TEST_new/Test-IDF-{0}.csv".format(str(iteration)),
                                sep=";")
        x_test = test_data.drop('ValueSale', axis=1)
        y_test = test_data['ValueSale']
        
        best_data = pd.read_csv("../datasets/Promos/BEST_new/Test-IDF-{0}.csv".format(str(iteration)),
                                sep=";")
        x_best = best_data.drop('ValueSale', axis=1)
        y_best = best_data['ValueSale']
        for (method, mode) in all_comb:
            if method=="MGEDTL" and mode=="TEST_new":
                pass
            gen = GENS2
            if iteration == 1:
                gen = GENS1
            fdir = "{0}/{1}/{2}/iteration{3}/{4}_front/".format(FOLDER, 
                                                                method,
                                                                mode, 
                                                                str(iteration), 
                                                                str(gen))
            all_files = glob(fdir+"*.txt")
            
            if mode == "BEST_new":
                x, y = x_best, y_best
            else:
                x, y = x_test, y_test
                
            for filepath in all_files:
                
                #filename = filepath.split("\\")[1]
                raw_content = open(filepath, 'r').read()
                content = raw_content.split("\n")
                
                tree_idx = content.index("Tree:") + 1
                tree = content[tree_idx]
                
                decision_nodes = tree.count("np.where")
                leaf_nodes = tree.count("<leaf>")
                nodes = decision_nodes + leaf_nodes
                fit_idx = content.index("Fitness:") + 1
                fitness = content[fit_idx]
                
                phen_idx = content.index("Phenotype:") + 1
                phenotype = content[phen_idx]
                
                pred = eval(phenotype)
                pred = np.nan_to_num(pred)
                #print(type(pred))
                if type(pred) == np.float64 or type(pred) == np.int32:
                    pred = np.repeat(pred, len(y))
                auc = - metrics.f1_score(y, pred, average="macro")*100
                
                complexity = int(fitness.split(",")[1][:-1])
                
                row = {"method" : method,
                       "mode" : mode,
                       "rw_iter" : iteration,
                       #"phenotype" : phenotype,
                       "macF1" : auc,
                       "complexity" : complexity,
                       "decision_nodes" : decision_nodes,
                       "leaf_nodes" : leaf_nodes,
                       "total_nodes" : nodes}
                all_info = all_info.append(row, ignore_index=True)
    
    
    
    trad_file = "{0}/{1}".format(FOLDER, "DT_results.csv")
    trad = pd.read_csv(trad_file, sep=";")
    #trad['ind'] = np.repeat('0', trad.shape[0])
    
    trad_file2 = "{0}/{1}".format(FOLDER, "RF_results.csv")
    trad2 = pd.read_csv(trad_file2, sep=";")
    #trad2['ind'] = np.repeat('0', trad.shape[0])
    
    all_info2 = pd.concat([all_info, trad, trad2], ignore_index=True)
    all_info2.to_csv(FOLDER+"/all_results.csv", sep=";", index=None)

#convertSolutions()

def MOGraph2(wilcoxon=True):
    #wilcoxon=True
    all_results = pd.read_csv("../results/all_results.csv", sep=";")
    
    cols = ['method', 'mode', 'macF1', 'median', 'upper', 'lower']
    
    mgedt = all_results.loc[all_results["method"]=="MGEDT"]
    mgedt['macF1'] = myround(mgedt['macF1'])
    mgedt['total_nodes'] = [log(x, 10) for x in mgedt['total_nodes']]
    agg_mgedt = mgedt.groupby(["method", 
                               "mode", 
                               "macF1"]).agg({'total_nodes': [aux_estimated, 
                                                            aux_upper, 
                                                            aux_lower]}).reset_index()
    agg_mgedt.columns = cols
    agg_mgedt_b = agg_mgedt[agg_mgedt['mode']=="BEST_new"].reset_index()
    agg_mgedt_t = agg_mgedt[agg_mgedt['mode']=="TEST_new"].reset_index()
    #"""
    mgedtl = all_results.loc[all_results["method"]=="MGEDTL"]
    mgedtl['macF1'] = myround(mgedtl['macF1'])
    mgedtl['total_nodes'] = [log(x, 10) for x in mgedtl['total_nodes']]
    agg_mgedtl = mgedtl.groupby(["method", 
                                 "mode", 
                                 "macF1"]).agg({'total_nodes': [aux_estimated, 
                                                              aux_upper, 
                                                              aux_lower]}).reset_index()
    agg_mgedtl.columns = cols
    agg_mgedtl_b = agg_mgedtl[agg_mgedtl['mode']=="BEST_new"].reset_index()
    agg_mgedtl_t = agg_mgedtl[agg_mgedtl['mode']=="TEST_new"].reset_index()
    #"""
    dt = all_results.loc[all_results["method"]=="DT"]
    dt['total_nodes'] = [log(x, 10) for x in dt['total_nodes']]
    agg_dt = dt.groupby(["method", "mode"]).agg({'macF1' : np.median,
                                                 'total_nodes' : np.median}).reset_index()
    agg_dt_b = agg_dt[agg_dt['mode']=="BEST_new"].reset_index()
    agg_dt_t = agg_dt[agg_dt['mode']=="TEST_new"].reset_index()
    
    rf = all_results.loc[all_results["method"]=="RF"]
    rf['total_nodes'] = [log(x, 10) for x in rf['total_nodes']]
    agg_rf = rf.groupby(["method", "mode"]).agg({'macF1' : np.median,
                                                 'total_nodes' : np.median}).reset_index()
    agg_rf_b = agg_rf[agg_dt['mode']=="BEST_new"].reset_index()
    agg_rf_t = agg_rf[agg_dt['mode']=="TEST_new"].reset_index()
    
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5,4), 
                                   sharex=True, sharey=True)
    
    #plt.figure(figsize=[6,5])
    if wilcoxon:
        ax1.plot(agg_mgedt_b['macF1'], agg_mgedt_b['upper'], color=C1, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
        ax1.plot(agg_mgedt_b['macF1'], agg_mgedt_b['lower'], color=C1, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
    if wilcoxon:
        for i in range(len(agg_mgedt_b['lower'])):
            ax1.plot((agg_mgedt_b['macF1'][i], agg_mgedt_b['macF1'][i]), 
                         (agg_mgedt_b['lower'][i], agg_mgedt_b['upper'][i]), 
                         color=C1, linewidth=WLW, alpha=ALPHA)
    
    ax1.plot(agg_mgedt_b['macF1'], agg_mgedt_b['median'], color=C1, marker='s', 
             ms=3,  linestyle="--", label="MGEDT", linewidth=LW)
    #"""
    if wilcoxon:
        ax1.plot(agg_mgedtl_b['macF1'], agg_mgedtl_b['upper'], color=C2, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
        ax1.plot(agg_mgedtl_b['macF1'], agg_mgedtl_b['lower'], color=C2, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
    if wilcoxon:
        for i in range(len(agg_mgedtl_b['upper'])):
            ax1.plot((agg_mgedtl_b['macF1'][i], agg_mgedtl_b['macF1'][i]), 
                         (agg_mgedtl_b['lower'][i], agg_mgedtl_b['upper'][i]), 
                         color=C2, linewidth=WLW, alpha=ALPHA)
    
    ax1.plot(agg_mgedtl_b['macF1'], agg_mgedtl_b['median'], color=C2, marker='o', 
             ms=3, linestyle="-", label="MGEDTL", linewidth=LW)
    #"""
    ax1.plot(agg_dt_b['macF1'], agg_dt_b['total_nodes'], color=C4, marker='D', 
             ms=TMS, label="DT", linestyle='None')
    
    ax1.plot(agg_rf_b['macF1'], agg_rf_b['total_nodes'], color=C5, marker=RFM, 
             ms=TMS, linestyle='None', label="RF")
    
    if wilcoxon:
        ax2.plot(agg_mgedt_t['macF1'], agg_mgedt_t['upper'], color=C1, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
        ax2.plot(agg_mgedt_t['macF1'], agg_mgedt_t['lower'], color=C1, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
    if wilcoxon:
        for i in range(len(agg_mgedt_t['lower'])):
            ax2.plot((agg_mgedt_t['macF1'][i], agg_mgedt_t['macF1'][i]), 
                         (agg_mgedt_t['lower'][i], agg_mgedt_t['upper'][i]), 
                         color=C1, linewidth=WLW, alpha=ALPHA)
    
    ax2.plot(agg_mgedt_t['macF1'], agg_mgedt_t['median'], color=C1, marker='s', 
             ms=3, linestyle="--", label="MGEDT", linewidth=LW)
    #"""
    if wilcoxon:
        ax2.plot(agg_mgedtl_t['macF1'], agg_mgedtl_t['upper'], color=C2, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
        ax2.plot(agg_mgedtl_t['macF1'], agg_mgedtl_t['lower'], color=C2, marker='_', 
                 ms=WMS, linestyle='None', markeredgewidth=WMW, alpha=ALPHA)
    if wilcoxon:
        for i in range(len(agg_mgedtl_t['upper'])):
            ax2.plot((agg_mgedtl_t['macF1'][i], agg_mgedtl_t['macF1'][i]), 
                         (agg_mgedtl_t['upper'][i], agg_mgedtl_t['lower'][i]), 
                         color=C2, linewidth=WLW, alpha=ALPHA)
    
    ax2.plot(agg_mgedtl_t['macF1'], agg_mgedtl_t['median'], color=C2, marker='o', 
             ms=3, linestyle="-", label="MGEDTL", linewidth=LW)
    #"""
    ax2.plot(agg_dt_t['macF1'], agg_dt_t['total_nodes'], color=C4, marker='D', 
             ms=TMS, label="DT", linestyle='None')
    
    ax2.plot(agg_rf_t['macF1'], agg_rf_t['total_nodes'], color=C5, marker=RFM, 
             ms=TMS, linestyle='None', label="RF")
    
    ax1.set_ylabel('Complexity (nodes)', fontsize=10)
    plt.xlim([-22, 0])
    #plt.ylim([0, 8])
    fig.text(0.52, 0.01, "- macro average F1-score", ha="center", fontsize=10)
    
    
    #fig.text(0.3, 0.89, "BEST", ha="left", fontsize=10)
    #fig.text(0.75, 0.89, "TEST", ha="left", fontsize=10)
    
    ax1.grid(color='black', linestyle='--', linewidth=0.2)
    ax2.grid(color='black', linestyle='--', linewidth=0.2)
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    #major_ticks = np.arange(-25, -10, 2)
    #minor_ticks = np.arange(-25, -10, 2)
    
    #ax1.set_xticks(major_ticks)
    #ax1.set_xticks(minor_ticks, minor=True)
    
    ax1.legend(loc='upper center', frameon=False, framealpha=1, ncol=4, 
               fancybox=True, shadow=True, bbox_to_anchor=(1, 1.1))
    plt.subplots_adjust(top = 0.92, bottom = 0.15, right = 0.98, left = 0.064,
                        hspace = 0, wspace = 0.1)
    plt.savefig("nodes_new.pdf")
    plt.close()

#MOGraph2(wilcoxon=True)


def seed_no_seed2():
    iters=25
    rw_iters = 7
    #AUCs_no = []
    #mean_AUCs_no = []
    
    #AUCs_seed = []
    #mean_AUCs_seed = []
    
    curves_info = pd.DataFrame({"rw_iter" : [],
                                "generation" : [],
                                "auc_no" : [],
                                "auc_seed" : []})
    for rw in range(2, rw_iters+1):
        folder_no = "../results/review_new/no_seed/MGEDT-BEST2/iteration{0}/"\
            .format(str(rw))
        
        folder_seed = "../results/review_new/seed/MGEDT-BEST2/iteration{0}/"\
            .format(str(rw))
    
        for it in range(iters+1):
            filename_no = "{0}/{1}_front/{2}".format(folder_no, str(it),
                                                     "pareto.csv")
            aucs_no = pd.read_csv(filename_no, sep=";")['m1']
            
            #AUCs_no.append(min(aucs_no))
            #mean_AUCs_no.append(np.mean(aucs_no))
            auc_no = min(aucs_no)
            
            filename_seed = "{0}/{1}_front/{2}".format(folder_seed, str(it),
                                                     "pareto.csv")
            aucs_seed = pd.read_csv(filename_seed, sep=";")['m1']
            
            #AUCs_seed.append(min(aucs_seed))
            #mean_AUCs_seed.append(np.mean(aucs_seed))
            auc_seed = min(aucs_seed)
            row = {"rw_iter" : rw, "generation" : it, 
                   "auc_no" : auc_no, "auc_seed" : auc_seed}
            curves_info = curves_info.append(row, ignore_index=True)
    
    data = curves_info.groupby("generation")\
        .agg({"auc_no":[np.mean, np.median],
              "auc_seed":[np.mean, np.median]}).reset_index()
    mean_no = data["auc_no", "mean"]
    meadian_no = data["auc_no", "median"]
    
    mean_seed = data["auc_seed", "mean"]
    meadian_seed = data["auc_seed", "median"]
    
    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(1, 1, figsize=(5,3))
    ax1.plot(range(iters+1), mean_no, color="steelblue", marker='o', ms=3,
             linestyle=":", label="No seed", linewidth=1)
    ax1.plot(range(iters+1), mean_seed, color="seagreen", marker='s', ms=3,
             linestyle="-.", label="Seeding", linewidth=1)
    
    ax1.set_ylabel('-AUC', fontsize=10)
    ax1.set_xlabel('Generations', fontsize=10)
    
    ax1.grid(color='black', linestyle='--', linewidth=0.2)
    ax1.set_facecolor('white')
    
    plt.legend(loc='upper center', frameon=False, framealpha=0.4, ncol=2, 
               fancybox=True, shadow=False, bbox_to_anchor=(0.5, 1.12))
    plt.subplots_adjust(top = 0.93, bottom = 0.13, right = 0.99, left = 0.13, 
                        hspace = 0, wspace = 0.01)
    plt.savefig("review/all_seed_noSeed_mean_BEST.pdf")
    plt.close()
    
    fig, ax1 = plt.subplots(1, 1, figsize=(5,3))
    ax1.plot(range(iters+1), meadian_no, color="steelblue", marker='o', ms=3,
             linestyle=":", label="No seed", linewidth=1)
    ax1.plot(range(iters+1), meadian_seed, color="seagreen", marker='s', ms=3,
             linestyle="-.", label="Seeding", linewidth=1)
    
    ax1.set_ylabel('-AUC', fontsize=10)
    ax1.set_xlabel('Generations', fontsize=10)
    
    ax1.grid(color='black', linestyle='--', linewidth=0.2)
    ax1.set_facecolor('white')
    
    plt.legend(loc='upper center', frameon=False, framealpha=0.4, ncol=2, 
               fancybox=True, shadow=False, bbox_to_anchor=(0.5, 1.12))
    plt.subplots_adjust(top = 0.93, bottom = 0.13, right = 0.99, left = 0.13, 
                        hspace = 0, wspace = 0.01)
    plt.savefig("review/all_seed_noSeed_median_BEST.pdf")
    plt.close()

#seed_no_seed2()

def evolution():
    method = "MGEDT"
    mode = "BEST_new"
    res_f = "../results"
    gen1 = 100
    gen2 = 25
    rw_iter = 7
    
    TRAIN_F1 = []
    TEST_F1 = []
    BEST_TEST_F1 = []
    RW_GENS = []
    cont=0
    for i in range(1, rw_iter+1):
        if i == 1:
            gen=gen1
        else:
            gen=gen2
        
        cont += gen+1
        RW_GENS.append(cont)
        for g in range(gen+1):
            folder1 = "{0}/{1}/{2}/iteration{3}/".format(res_f, method, 
                                                           mode, str(i))
            tr_file = "{0}{1}_front/0.txt".format(folder1, str(g))
            ts_file = "{0}{1}_front-testFitness.txt".format(folder1, str(g))
            
            with open(tr_file, 'r') as f:
                content = f.read()
            tr_f1 = float(content.split('Fitness:\n[')[1].split(',')[0])
            TRAIN_F1.append(tr_f1*-1)
            
            vals = []
            with open(ts_file, 'r') as f2:
                reader = csv.reader(f2)
                for row in reader:
                    vals.append(float(row[0][1:]))
            
            TEST_F1.append(vals[0]*-1)
            BEST_TEST_F1.append(np.min(vals)*-1)
            
    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(1, 1, figsize=(5,3))
    ax1.plot(TRAIN_F1, color="steelblue", marker='o', ms=1,
             linestyle=":", label="BTrain (train)", linewidth=1)
    ax1.plot(TEST_F1, color="seagreen", marker='s', ms=1,
             linestyle="-.", label="BTrain (test)", linewidth=1)
    ax1.plot(BEST_TEST_F1, color=C1, marker='s', ms=1,  
             linestyle="--", label="BTest", linewidth=1)
    
    ax1.vlines(RW_GENS[:-1], 10, 25, colors='gray', linestyles='--', label='',
               linewidth=1)
    
    ax1.set_ylabel('macF1', fontsize=10)
    ax1.set_xlabel('Generations', fontsize=10)
    
    ax1.grid(color='black', linestyle='--', linewidth=0.2)
    ax1.set_facecolor('white')
    
    plt.legend(loc='upper center', frameon=False, framealpha=0.4, ncol=3, 
               fancybox=True, shadow=False, bbox_to_anchor=(0.5, 1.12))
    plt.subplots_adjust(top = 0.93, bottom = 0.13, right = 0.99, left = 0.13, 
                        hspace = 0, wspace = 0.01)
    plt.savefig("evolution_mgedt.pdf")
    plt.close()

#evolution()