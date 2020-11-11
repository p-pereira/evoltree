# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:29:16 2019

@author: pedro
"""
from representation import grammar
from algorithm.parameters import params
import numpy as np
from utilities.stats.trackers import runtime_error_cache
from utilities.fitness.error_metric import auc_metric
from scipy.optimize import minimize
from utilities.fitness.math_functions import sigmoid, inverse, symmetric, reLu, leakyReLu, sqrt
from random import sample
from utilities.fitness.get_data import get_data
from scipy.optimize import leastsq

def lamarck(ind, results, pool):
    """
    ....
    ....
    :param ind: population of individuals to be changed.
    :returns: The changed population.
    """

    if params['MULTICORE']:
        if params['MODEL'] == 'DT':
            # Add the individual to the pool of jobs.
            results.append(pool.apply_async(ind.applyLamarck, ()))
        else:
            results.append(pool.apply_async(ind.applyLamarckSYMB, ()))
        return results
    else:
        try:
            if params['MODEL'] == 'DT':
                ind2 = ind.applyLamarck()
            else:
                ind2 = ind.applyLamarckSYMB()
            results.append(ind2)
        except Exception as e:
            print(e)
            results.append(ind)
    
    return results

def lamarck_pop(pop):
    results, pool = [], None
    
    if params['MULTICORE']:
        pool = params['POOL']

    for name, ind in enumerate(pop):
        ind.name = name
        results = lamarck(ind, results, pool)

    new_pop = []
    if params['MULTICORE']:
        for result in results:
            # Execute all jobs in the pool.
            ind = result.get()

            # Set the fitness of the evaluated individual by placing the
            # evaluated individual back into the population.
            #pop[ind.name] = ind
            new_pop.append(ind)

            # Add the evaluated individual to the cache.
            #cache[ind.phenotype] = ind.fitness
        
            # Check if individual had a runtime error.
            if ind.runtime_error:
                runtime_error_cache.append(ind.phenotype)
        return new_pop
    else:
        return results
    
# save tree rules
def tree_to_code(tree, feature_names):
    #print(feature_names)
    """
    Converts a traditional decision tree to the used grammar.
    
    :param tree: The traditional decision tree.
    :param feature_names: The data attributes used for training.
    :return: The tree rules in the grammar's format.
    """
    from sklearn.tree import _tree
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #f = open(("./dt-rules.txt"), "w+")
    rules = ""
    
    def recurse(node, depth, rules):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            rules = rules + "np.where(x['{}'] <= {}, ".format(name, round(threshold, 6))
            rules = recurse(tree_.children_left[node], depth + 1, rules)
            rules = rules + ", "
            rules = recurse(tree_.children_right[node], depth + 1, rules)
            rules = rules + ")"
            return rules
        else:
            prob =  round(1 - tree_.value[node][0][0] / (tree_.value[node][0][0] + tree_.value[node][0][1]), 3)
            rules = rules + "({})".format(prob)
            return rules
    rules = recurse(0, 1, rules)
    return rules

# save tree rules
def tree_to_file(tree, feature_names):
    from sklearn.tree import _tree
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    f = open((params['FILE_PATH'] + "\TEST-dt-rules.txt"), "w", encoding='utf-8')
    #f.write("def tree({}):".format(", ".join(feature_names)))
    #print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        #indent = "  " * depth
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            f.write("np.where(x['{}'] <= {}, ".format(name, round(threshold, 6)))
            #f.write("\n{}if {} <= {}:".format(indent, name, threshold))
            #print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            #f.write("\n{}else:  # if {} > {}".format(indent, name, threshold))
            f.write(", ")
            #print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
            f.write(")")
        else:
            #f.write("\n{}return {}".format(indent, tree_.value[node]))
            prob = round(1 - tree_.value[node][0][0] / (tree_.value[node][0][0] + tree_.value[node][0][1]), 3)
            f.write("({})".format(prob))
            #print("{}return {}".format(indent, tree_.value[node]))
    recurse(0, 1)
    f.close()
    
# Aux function to apply Lamarck in symbolic expressions
def applyLamarckToPhen(phenotype, vals2):
    global phenotype2, x, y
    phenotype2 = phenotype
    #print(phenotype2)
    x, y, x_test, y_test = get_data(params['DATASET_TRAIN'], False)
    
    if params['N_SAMPLING'] > 0:
        N = params['N_SAMPLING']
        pos = np.where(y == 'Sale')[0]
        neg = np.where(y == 'NoSale')[0]
        randPos = sample(list(pos), N)
        randNeg = sample(list(neg), N)
        x = x.iloc[(randPos+randNeg),:]
        y = y.iloc[(randPos+randNeg)]
    
    #global res
    
    def aux_function(a):
        #print(a)
        global phenotype2
        #print(phenotype2)
        phenotype3 = phenotype2
        for i in range(len(a)):    
            phenotype3 = phenotype3.replace("a[" + str(i) + "]", str(round(a[i], 3)), 1)
        global x, y
        #print(phenotype3)
        pred = eval(phenotype3)
        auc = auc_metric(y, pred) #round(auc_metric(y, pred),2)
        return -auc
    import time
    #t0 = time.time()
    #print(phenotype2)
    res = minimize(aux_function, vals2, method='Powell', options={'maxiter' : 1, 'maxfev' : 1})
    #t1 = time.time()
    
    #print("num vals: ", len(vals2), "time: ", round(t1-t0, 2))
    if res.x.size == 1:
        phenotype5 = phenotype2.replace("a[0]", str(round(float(res.x), 3)), 1)
    else:
        #print(phenotype2)
        phenotype5 = phenotype2
        for i in range(res.x.size):
            phenotype5 = phenotype5.replace("a[" + str(i) + "]", str(round(res.x[i], 3)), 1)
        #print(phenotype5)

    #print(res.x.size)
    #print(phenotype5)
    return phenotype5


# Aux function to apply Lamarck in symbolic expressions
def applyLamarckToPhen2(phenotype, vals2):
    global phenotype2, x, y
    phenotype2 = phenotype
    #print(phenotype2)
    x, y, x_test, y_test = get_data(params['DATASET_TRAIN'], False)
    
    if params['N_SAMPLING'] > 0:
        N = params['N_SAMPLING']
        pos = np.where(y == 'Sale')[0]
        neg = np.where(y == 'NoSale')[0]
        randPos = sample(list(pos), N)
        randNeg = sample(list(neg), N)
        x = x.iloc[(randPos+randNeg),:]
        y = y.iloc[(randPos+randNeg)]

    y2 = np.where(y == 'Sale', 1, 0)

    #global res
    def func(a, y):
        #print(a)
        global phenotype2
        phenotype3 = phenotype2
        for i in range(len(a)):    
            phenotype3 = phenotype3.replace("a[" + str(i) + "]", str(round(a[i], 3)), 1)
        
        yhat = eval(phenotype2)
        return (y - yhat)
    
    import time
    t0 = time.time()
    out = leastsq(func, vals2, args=(y2))
    t1 = time.time()
    res = out[0]
    
    
    print("num vals: ", len(vals2), "time: ", round(t1-t0, 2))
    if res.x.size == 1:
        phenotype5 = phenotype2.replace("a[0]", str(round(float(res.x), 3)), 1)
    else:
        #print(phenotype2)
        phenotype5 = phenotype2
        for i in range(res.x.size):
            phenotype5 = phenotype5.replace("a[" + str(i) + "]", str(round(res.x[i], 3)), 1)
        #print(phenotype5)

    #print(res.x.size)
    #print(phenotype5)
    return phenotype5
