# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:16:15 2019

@author: pedro
"""
import pandas as pd
from sklearn import tree
from sklearn.metrics import f1_score 
import time
from sklearn.tree import _tree
import os
import numpy as np
from random import seed

seed(1234)

# save tree rules
def tree_to_code(tree, feature_names, i, transf, nrow):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    f = open((directory + str(i) + ".txt"), "w+")
    #f.write("def tree({}):".format(", ".join(feature_names)))
    #print("def tree({}):".format(", ".join(feature_names)))
    
    mapping_dict = {0 : 'high',
                    1 : 'low',
                    2 : 'medium',
                    3 : 'none',
                    4 : 'verylow'}

    def recurse(node, depth):
        #indent = "  " * depth
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            f.write("np.where(x['{}'] <= ({}), ".format(name, threshold))
            #f.write("\n{}if {} <= {}:".format(indent, name, threshold))
            #print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            #f.write("\n{}else:  # if {} > {}".format(indent, name, threshold))
            f.write(", ")
            #print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
            f.write(")")
        else:
            # class probabilities array in the following order:
                ## High, Low, Medium, None, VeryLow
            array_probs = tree_.value[node][0].tolist()
            pos = array_probs.index(max(array_probs)) # get index of max prob
            chosen_class = mapping_dict.get(pos)
            f.write("'{}'".format(chosen_class))
            #prob = 1 - round(tree_.value[node][0][0] / (tree_.value[node][0][0] + tree_.value[node][0][1]), 3)
            #f.write("({})".format(prob))
            #print("{}return {}".format(indent, tree_.value[node]))
    recurse(0, 1)
    f.close()
    f = open((directory + str(i) + ".txt"), "r")
    rules = f.read()
    f.close()
    return rules

allmacF1 = []
trainTime = []
predTime = []
transf = "IDF"
MODE = "TEST_new"
def toCategory(col):
    return col.astype('category')
#global dt
#dt = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=4)
    
for i in range(1,8):
    directory = MODE + "/" + transf + '/iteration' + str(i) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    dtrain = pd.read_csv(('../datasets/Promos/'+MODE+'/Train-' + transf + "-" 
                          + str(i)+'.csv'), delimiter=";") #read training data
    dtest = pd.read_csv(('../datasets/Promos/'+MODE+'/Test-' + transf + "-" + 
                         str(i)+'.csv'), delimiter=";") #read testing data
    #dtrain = dtrain.apply(toCategory, axis=0)
    #dtest = dtest.apply(toCategory, axis=0)
    # Test implementation of traditional decision trees in python
    # read the dataset
    x_train = dtrain.drop('ValueSale', 1)
    y_train = dtrain['ValueSale'].astype('category')
    
    x_test = dtest.drop('ValueSale', 1)
    y_test = dtest['ValueSale'].astype('category')
    
    print("Training the model...")
    start = time.time()
    # declare and train decision tree
    dt = tree.DecisionTreeClassifier().fit(x_train, y_train)
    end = time.time()
    t = round((end - start), 2)
    # predict
    start = time.time()
    y_pred = dt.predict(x_test)
    end = time.time()
    t2 = round(((end - start)/len(y_test))*1000, 5)
    predTime.append(t2)
    # macF1
    macF1 = f1_score(y_test, y_pred, average="macro") * -100
    allmacF1.append(macF1)
    trainTime.append(t)
    print("macF1: ", round(macF1,2), "\t Time train: ", t,
          "\t Time pred: ", round(t2,4))
    
    predictions = pd.DataFrame({'Y':y_test,'pred':y_pred}, index=None)
    predictions.to_csv(directory+"preds.csv",sep=";",index=False)
    
    #print("Saving tree...")
    rules = tree_to_code(dt, x_train.columns, i, transf, len(dtrain.index))
    #print(rules)
    f = open(directory + 'macF1_all.txt', 'w')
    f.write(str(macF1))
    f.close()

metrics = pd.DataFrame({'macF1': allmacF1, 
                        'TIME': trainTime, 
                        'TIMPEPRED': predTime})

metrics.to_csv((MODE + "/" + transf + '/' + 'TEST_metrics-RW_all.csv'),
               header=True, sep=";", index=False)
print(round(np.mean(allmacF1),3), np.mean(trainTime), np.mean(predTime))

