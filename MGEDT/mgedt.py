# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:26:33 2020

@author: pedro
"""
class MGEDT(object):
    """
    MGEDT object.
    
    """
   
    def __init__(self):
        self.params = {}
        self.population = []
        self.stats = {}
        self.fitted = False
    
    def fit(self, X, y, X_val=None, y_val=None, pop=100, gen=100, 
            lamarck=True, multicore=True, **extra_params):
        from src.algorithm.parameters import params, set_params
        from src.stats.stats import get_stats, stats
        from src.operators.initialisation import initialisation
        from src.fitness.evaluation import evaluate_fitness
        from tqdm import tqdm
        from multiprocessing import Pool
        from src.utilities.algorithm.initialise_run import pool_init
        
        new_params = {'X_train': X, 'y_train': y,
                      'X_test': X_val, 'y_test': y_val,
                      'POPULATION_SIZE': pop, 'GENERATIONS': gen,
                      'LAMARCK': lamarck, 'MULTICORE': multicore}
        params.update(new_params)
        
        param_list = list_params(pop, gen, lamarck, multicore, **extra_params)
        set_params(param_list)
        
        if params["MULTICORE"]:
            if "POOL" in params.keys():
                params["POOL"].close()
                params["POOL"] = None
            # initialize pool once, if mutlicore is enabled
            params['POOL'] = Pool(processes=params['CORES'], 
                                  initializer=pool_init,
                                  initargs=(params,))
        self.params.update(params)
        
        # Initialise population
        self.population = initialisation(params['POPULATION_SIZE'])
        # Evaluate initial population
        self.population = evaluate_fitness(self.population)
        stats['gen'] = 0
        # Generate statistics for run so far
        get_stats(self.population)
        population = self.population
        
        mlflow = get_mlflow(params['EXPERIMENT_NAME'])
        total_gens = params['GENERATIONS']+1
        range_generations = tqdm(range(1, total_gens))
        
        population = evolve(params, range_generations, mlflow, population)
        get_stats(population, end=True)
        store_pop(population)
        
        self.stats = stats
        self.population = population
        self.fitted = True
    
    def refit(self, gen):
        if not self.fitted:
            raise Exception("MGEDT needs to be fitted first. Use MGEDT.fit")
        from src.algorithm.parameters import params
        from src.stats.stats import get_stats, stats
        from tqdm import tqdm
        
        population = self.population
        # Generate statistics for run so far
        stats['gen'] = params['GENERATIONS']
        get_stats(population)
        
        mlflow = get_mlflow(params['EXPERIMENT_NAME'])
        
        total_gens = params['GENERATIONS'] + 1 + gen
        range_generations = tqdm(range(params['GENERATIONS'] + 1, total_gens))
        population = evolve(params, range_generations, mlflow, 
                            population, refit=True)
        
        get_stats(population, end=True)
        store_pop(population)
        self.stats = stats
        self.population = population
    
    def fit_new_data(self, X, y, X_val=None, y_val=None, pop=100, gen=100, 
                     lamarck=True, multicore=True, **extra_params):
        if not self.fitted:
            raise Exception("MGEDT needs to be fitted first. Use MGEDT.fit")
        from src.algorithm.parameters import params, set_params
        from src.stats.stats import get_stats, stats
        from tqdm import tqdm
        new_params = {'X_train': X, 'y_train': y,
                      'X_test': X_val, 'y_test': y_val,
                      'POPULATION_SIZE': pop, 'GENERATIONS': gen,
                      'LAMARCK': lamarck, 'MULTICORE': multicore}
        params.update(new_params)
        param_list = list_params(pop, gen, lamarck, multicore, **extra_params)
        set_params(param_list)
        
        self.params = params
        
        population = self.population
        
        mlflow = get_mlflow(params['EXPERIMENT_NAME'])
        
        total_gens = params['GENERATIONS']+1 + gen
        range_generations = tqdm(range(params['GENERATIONS']+1, total_gens))
        population = evolve(params, range_generations, mlflow, population)
        
        get_stats(population, end=True)
        store_pop(population)
        self.stats = stats
        self.population = population
    
    def predict(self, x, mode="best"):
        if mode == "all":
            preds = [ind.predict(x) for ind in self.population]
        elif mode == "best":
            best = min(self.population, key=lambda x: x.fitness[0])
            preds = best.predict(x)
        elif mode == "simplest":
            simplest = min(self.population, key=lambda x: x.fitness[1])
            preds = simplest.predict(x)
        elif mode == "balanced":
            from math import log10
            min_y = log10(min(self.population, 
                              key=lambda x: x.fitness[1]).fitness[1])
            max_y = log10(max(self.population, 
                              key=lambda x: x.fitness[1]).fitness[1])
            # get individual with greater distance to point (0, 1)
            balanced = max(self.population,
                           key=lambda x: self.__get_distance__(x, 
                                                               min_y, 
                                                               max_y))
            preds = balanced.predict(x)
        
        return preds
    
    def evaluate_all(self, X_test, y_test):
        import pandas as pd
        from src.utilities.fitness.error_metric import AUC
        aucs = [-1*AUC(y_test, ind.predict(X_test)) for ind in self.population]
        nodes = [ind.fitness[1] for ind in self.population]
        ev = pd.DataFrame([aucs, nodes]).T
        ev.columns=['auc','node']
        ev2 = ev.groupby('auc').agg({'node':min}).reset_index().values
        return [ev2[:,0], ev2[:,1]]
    
    def __get_tree_complexity__(self, dt, columns):
        nodes = get_nodes_from_tree(dt, columns, self.params)
        return nodes
    
    def __get_randForest_complexity__(self, rf, columns):
        nodes = [get_nodes_from_tree(dt, columns, self.params) 
                 for dt in rf.estimators_]
        return sum(nodes)
    
def store_pop(population):
    import os
    from src.algorithm.parameters import params
    if not os.path.exists("./seeds/" + params['TARGET_SEED_FOLDER']):
        os.makedirs("./seeds/" + params['TARGET_SEED_FOLDER'], 
                    exist_ok=True)
    for cont, item in enumerate(population):
        if item.phenotype != None:
            with open(("./seeds/" + params['TARGET_SEED_FOLDER'] 
                       + "/" + str(cont) + ".txt"), 'w+', 
                      encoding="utf-8") as f:
                f.write("Phenotype:\n")
                f.write("%s\n" % item.phenotype)
                f.write("Genotype:\n")
                f.write("%s\n" % item.genome)
                f.write("Tree:\n")
                f.write("%s\n" % str(item.tree))
                f.write("Training fitness:\n")
                f.write("%s\n" % item.fitness)
                f.close()
    
def get_distance(ind, min_y, max_y):
    import math
    auc = ind.fitness[0] / -100 # auc (positive, from 0 to 1)
    comp = math.log10(ind.fitness[1]) #complexity
    # scale complexity to [0, 1]
    comp = (comp - min_y) / (max_y - min_y)
    # worst result: (0, 1)
    x = 0
    y = 1
    # get distance:
    dist = math.hypot(auc-x, comp-y)
    return dist
    
def list_params(pop, gen, lamarck, multicore, **extra_params):
    param_list = []
    if 'population_size' not in extra_params.keys():
        param_list.append('--population_size={0}'.format(str(pop)))
    
    if 'generations' not in extra_params.keys():
        param_list.append('--generations={0}'.format(str(gen)))
    
    if multicore and 'multicore' not in extra_params.keys():
        param_list.append("--multicore")
    if lamarck and 'lamarck' not in extra_params.keys():
        param_list.append("--lamarck")
    
    for (key, val) in extra_params.items():
        if val == "True":
            param_list.append("--"+key)
        elif val=="False" or val=="":
            continue
        else:
            param_list.append("--{0}={1}".format(key, val))
    return param_list

def get_mlflow(experiment_name):
    import mlflow
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    try:
        _ = mlflow.create_experiment(experiment_name)
    except:
        _ = client.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)
    return mlflow

def evolve(params, range_generations, mlflow, population, refit=False):
    import numpy as np
    from src.stats.stats import stats
    from src.utilities.fitness.error_metric import AUC
    
    with mlflow.start_run():
        mlflow.log_param("REFIT", refit)
        for key, val in params.items():
            if key in ['X_train', 'y_train', 'X_test', 'y_test', 'POOL',
                       'BNF_GRAMMAR', 'SEED_INDIVIDUALS']:
                continue
            else:
                mlflow.log_param(key, val)
        # Traditional GE
        for generation in range_generations:
            stats['gen'] = generation
            
            population = params['STEP'](population)
            all_auc = [ind.fitness[0] for ind in population]
            all_nodes = [ind.fitness[1] for ind in population]
            mlflow.log_metrics(metrics={"1st ind AUC" : -min(all_auc),
                                        "1st ind NODES" : max(all_nodes),
                                        "last ind AUC" : -max(all_auc),
                                        "last ind NODES" : min(all_nodes),
                                        "mean AUC" : -np.mean(all_auc),
                                        "mean nodes" : np.mean(all_nodes)},
                               step=generation)
        val_aucs = [AUC(params['y_test'], 
                        ind.predict(params['X_test']))\
                    for ind in population]
        mlflow.log_metric("best val AUC", -min(val_aucs))
    return population

def get_nodes_from_tree(tree, feature_names, params):
    from importlib import import_module
    from sklearn.tree import _tree
    from src.algorithm.mapper import map_tree_from_genome
    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    def recurse(node, depth, rule=""):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            rule = rule + "np.where(x['{}'] <= {}, ".format(name, threshold)
            rule = recurse(tree_.children_left[node], depth + 1, rule)
            rule = rule + ", "
            rule = recurse(tree_.children_right[node], depth + 1, rule)
            rule = rule + ")"
            return rule
        else:
            prob = 1 - round(tree_.value[node][0][0] / 
                             (tree_.value[node][0][0] + 
                              tree_.value[node][0][1]), 3)
            rule = rule + "({})".format(prob)
            return rule
    
    rules = recurse(0, 1)
    i = import_module(params['LAMARCK_MAPPER'])
    genome = i.get_genome_from_dt_idf(rules)
    nodes = map_tree_from_genome(genome)[3]
    return nodes