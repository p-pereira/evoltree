# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:26:33 2020

@author: pedro
"""
class MGEDT(object):
    """
    MGEDT object.
    """
   # from warnings import simplefilter
    # ignore all future warnings
    #simplefilter(action='ignore', category=FutureWarning)
    def __init__(self):
        from stats.stats import stats
        
        self.param_list = []
        self.params = {}
        self.population = []
        self.stats = stats
        self.fitted = False
    
    def fit(self, X, y, X_val=None, y_val=None, pop=100, gen=100, 
            lamarck=True, multicore=True, **extra_params):
        from stats.stats import get_stats, stats
        from operators.initialisation import initialisation
        from fitness.evaluation import evaluate_fitness
        from tqdm import tqdm
        from algorithm.parameters import params, set_params
        from multiprocessing import Pool
        from utilities.algorithm.initialise_run import pool_init
        
        #if len(self.params) == 0:
        self.param_list = self.__set_params__(pop, gen, lamarck, multicore,
                                              **extra_params)
        set_params(self.param_list)
        
        if params["MULTICORE"]:
            if "POOL" in params.keys():
                params["POOL"] = None
            # initialize pool once, if mutlicore is enabled
            params['POOL'] = Pool(processes=params['CORES'], 
                                  initializer=pool_init,
                                  initargs=(params,))
        self.ml_params = params
        (params['X_train'], params['y_train'], 
         params['X_test'], params['y_test']) = X, y, X_val, y_val
        
        self.params = params
        
        if self.population == []:
            # Initialise population
            self.population = initialisation(params['POPULATION_SIZE'])
            # Evaluate initial population
            self.population = evaluate_fitness(self.population)
        population = self.population
        
        # Generate statistics for run so far
        get_stats(population)
        
        mlflow = self.__get_mlflow__(params['EXPERIMENT_NAME'])
        total_gens = params['GENERATIONS']+1
        range_generations = tqdm(range(1, total_gens))
        
        population = self.__evolve__(params, range_generations,
                                     mlflow, population)
        get_stats(population, end=True)
        
        self.stats = stats
        self.population = population
        self.fitted = True
    
    def refit(self, gen):
        if not self.fitted:
            raise Exception("MGEDT needs to be fitted first. Use MGEDT.fit")
        from stats.stats import get_stats, stats
        from algorithm.parameters import params
        from tqdm import tqdm
        
        population = self.population
        # Generate statistics for run so far
        get_stats(population)
        
        mlflow = self.__get_mlflow__(params['EXPERIMENT_NAME'])
        
        total_gens = params['GENERATIONS'] + 1 + gen
        range_generations = tqdm(range(params['GENERATIONS'] + 1, total_gens))
        population = self.__evolve__(params, range_generations,
                                     mlflow, population, refit=True)
        
        get_stats(population, end=True)
        self.stats = stats
        self.population = population
    
    def fit_new_data(self, X, y, X_val=None, y_val=None, pop=100, gen=100, 
                     lamarck=True, multicore=True, **extra_params):
        if not self.fitted:
            raise Exception("MGEDT needs to be fitted first. Use MGEDT.fit")
        from stats.stats import get_stats, stats
        from algorithm.parameters import params, set_params
        from tqdm import tqdm
        
        self.param_list = self.__set_params__(pop, gen, lamarck, multicore,
                                              **extra_params)
        set_params(self.param_list)
        
        self.ml_params = params
        
        (params['X_train'], params['y_train'], 
         params['X_test'], params['y_test']) = X, y, X_val, y_val
        
        self.params = params
        
        population = self.population
        # Generate statistics for run so far
        get_stats(population)
        
        mlflow = self.__get_mlflow__(params['EXPERIMENT_NAME'])
        
        total_gens = params['GENERATIONS']+1 + gen
        range_generations = tqdm(range(params['GENERATIONS']+1, total_gens))
        population = self.__evolve__(params, range_generations,
                                     mlflow, population)
        
        get_stats(population, end=True)
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
    
    def __evolve__(self, params, range_generations, 
                   mlflow, population, refit=False):
        from stats.stats import stats
        import numpy as np
        from utilities.fitness.error_metric import AUC
        
        with mlflow.start_run():
            mlflow.log_param("REFIT", refit)
            for key, val in self.ml_params.items():
                if key in ['X_train', 'y_train', 'X_test', 'y_test', 'POOL',
                           'BNF_GRAMMAR', 'SEED_INDIVIDUALS']:
                    continue
                else:
                    mlflow.log_param(key, val)
            # Traditional GE
            for generation in range_generations:
                stats['gen'] = generation
                population = params['STEP'](population)
                #population.sort(key=lambda x: x.fitness[0], reverse=False)
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
    
    def __get_mlflow__(self, experiment_name):
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        try:
            _ = mlflow.create_experiment(experiment_name)
        except:
            _ = client.get_experiment_by_name(experiment_name)
        mlflow.set_experiment(experiment_name)
        return mlflow
    
    def __store_pop__(self, population):
        import os
        from algorithm.parameters import params
        
        if not os.path.exists("../seeds/" + params['TARGET_SEED_FOLDER']):
            os.makedirs("../seeds/" + params['TARGET_SEED_FOLDER'], 
                        exist_ok=True)
        for cont, item in enumerate(population):
            if item.phenotype != None:
                with open(("../seeds/" + params['TARGET_SEED_FOLDER'] 
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
    
    def __get_distance__(self, ind, min_y, max_y):
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
    
    def __set_params__(self, pop=100, gen=100, lamarck=True, multicore=True, 
                       UI=False, UI_params=None, param_list=[], 
                       **extra_params):
        if len(param_list) == 0:
            if UI:
                for (key, val) in UI_params.items():
                    if val == "True":
                        param_list.append("--"+key)
                    elif val=="False" or val=="":
                        continue
                    else:
                        param_list.append("--{0}={1}".format(key, val))
            else:
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

if __name__ == "__main__":
    # IMPLEMENTATION EXAMPLE
    import pandas as pd
    from sklearn.model_selection import train_test_split
    d = pd.read_csv("../datasets/Promos/TEST2/Train-IDF-1.csv", sep=";")
    
    dtrain, dtest = train_test_split(d, test_size=0.3, stratify=d['target'])
    dtrain, dval = train_test_split(dtrain, test_size=0.3, stratify=dtrain['target'])
    X = dtrain.drop('target', axis=1)
    y = dtrain['target']
    X_val = dval.drop('target', axis=1)
    y_val = dval['target']
    X_test = dtest.drop('target', axis=1)
    y_test = dtest['target']
    mgedt = MGEDT(pop=20)
    mgedt.fit(X, y, X_val, y_val)
    mgedt.predict(X_test)
    