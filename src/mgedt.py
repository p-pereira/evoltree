# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:26:33 2020

@author: pedro
"""
class MGEDT(object):
    """
    MGEDT object.
    """
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    def __init__(self, pop=10, gen=5, lamarck=True, multicore=True, UI=False,
                 UI_params=None, param_list=[], **extra_params):
        from stats.stats import stats
        
        if len(param_list) == 0:
            if UI:
                for (key, val) in UI_params.items():
                    if val == "True":
                        param_list.append("--"+key)
                    elif val=="False" or val=="":
                        continue
                    #elif key == "X_train":
                    #    params[key] = val
                    #elif key == "y_train":
                    #    params[key] = val
                    #elif key == "X_test":
                    #    params[key] = val
                    #elif key == "y_test":
                    #    params[key] = val
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
                    #elif key == "X_train":
                    #    params[key] = val
                    #elif key == "y_train":
                    #    params[key] = val
                    #elif key == "X_test":
                    #    params[key] = val
                    #elif key == "y_test":
                    #    params[key] = val
                    else:
                        param_list.append("--{0}={1}".format(key, val))
        
        self.param_list = param_list
        self.params = {}
        self.population = []
        self.stats = stats
    
    def fit(self, X, y, X_val=None, y_val=None):
        from stats.stats import get_stats, stats
        from operators.initialisation import initialisation
        from fitness.evaluation import evaluate_fitness
        from tqdm import tqdm
        from algorithm.parameters import params, set_params
        from multiprocessing import Pool
        from utilities.algorithm.initialise_run import pool_init
        
        params["X_train"] = X
        params["y_train"] = y
        params["X_test"] = X
        params["y_test"] = y
        
        if len(self.params) == 0:
            set_params(self.param_list)
        
            if params["MULTICORE"]:
                if "POOL" in params.keys():
                    params["POOL"] = None
                # initialize pool once, if mutlicore is enabled
                params['POOL'] = Pool(processes=params['CORES'], 
                                      initializer=pool_init,
                                      initargs=(params,))
        self.params = params
        
        if self.population == []:
            # Initialise population
            self.population = initialisation(params['POPULATION_SIZE'])
            # Evaluate initial population
            self.population = evaluate_fitness(self.population)
        population = self.population
        
        # Generate statistics for run so far
        get_stats(population)
        
        total_gens = params['GENERATIONS']+1
        # Traditional GE
        for generation in tqdm(range(1, total_gens)):
            stats['gen'] = generation
            population = params['STEP'](population)
            population.sort(key=lambda x: x.fitness[0], reverse=False)
        
        if params['TARGET_SEED_FOLDER'] != "":
            self.store_pop(population)
        
        get_stats(population, end=True)
        
        self.stats = stats
        self.population = population
    
    def refit(self, generations):
        from stats.stats import get_stats, stats
        from algorithm.parameters import params
        from tqdm import tqdm
        
        population = self.population
        # Generate statistics for run so far
        get_stats(population)
        
        total_gens = params['GENERATIONS']+1 + generations
        # Traditional GE
        for generation in tqdm(range(params['GENERATIONS']+1, total_gens)):
            stats['gen'] = generation
            population = params['STEP'](population)
            population.sort(key=lambda x: x.fitness[0], reverse=False)
        
        get_stats(population, end=True)
        self.stats = stats
        self.population = population
    
    def fit_new_data(self, generations, X, y, 
                        X_test=None, y_test=None):
        from stats.stats import get_stats, stats
        from algorithm.parameters import params
        from tqdm import tqdm
        from math import log10
        
        (params['X_train'], params['y_train'], 
         params['X_test'], params['y_test']) = X, y, X_test, y_test
        
        population = self.population
        # Generate statistics for run so far
        get_stats(population)
        
        total_gens = params['GENERATIONS']+1 + generations
        # Traditional GE
        for generation in tqdm(range(params['GENERATIONS']+1, total_gens)):
            stats['gen'] = generation
            population = params['STEP'](population)
            
            min_y = log10(min(population, 
                              key=lambda x: x.fitness[1]).fitness[1])
            max_y = log10(max(population, 
                              key=lambda x: x.fitness[1]).fitness[1])
            population.sort(key=lambda x: self.get_distance(x, 
                                                            min_y, 
                                                            max_y), 
                            reverse=True)
        
        if params['TARGET_SEED_FOLDER'] != "":
            self.store_pop(population)
        
        get_stats(population, end=True)
        self.stats = stats
        self.population = population

    def store_pop(self, population):
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
                           key=lambda x: self.get_distance(x, 
                                                           min_y, 
                                                           max_y))
            preds = balanced.predict(x)
        
        return preds
    
    def get_distance(self, ind, min_y, max_y):
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

"""
def set_params1(train_data, test_data, target, # data parameters are mandatory
          delimiter=";", pop=12, gen=100, sampling=1000, 
          lamarck=True, multicore=True, **extra_params):
    param_list = ['--population_size={0}'.format(str(pop)),
                  '--generations={0}'.format(str(gen)),
                  '--dataset_train={0}'.format(train_data),
                  '--dataset_test={0}'.format(test_data),
                  '--dataset_delimiter={0}'.format(delimiter),
                  '--target={0}'.format(target)]
    if multicore:
        param_list.append("--multicore")
    if lamarck:
        param_list.append("--lamarck")
    for (key, val) in extra_params.items():
        if val == "True":
            param_list.append("--"+key)
        elif val=="False" or val=="":
            continue
        else:
            param_list.append("--{0}={1}".format(key, val))
    #set_params(param_list)
    
    return param_list

def evolve():
    from os import path
    import pandas as pd
    #import GE
    ""\" Run program ""\"
    from stats.stats import get_stats
    from algorithm.parameters import params
    
    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)
    
    # Get stats
    fileName = path.join(params['FILE_PATH'], 'stats.tsv')
    results = pd.read_csv(fileName, delimiter='\t')
    
    return individuals, results

def mgedt(parametrization="auto"):
    from algorithm.parameters import set_params
    if parametrization == "auto":
        set_params('')
    pop, res = evolve()

def mgedt_gui():
    from multiprocessing import Pool
    from algorithm.parameters import params
    from fitness.evaluation import evaluate_fitness
    from stats.stats import stats, get_stats
    from operators.initialisation import initialisation
    from utilities.algorithm.initialise_run import pool_init
    from os import path
    
    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])
    
    # Evaluate initial population
    individuals = evaluate_fitness(individuals)
    
    if params['SAVE_POP']:
        filename1 = path.join(params['FILE_PATH'], 'Begin-initialPop.txt')
        with open(filename1, 'w+', encoding="utf-8") as f:
            for item in individuals:
                f.write("%s\n" % item)
            f.close()
    
    # Generate statistics for run so far
    get_stats(individuals)
    
    total_gens = params['GENERATIONS']+1
    # Traditional GE
    for generation in range(1, total_gens):
        stats['gen'] = generation
        individuals = params['STEP'](individuals)
        
    
    get_stats(individuals, end=True)
    
"""

if __name__ == "__main__":
    import sys
    mgedt = MGEDT(param_list=sys.argv[1:])  # exclude the ponyge.py arg itself
    mgedt.evolve()
    mgedt.reevolve(10)