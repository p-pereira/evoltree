# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:26:33 2020

@author: pedro
"""

from utilities.algorithm.general import check_python_version
import pandas as pd
from os import path

check_python_version()

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class MGEDT(object):
    """
    MGEDT object.
    """
    def __init__(self, pop=10, gen=100, lamarck=True, multicore=True, 
                 **extra_params):
        from stats.stats import stats
        from algorithm.parameters import params, set_params
        from multiprocessing import Pool
        from utilities.algorithm.initialise_run import pool_init
        from operators.initialisation import initialisation
        from fitness.evaluation import evaluate_fitness
        
        param_list = ['--population_size={0}'.format(str(pop)),
                      '--generations={0}'.format(str(gen))]
        if multicore:
            param_list.append("--multicore")
        if lamarck:
            param_list.append("--lamarck")
        
        for (key, val) in extra_params.items():
            if val == "True":
                param_list.append("--"+key)
            elif val=="False" or val=="":
                continue
            elif key == "X_train":
                params[key] = val
            elif key == "y_train":
                params[key] = val
            elif key == "X_test":
                params[key] = val
            elif key == "y_test":
                params[key] = val
            else:
                param_list.append("--{0}={1}".format(key, val))
        set_params(param_list)
        
        if multicore:
            if "POOL" in params.keys():
                params["POOL"] = None
            # initialize pool once, if mutlicore is enabled
            params['POOL'] = Pool(processes=params['CORES'], 
                                  initializer=pool_init,
                                  initargs=(params,))
        # Initialise population
        individuals = initialisation(pop)
        # Evaluate initial population
        individuals = evaluate_fitness(individuals)
        self.params = params
        self.population = individuals
        self.stats = stats
    
    def evolve(self):
        from stats.stats import get_stats, stats
        from algorithm.parameters import params
        from tqdm import tqdm
        
        population = self.population
        # Generate statistics for run so far
        get_stats(population)
        
        total_gens = params['GENERATIONS']+1
        # Traditional GE
        for generation in tqdm(range(1, total_gens)):
            stats['gen'] = generation
            population = params['STEP'](population)
        
        get_stats(population, end=True)
        self.stats = stats
        self.population = population
    
    def reevolve(self, generations):
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
        
        get_stats(population, end=True)
        self.stats = stats
        self.population = population
    

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
    #import GE
    """ Run program """
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
    

#if __name__ == "__main__":
#    #set_params(sys.argv[1:])
#    #_ = evolve()
#    #mane()