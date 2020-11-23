# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:26:33 2020

@author: pedro
"""

from utilities.algorithm.general import check_python_version
import matplotlib.pyplot as plt
import pandas as pd
from os import path

check_python_version()

from stats.stats import get_stats
from algorithm.parameters import params, set_params
import sys

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def set_params1(train_data, test_data, target, # data parameters are mandatory
          pop=12, gen=100, sampling=1000, 
          lamarck=True, sep=";", multicore=True):
    param_list = ['--population_size={0}'.format(str(pop)),
                  '--generations={0}'.format(str(gen)),
                  '--dataset_train={0}'.format(train_data),
                  '--dataset_test={0}'.format(test_data),
                  '--dataset_delimiter={0}'.fotmat(sep),
                  '--target={0}'.format(target),
                  '--multicore={0}'.format(str(multicore))]
    
    set_params(param_list)

def evolve():
    #import GE
    """ Run program """
    
    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)
    
    # Get stats
    fileName = path.join(params['FILE_PATH'], 'stats.tsv')
    results = pd.read_csv(fileName, delimiter='\t')
    
    return individuals, results

def mgedt(parametrization="auto"):
    if parametrization == "auto":
        set_params('')
    pop, res = evolve()

from multiprocessing import Pool
from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness
from stats.stats import stats, get_stats
from utilities.stats import trackers
from operators.initialisation import initialisation
from utilities.algorithm.initialise_run import pool_init
import json
from os import path

def mgedt_gui():
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
        # GUI
        #vid_dict = {}
        #vid_dict[0] = min((generation * 100) / total_gens, 100)
        #yield "data:" + str(x) + "\n\n"
        #ret_string = "data:" + json.dumps(vid_dict) + "\n\n"
        #print(ret_string)
        #yield ret_string
        stats['gen'] = generation
        # New generation
        individuals = params['STEP'](individuals)
        
    
    get_stats(individuals, end=True)
    

if __name__ == "__main__":
    set_params(sys.argv[1:])
    _ = evolve()
    #mane()