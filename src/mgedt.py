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


if __name__ == "__main__":
    set_params(sys.argv[1:])
    _ = evolve()
    #mane()