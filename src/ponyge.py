#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

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

def mane():
    #import GE
    """ Run program """
    
    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)
    
    # Load and plot statistics
    # TODO: change graphics!
    if params['SAVE_PLOTS']:
        if hasattr(params['FITNESS_FUNCTION'], 'multi_objective'):
            
            fileName1 = path.join(params['FILE_PATH'], "Pareto-Curve-Evolution.pdf")
            plt.style.use('grayscale')
            plt.xlabel('-AUC')
            plt.ylabel('No. nodes')
            for i in range(0,params['GENERATIONS']):
                pc = pd.read_csv((params['FILE_PATH'] + "/" + str(i) + 
                                  "_front/pareto.csv"), delimiter=";")
                plt.plot(pc['m1'], pc['m2'], marker='o')
            plt.savefig(fileName1)
            plt.close()
            
        else:
            # get stats
            fileName = path.join(params['FILE_PATH'], 'stats.tsv')
            results = pd.read_csv(fileName, delimiter='\t')
            ave_fitness = results.loc[:, "ave_fitness"]
            gens = results.loc[:, "gen"]
            best_fitness = results.loc[:, "best_fitness"]
            total_time = results.loc[:, "total_time"]
            fileName0 = path.join(params['FILE_PATH'], 'testFitness.txt')
            testFit = pd.read_csv(fileName0, delimiter='\n', header=None)
            if not len(testFit) == len(ave_fitness):
                testFit = testFit.iloc[0:len(ave_fitness),]
                #print("This was not supposed to happen")
            # AUC per generation (train + test)
            fileName1 = path.join(params['FILE_PATH'], "AUC-GEN.pdf")
            plt.plot(gens, ave_fitness, 'b', gens, best_fitness, 'r', gens, testFit, 'g')
            #plt.plot(gens, testFit, color='green')
            plt.grid(True)
            plt.xlabel('Generation')
            plt.ylabel('AUC')
            plt.legend(['Average Fitness', 'Best Fitness', 'Best Ind. on test data'])
            #plt.axis([0, max(gens), 0, 100])
            plt.ylim(top=100)
            plt.savefig(fileName1)
            plt.close()
            # AUC per time
            fileName2 = path.join(params['FILE_PATH'], "AUC-TIME.pdf")
            plt.plot(total_time, ave_fitness, 'b', total_time, best_fitness, 'r', total_time, testFit, 'g')
            plt.grid(True)
            plt.xlabel('Time (s)')
            plt.ylabel('AUC')
            plt.legend(['Average Fitness', 'Best Fitness'])
            #plt.axis([min(total_time), max(total_time), 0, 100])
            plt.ylim(top=100)
            plt.savefig(fileName2)
            plt.close()
        
    ### save predictions
    #best = individuals[0] #max(individuals)
    #indTest = round(params['FITNESS_FUNCTION'](best, dist='test', savePred=True),2)
    #print(indTest)
    

if __name__ == "__main__":
    set_params(sys.argv[1:])  # exclude the ponyge.py arg itself
    mane()