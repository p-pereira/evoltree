from multiprocessing import Pool
import os
from os import path

from ..algorithm.parameters import params
from ..fitness.evaluation import evaluate_fitness
from ..stats.stats import stats, get_stats
from ..utilities.stats import trackers
from ..operators.initialisation import initialisation
from ..utilities.algorithm.initialise_run import pool_init


def search_loop():
    """
    This is a standard search process for an evolutionary algorithm. Loop over
    a given number of generations.
    
    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])
    
    # Evaluate initial population
    individuals = evaluate_fitness(individuals)
    
    if params['SAVE_POP']:
        filename1 = path.join(params['FILE_PATH'], 'InitialPop.txt')
        with open(filename1, 'w+', encoding="utf-8") as f:
            for item in individuals:
                f.write("%s\n" % item)
            f.close()
    
    # Generate statistics for run so far
    get_stats(individuals)

    # Traditional GE
    for generation in range(1, (params['GENERATIONS']+1)):
        stats['gen'] = generation

        # New generation
        individuals = params['STEP'](individuals)

    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()
    
    if params['SAVE_POP']:
        filename2 = path.join(params['FILE_PATH'], 'FinalPop.txt')
        with open(filename2, 'w+') as f:
            for item in individuals:
                f.write("%s\n" % item)
            f.close()
    
    """ NEW 27-05-2021: This is now managed in mgedt.py
    if params['TARGET_SEED_FOLDER'] != "":
        import pkg_resources
        SEEDS_PATH = pkg_resources.resource_filename('evoltree', 'seeds')
        os.makedirs(path.join(SEEDS_PATH, params['TARGET_SEED_FOLDER']),
                    exist_ok=True)
        for cont, item in enumerate(individuals):
            fname = path.join(SEEDS_PATH, params['TARGET_SEED_FOLDER'],
                              "{0}.txt".format(str(cont)))
            if item.phenotype != None:
                with open(fname, 'w+', encoding="utf-8") as f:
                    f.write("Phenotype:\n")
                    f.write("%s\n" % item.phenotype)
                    f.write("Training fitness:\n")
                    f.write("%s\n" % item.fitness)
                    f.close()
    """
    return individuals


def search_loop_from_state():
    """
    Run the evolutionary search process from a loaded state. Pick up where
    it left off previously.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """
    
    individuals = trackers.state_individuals
    
    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)
    
    # Traditional GE
    for generation in range(stats['gen'] + 1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation
        
        # New generation
        individuals = params['STEP'](individuals)
    
    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()
    
    return individuals