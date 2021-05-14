from multiprocessing import Pool
import os
from os import path

from src.algorithm.parameters import params
from src.fitness.evaluation import evaluate_fitness
from src.stats.stats import stats, get_stats
from src.utilities.stats import trackers
from src.operators.initialisation import initialisation
from src.utilities.algorithm.initialise_run import pool_init


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
        filename1 = path.join(params['FILE_PATH'], 'Begin-initialPop.txt')
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
        filename2 = path.join(params['FILE_PATH'], 'End-finalPop.txt')
        with open(filename2, 'w+') as f:
            for item in individuals:
                f.write("%s\n" % item)
            f.close()
        
    if params['TARGET_SEED_FOLDER'] != "":
        if not os.path.exists("seeds/" + params['TARGET_SEED_FOLDER']):
            os.makedirs("seeds/" + params['TARGET_SEED_FOLDER'], 
                        exist_ok=True)
        for cont, item in enumerate(individuals):
            if item.phenotype != None:
                with open(("seeds/" + params['TARGET_SEED_FOLDER'] 
                           + "/" + str(cont) + ".txt"), 'w+', 
                          encoding="utf-8") as f:
                    f.write("Phenotype:\n")
                    f.write("%s\n" % item.phenotype)
                    #f.write("Genotype:\n")
                    #f.write("%s\n" % item.genome)
                    #f.write("Tree:\n")
                    #f.write("%s\n" % str(item.tree))
                    f.write("Training fitness:\n")
                    f.write("%s\n" % item.fitness)
                    f.close()
        
    
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