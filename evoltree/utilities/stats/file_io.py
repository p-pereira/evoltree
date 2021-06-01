from os import path, getcwd, makedirs
from shutil import rmtree
from copy import copy

from ...algorithm.parameters import params
from . import trackers
import pandas as pd

def save_stats_to_file(stats, end=False):
    """
    Write the results to a results file for later analysis

    :param stats: The stats.stats.stats dictionary.
    :param end: A boolean flag indicating whether or not the evolutionary
    process has finished.
    :return: Nothing.
    """

    if params['VERBOSE']:
        filename = path.join(params['FILE_PATH'], "stats.tsv")
        savefile = open(filename, 'a')
        for stat in sorted(stats.keys()):
            savefile.write(str(stats[stat]) + "\t")
        savefile.write("\n")
        savefile.close()

    elif end:
        filename = path.join(params['FILE_PATH'], "stats.tsv")
        savefile = open(filename, 'a')
        for item in trackers.stats_list:
            for stat in sorted(item.keys()):
                savefile.write(str(item[stat]) + "\t")
            savefile.write("\n")
        savefile.close()


def save_stats_headers(stats):
    """
    Saves the headers for all stats in the stats dictionary.

    :param stats: The stats.stats.stats dictionary.
    :return: Nothing.
    """

    filename = path.join(params['FILE_PATH'], "stats.tsv")
    savefile = open(filename, 'w')
    for stat in sorted(stats.keys()):
        savefile.write(str(stat) + "\t")
    savefile.write("\n")
    savefile.close()


def save_best_ind_to_file(stats, ind, end=False, name="best"):
    """
    Saves the best individual to a file.

    :param stats: The stats.stats.stats dictionary.
    :param ind: The individual to be saved to file.
    :param end: A boolean flag indicating whether or not the evolutionary
    process has finished.
    :param name: The name of the individual. Default set to "best".
    :return: Nothing.
    """

    filename = path.join(params['FILE_PATH'], (str(name) + ".txt"))
    savefile = open(filename, 'w')
    savefile.write("Generation:\n" + str(stats['gen']) + "\n\n")
    savefile.write("Phenotype:\n" + str(ind.phenotype) + "\n\n")
    savefile.write("Genotype:\n" + str(ind.genome) + "\n")
    savefile.write("Tree:\n" + str(ind.tree) + "\n")
    if hasattr(params['FITNESS_FUNCTION'], "training_test"):
        if end:
            savefile.write("\nTraining fitness:\n" + str(ind.training_fitness))
            savefile.write("\nTest fitness:\n" + str(ind.test_fitness))
        else:
            savefile.write("\nFitness:\n" + str(ind.fitness))
    else:
        savefile.write("\nFitness:\n" + str(ind.fitness))
    savefile.close()

    if hasattr(params['FITNESS_FUNCTION'], "training_test") and end:
        with open(path.join(params['FILE_PATH'], "test.txt"), 'w') as the_file:
            the_file.write(str(ind.test_fitness))

def save_first_front_to_file(stats, end=False, name="first"):
    """
    Saves all individuals in the first front to individual files in a folder.

    :param stats: The stats.stats.stats dictionary.
    :param end: A boolean flag indicating whether or not the evolutionary
                process has finished.
    :param name: The name of the front folder. Default set to "first_front".
    :return: Nothing.
    """

    # Save the file path (we will be over-writing it).
    orig_file_path = copy(params['FILE_PATH'])

    # Define the new file path.
    params['FILE_PATH'] = path.join(orig_file_path, str(name)+"_front")

    # Check if the front folder exists already
    if path.exists(params['FILE_PATH']):

        # Remove previous files.
        rmtree(params['FILE_PATH'])

    # Create front folder.
    makedirs(params['FILE_PATH'])

    paretoFront = pd.DataFrame(columns=['m1','m2'])
    #print(params['FILE_PATH'] + "-testFitness.txt")
    filename = params['FILE_PATH'] + "-validationFitness.txt"
    
    #if path.exists(filename):
    #    append_write = 'a' # append if already exists
    #else:
    #    append_write = 'w' # make a new file if not
    savefile = open(filename, 'w')
    for i, ind in enumerate(trackers.best_ever):
        # Save each individual in the first front to file.
        save_best_ind_to_file(stats, ind, end, name=str(i))
        # Save Pareto fronts
        paretoFront.loc[i] = ind.fitness
        
        # Save individuals' fitness on test data <<- NEW
        f = params['FITNESS_FUNCTION']
        indTest = [f.fitness_functions[0](ind, dist='test'), f.fitness_functions[1](ind, dist='test')]
        #print(indTest)
        
        savefile.write((str(indTest) + "\n"))
    savefile.close()
    
    # Save Pareto Front's metrics --> NEW
    paretoFront.to_csv(path.join(params['FILE_PATH'], "pareto.csv"), sep=";", 
                       header=True, index=False)
    # Re-set the file path.
    params['FILE_PATH'] = copy(orig_file_path)


def generate_folders_and_files():
    """
    Generates necessary folders and files for saving statistics and parameters.

    :return: Nothing.
    """

    if params['EXPERIMENT_NAME']:
        # Experiment manager is being used.
        path_1 = path.join(getcwd(), "evoltree", "results")

        if not path.isdir(path_1):
            # Create results folder.
            makedirs(path_1)

        # Set file path to include experiment name.
        params['FILE_PATH'] = path.join(path_1, params['EXPERIMENT_NAME'])

    else:
        # Set file path to results folder.
        params['FILE_PATH'] = path.join(getcwd(), "results")

    # Generate save folders
    if not path.isdir(params['FILE_PATH']):
        makedirs(params['FILE_PATH'])

    #NEW
    if params['FOLDER_NAME'] is None:
        if not path.isdir(path.join(params['FILE_PATH'], str(params['TIME_STAMP']))):
            makedirs(path.join(params['FILE_PATH'], str(params['TIME_STAMP'])))
        params['FILE_PATH'] = path.join(params['FILE_PATH'], str(params['TIME_STAMP']))

    else:
        if not path.isdir(path.join(params['FILE_PATH'], params['FOLDER_NAME'] )):
            makedirs(path.join(params['FILE_PATH'], params['FOLDER_NAME'] ))
        params['FILE_PATH'] = path.join(params['FILE_PATH'], params['FOLDER_NAME'] )

    save_params_to_file()


def save_params_to_file():
    """
    Save evolutionary parameters in a parameters.txt file.

    :return: Nothing.
    """

    # Generate file path and name.
    filename = path.join(params['FILE_PATH'], "parameters.txt")
    savefile = open(filename, 'w')

    # Justify whitespaces for pretty printing/saving.
    col_width = max(len(param) for param in params.keys())

    for param in sorted(params.keys()):

        # Create whitespace buffer for pretty printing/saving.
        spaces = [" " for _ in range(col_width - len(param))]
        ### NEW 29-11-2020: ignore extensive parameters
        if param not in ['X_train', 'y_train', 'X_test', 'y_test',
                         'BNF_GRAMMAR']:
            savefile.write(str(param) + ": " + "".join(spaces) +
                           str(params[param]) + "\n")

    savefile.close()
