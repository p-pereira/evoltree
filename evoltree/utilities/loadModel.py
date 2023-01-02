# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:11:57 2020

@author: pedro

load an individual or entire population and apply to new data
"""
from ..operators.initialisation import load_population
from ..algorithm.parameters import set_params, params
from ..representation.individual import Individual
from importlib import import_module
import time
from os import getcwd, path
import logging

if __name__ == "__main__":
    set_params('')

def loadPop(folder="Test", otherDir=getcwd()):
    """
    Given a target folder, read all files in the folder and load/parse
    solutions found in each file.
    
    :param folder: A target folder stored in the "seeds" folder.
    :return: A list of all parsed individuals stored in the target folder.
    """
    population = load_population(folder, otherDir)
    return population

def loadBestInd(folder="Test",otherDir=getcwd()):
    """
    Given a target folder, read the file 0.txt that contains the individual
    with the best fitness value.
    
    :param folder: A target folder stored in the otherDir folder.
    :param otherDir: A target directory with the 'folder' with solutions.
    :return: The best parsed individual stored in the target folder.
    """
    # Set path for seeds folder
    if otherDir == getcwd():
        path_1 = path.join(otherDir, "seeds")
    else:
        path_1 = otherDir

    if not path.isdir(path_1):
        # Seeds folder does not exist.
    
        s = "scripts.seed_PonyGE2.load_population\n" \
            "Error: `seeds` folder does not exist in root directory."
        raise Exception(s)
    
    path_2 = path.join(path_1, folder)

    if not path.isdir(path_2):
        # Target folder does not exist.
    
        s = "scripts.seed_PonyGE2.load_population\n" \
            "Error: target folder " + folder + \
            " does not exist in seeds directory."
        raise Exception(s)
    file_name = path.join(path_2, "0.txt")
    ind = loadSpecificInd(file_name)
    return ind
    
def loadSpecificInd(file_name):
    """
    Given a specific file_name, reads the file and returns the parsed 
    individual on that file.
    
    :param folder: A file_name where the individual is stored.
    :return: The parsed individuals stored in the file.
    """
    # Initialise None data for ind info.
    genotype, phenotype, fitness = None, None, None
    # Open file.
    with open(file_name, "r") as f:
        # Read file.
        raw_content = f.read()
        # Split content by \n
        content = raw_content.split("\n")
        # Check if phenotype (target string) is already saved in file.
        if "Phenotype:" in content:
            # Get index location of genotype.
            phen_idx = content.index("Phenotype:") + 1
            # Get the phenotype.
            phenotype = content[phen_idx]
            # Treat string
            phenotype = phenotype.replace("(1.0)", "(1)")            
            # TODO: Current phenotype is read in as single-line only. Split is performed on "\n", meaning phenotypes that span multiple lines will not be parsed correctly. This must be fixed in later editions.        
        # Check if genotype is already saved in file.
        elif "Genotype:" in content:
            # Get index location of genotype.
            gen_idx = content.index("Genotype:") + 1
            # Get the genotype.
            try:
                genotype = eval(content[gen_idx])
            except:
                s = "scripts.seed_PonyGE2.load_population\n" \
                    "Error: Genotype from file " + file_name + \
                    " not recognized: " + content[gen_idx]
                raise Exception(s)
        elif "Genotype:" not in content:
            # There is no explicit genotype or phenotype in the target
            # file, read in entire file as phenotype.
            phenotype = raw_content
        # Check if fitness is already saved in file.
        if "Training fitness:" in content:            
            # Get index location of genotype.
            gen_idx = content.index("Training fitness:") + 1            
            # Get the genotype.
            try:
                fitness = content[gen_idx]
            except:
                s = "scripts.seed_PonyGE2.load_population\n" \
                    "Error: Fitness from file " + file_name + \
                    " not recognized: " + content[gen_idx]
                raise Exception(s)
    if genotype:
        # Generate individual from genome.
        ind = Individual(genotype, None)
        if phenotype and ind.phenotype != phenotype:
            s = "scripts.seed_PonyGE2.load_population\n" \
                "Error: Specified genotype from file " + file_name + \
                " doesn't map to same phenotype. Check the specified " \
                "grammar to ensure all is correct: " + \
                params['GRAMMAR_FILE']
            raise Exception(s)
    else:
        ### new: save phenotype only and get the genotype form it
        get_genome_from_dt_idf = import_module(params['LAMARCK_MAPPER'])
        genotype = get_genome_from_dt_idf(phenotype)
        # Generate individual from genome.
        ind = Individual(genotype, None)
    # Set individual's fitness
    if fitness and not hasattr(params['FITNESS_FUNCTION'], 'multi_objective'):
        ind.fitness = float(fitness)
    
    return ind

def getPredictions(phenotype, x, verbose=False):
    """
    Given an individual's phenotype and the data, returns the predictions.
    
    :param phenotype: An expression to apply on data.
    :param x: Data where predictions will be made.
    :return: A list with the predictions.
    """
    import numpy as np # Decision trees are in the format np.where(expression)
    
    start = time.time()
    predictions = eval(phenotype)
    end = time.time()
    t1 = (end - start)
    if type(predictions) == np.float or type(predictions) == np.int:
        predictions = np.repeat(predictions, len(x.iloc[:,0]))
    #print(type(predictions))
    tpred = (t1 / len(predictions))*1000 # ms
    if verbose:
        logging.info('Time per prediction (ms): ' + str(tpred)) #time per prediction
    return predictions, tpred