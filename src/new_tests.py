# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 22:15:06 2021

@author: pedro
"""

from src.algorithm.parameters import set_params, params
from src.operators.initialisation import initialisation
from src.fitness.evaluation import evaluate_fitness
from src.operators.crossover import crossover
from src.operators.mutation import mutation
from src.operators.replacement import replacement
from src.operators.selection import selection
import re
from random import sample, uniform
from importlib import import_module
from src.algorithm.mapper import map_tree_from_genome
from time import time

def aux_evals(pop):
    return [ind.fitness for ind in pop]

def aux_invalids(pop):
    return [ind.invalid for ind in pop]


def get_all_nodes(phenotype):
    """
    Extracts all decision nodes and leafs from a DT phenotype.
    :param phenotype: an individual's phenotype in DT format.
    :returns: All leafs and decision dodes.
    """
    inits = [node.start() for node in re.finditer('np.where', phenotype)]
    ends = []
    for nodePosition in inits:
        # get the expression
        #   get opened and closed brackets' positions
        openBrPos = [m.start() for m in re.finditer('\(', 
                                                    phenotype[nodePosition:])]
        closeBrPos = [m.start() for m in re.finditer('\)', 
                                                     phenotype[nodePosition:])]
        allBrPos = sorted(openBrPos + closeBrPos)
        # get the position where the expression to be replaced ends
        openBr = 0
        closeBr = 0
        position = 1
        for i in range(0,len(allBrPos)):
            pos = allBrPos[i]
            if pos in openBrPos:
                openBr = openBr + 1
            elif pos in closeBrPos:
                closeBr = closeBr + 1
            if openBr == closeBr:
                position = pos
                ends.append(position)
                break
    
    decision_nodes = [phenotype[start:][:end+1] for start, end in zip(inits, 
                                                                      ends)]
    leafs = re.findall(r"\(\d+.\d+\)|\(\d+\)", phenotype)
    
    return leafs + decision_nodes

def DT_crossover_inds(ind1, ind2):
    """
    Applies crossover to individuals. Takes two DTs, randomly chooses a subtree
    from each and changes it.
    :param ind1: individual1 to apply crossover operator.
    :param ind2: individual2 to apply crossover operator.
    :returns: both changed individuals.
    """
    # Get phenotypes from individuals
    phen1 = ind1.phenotype
    phen2 = ind2.phenotype
    # Get all decision nodes and leafs from each individual's phenotype
    nodes1 = get_all_nodes(phen1)
    nodes2 = get_all_nodes(phen2)
    # Randomly choose which node to replace in each phenotype
    node1 = sample(nodes1, 1)[0]
    node2 = sample(nodes2, 1)[0]
    # Replace node in each phenotype
    phen1 = phen1.replace(node1, node2)
    phen2 = phen2.replace(node2, node1)
    # Get new genome from new phenotype for both individuals
    try : # invalid individuals can be generated
        i = import_module(params['LAMARCK_MAPPER'])
        # individual1
        genome1 = i.get_genome_from_dt_idf(phen1)
        mapped1 = map_tree_from_genome(genome1)
        ind1.phenotype = phen1
        ind1.genome = genome1
        ind1.tree = mapped1[2]
        ind1.nodes = mapped1[3]
        # individual2
        genome2 = i.get_genome_from_dt_idf(phen2)
        mapped2 = map_tree_from_genome(genome2)
        ind2.phenotype = phen2
        ind2.genome = genome2
        ind2.tree = mapped2[2]
        ind2.nodes = mapped2[3]
        # done!
        return ind1, ind2
    except:
        return None
        
def DT_crossover(pop):
    # Initialise an empty population.
    cross_pop = []
    #cont = 0
    
    while len(cross_pop) < params['GENERATION_SIZE']:
        
        # Randomly choose two parents from the parent population.
        inds_in = sample(pop, 2)

        # Perform crossover on chosen parents.
        inds_out = DT_crossover_inds(inds_in[0], inds_in[1])
        
        if inds_out is None:
            # Crossover failed.
            #cont +=1
            #print("Crossover failed for the {0} time".format(str(cont)))
            pass
        
        else:
                        
            # Extend the new population.
            cross_pop.extend(inds_out)

    return cross_pop

def DT_mutation_ind(ind):
    phen = ind.phenotype
    for i in range(params['MUTATION_EVENTS']):
        leafs = re.findall(r"\(\d+.\d+\)|\(\d+\)", phen)
        rand_prob = uniform(0, 1)
        rand_node = sample(leafs, 1)[0]
        phen = phen.replace(rand_node, '({0})'.format(str(rand_prob)))
    i = import_module(params['LAMARCK_MAPPER'])
    genome = i.get_genome_from_dt_idf(phen)
    mapped = map_tree_from_genome(genome)
    ind.phenotype = phen
    ind.genome = genome
    ind.tree = mapped[2]
    ind.nodes = mapped[3]
    return ind

def DT_mutation(pop):
    # Initialise empty pop for mutated individuals.
    new_pop = []
    
    for ind in pop:
        new_ind = DT_mutation_ind(ind)
        new_pop.append(new_ind)
    return new_pop

set_params('')

print("Start Initialization...")
s = time()
# Initialise population
pop0 = initialisation(params['POPULATION_SIZE'])
pop0 = evaluate_fitness(pop0)
evals0 = aux_evals(pop0)
invalids0 = aux_invalids(pop0)
e = time()
print("Initialization finished. Time: ", str(e-s))
print("Start Selection...")
s = time()
# Select parents from the original population.
pop1 = selection(pop0)
evals1 = aux_evals(pop1)
invalids1 = aux_invalids(pop1)
e = time()
print("Selection finished. Time: ", str(e-s))
print("Start Crossover GE...")
s = time()
# Crossover parents and add to the new population.
pop2 = crossover(pop1)
pop2 = evaluate_fitness(pop2)
evals2 = aux_evals(pop2)
invalids2 = aux_invalids(pop2)
e = time()
print("Crossover GE finished. Time: ", str(e-s))
print("Start Crossover Pedro...")
s = time()
# Crossover parents and add to the new population.
pop_ = DT_crossover(pop1)
pop_ = evaluate_fitness(pop_)
evals_ = aux_evals(pop_)
invalids_ = aux_invalids(pop_)
e = time()
print("Crossover Pedro finished. Time: ", str(e-s))
print("Start Mutation GE...")
s = time()
# Mutate the new population.
pop3 = mutation(pop2)
pop3 = evaluate_fitness(pop3)
evals3 = aux_evals(pop3)
invalids3 = aux_invalids(pop3)
e = time()
print("Mutation GE finished. Time: ", str(e-s))
print("Start Mutation Pedro...")
s = time()
# Mutate the new population.
pop_3 = DT_mutation(pop_)
pop_3 = evaluate_fitness(pop_3)
evals_3 = aux_evals(pop_3)
invalids_3 = aux_invalids(pop_3)
e = time()
print("Mutation Pedro finished. Time: ", str(e-s))
# Replace the old population with the new population.
#pop4 = replacement(pop3, pop0)
#evals4 = aux_evals(pop4)
#invalids4 = aux_invalids(pop4)


