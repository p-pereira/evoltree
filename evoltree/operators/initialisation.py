from math import floor
from os import path, getcwd, listdir, makedirs
from random import shuffle, randint
from importlib import import_module
import logging

from ..algorithm.parameters import params
from ..representation import individual
from ..representation.derivation import generate_tree, pi_grow
from ..representation.individual import Individual
from ..representation.tree import Tree
from ..representation.latent_tree import latent_tree_random_ind
from ..utilities.representation.python_filter import python_filter


def initialisation(size):
    """
    Perform selection on a population in order to select a population of
    individuals for variation.
    
    :param size: The size of the required population.
    :return: A full population generated using the specified initialisation
    technique.
    """

    # Decrease initialised population size by the number of seed individuals
    # (if any) to ensure that the total initial population size does not exceed
    # the limit.
    size -= len(params['SEED_INDIVIDUALS'])

    # Initialise empty population.
    individuals = params['INITIALISATION'](size)

    # Add seed individuals (if any) to current population.
    individuals.extend(params['SEED_INDIVIDUALS'])

    return individuals
    

def sample_genome():
    """
    Generate a random genome, uniformly.
    
    :return: A randomly generated genome.
    """
    genome = [randint(0, params['CODON_SIZE']) for _ in
              range(params['INIT_GENOME_LENGTH'])]
    return genome


def uniform_genome(size):
    """
    Create a population of individuals by sampling genomes uniformly.

    :param size: The size of the required population.
    :return: A full population composed of randomly generated individuals.
    """

    return [individual.Individual(sample_genome(), None) for _ in range(size)]


def uniform_tree(size):
    """
    Create a population of individuals by generating random derivation trees.
     
    :param size: The size of the required population.
    :return: A full population composed of randomly generated individuals.
    """
    
    return [generate_ind_tree(params['MAX_TREE_DEPTH'],
                              "random") for _ in range(size)]
    

def seed_individuals(size):
    """
    Create a population of size where all individuals are copies of the same
    seeded individuals.
    
    :param size: The size of the required population.
    :return: A full population composed of the seeded individuals.
    """
    
    # Get total number of seed inds.
    no_seeds = len(params['SEED_INDIVIDUALS'])
    
    # Initialise empty population.
    individuals = []
    
    if no_seeds > 0:
        # A list of individuals has been specified as the seed.
        
        # Divide requested population size by the number of seeds.
        num_per_seed = floor(size/no_seeds)
        
        for ind in params['SEED_INDIVIDUALS']:
        
            if not isinstance(ind, individual.Individual):
                # The seed object is not a PonyGE individual.
                s = "operators.initialisation.seed_individuals\n" \
                    "Error: SEED_INDIVIDUALS instance is not a PonyGE " \
                    "individual."
                raise Exception(s)
            
            else:
                # Generate num_per_seed identical seed individuals.
                individuals.extend([ind.deep_copy() for _ in
                                    range(num_per_seed)])
    
        return individuals
    
    else:
        # No seed individual specified.
        s = "operators.initialisation.seed_individuals\n" \
            "Error: No seed individual specified for seed initialisation."
        raise Exception(s)
    

def rhh(size):
    """
    Create a population of size using ramped half and half (or sensible
    initialisation) and return.

    :param size: The size of the required population.
    :return: A full population of individuals.
    """

    # Calculate the range of depths to ramp individuals from.
    depths = range(params['BNF_GRAMMAR'].min_ramp + 1,
                   params['MAX_INIT_TREE_DEPTH']+1)
    population = []

    if size < 2:
        # If the population size is too small, can't use RHH initialisation.
        """logging.info("Population size too small for RHH initialisation."
                     "Returning randomly built trees.")"""
        if params['VERBOSE']:
            logging.info("Not enough individuals to be generated, ",
                         "returning randomly built trees.")
        return [individual.Individual(sample_genome(), None)
                for _ in range(size)]

    elif not depths:
        # If we have no depths to ramp from, then params['MAX_INIT_DEPTH'] is
        # set too low for the specified grammar.
        s = "operators.initialisation.rhh\n" \
            "Error: Maximum initialisation depth too low for specified " \
            "grammar."
        raise Exception(s)

    else:
        if size % 2:
            # Population size is odd, need an even population for RHH
            # initialisation.
            size += 1
            logging.info("Specified population size is odd, "
                         "RHH initialisation requires an even population size. "
                         "Incrementing population size by 1.")

        if size/2 < len(depths):
            # The population size is too small to fully cover all ramping
            # depths. Only ramp to the number of depths we can reach.
            depths = depths[:int(size/2)]

        # Calculate how many individuals are to be generated by each
        # initialisation method.
        times = int(floor((size/2)/len(depths)))
        remainder = int(size/2 - (times * len(depths)))

        # Iterate over depths.
        for depth in depths:
            # Iterate over number of required individuals per depth.
            for i in range(times):

                # Generate individual using "Grow"
                ind = generate_ind_tree(depth, "random")

                # Append individual to population
                population.append(ind)

                # Generate individual using "Full"
                ind = generate_ind_tree(depth, "full")

                # Append individual to population
                population.append(ind)

        if remainder:
            # The full "size" individuals were not generated. The population
            # will be completed with individuals of random depths.
            depths = list(depths)
            shuffle(depths)

        for i in range(remainder):
            depth = depths.pop()

            # Generate individual using "Grow"
            ind = generate_ind_tree(depth, "random")

            # Append individual to population
            population.append(ind)

            # Generate individual using "Full"
            ind = generate_ind_tree(depth, "full")

            # Append individual to population
            population.append(ind)

        return population


def PI_grow(size):
    """
    Create a population of size using Position Independent Grow and return.

    :param size: The size of the required population.
    :return: A full population of individuals.
    """

    # Calculate the range of depths to ramp individuals from.
    depths = range(params['BNF_GRAMMAR'].min_ramp + 1,
                   params['MAX_INIT_TREE_DEPTH']+1)
    population = []

    if size < 2:
        # If the population size is too small, can't use PI Grow
        # initialisation.
        if params['VERBOSE']:
            logging.info("Not enough individuals to be generated, ",
                         "returning randomly built trees.")
        return [individual.Individual(sample_genome(), None)
                for _ in range(size)]

    elif not depths:
        # If we have no depths to ramp from, then params['MAX_INIT_DEPTH'] is
        # set too low for the specified grammar.
        s = "operators.initialisation.PI_grow\n" \
            "Error: Maximum initialisation depth too low for specified " \
            "grammar."
        raise Exception(s)

    else:
        if size < len(depths):
            # The population size is too small to fully cover all ramping
            # depths. Only ramp to the number of depths we can reach.
            depths = depths[:int(size)]

        # Calculate how many individuals are to be generated by each
        # initialisation method.
        times = int(floor(size/len(depths)))
        remainder = int(size - (times * len(depths)))

        # Iterate over depths.
        for depth in depths:
            # Iterate over number of required individuals per depth.
            for i in range(times):

                # Generate individual using "Grow"
                ind = generate_PI_ind_tree(depth)

                # Append individual to population
                population.append(ind)

        if remainder:
            # The full "size" individuals were not generated. The population
            #  will be completed with individuals of random depths.
            depths = list(depths)
            shuffle(depths)

        for i in range(remainder):
            depth = depths.pop()

            # Generate individual using "Grow"
            ind = generate_PI_ind_tree(depth)

            # Append individual to population
            population.append(ind)

        return population


def generate_ind_tree(max_depth, method):
    """
    Generate an individual using a given subtree initialisation method.

    :param max_depth: The maximum depth for the initialised subtree.
    :param method: The method of subtree initialisation required.
    :return: A fully built individual.
    """

    # Initialise an instance of the tree class
    ind_tree = Tree(str(params['BNF_GRAMMAR'].start_rule["symbol"]), None)

    # Generate a tree
    genome, output, nodes, _, depth = generate_tree(ind_tree, [], [], method,
                                                    0, 0, 0, max_depth)

    # Get remaining individual information
    phenotype, invalid, used_cod = "".join(output), False, len(genome)

    if params['BNF_GRAMMAR'].python_mode:
        # Grammar contains python code

        phenotype = python_filter(phenotype)

    # Initialise individual
    ind = individual.Individual(genome, ind_tree, map_ind=False)

    # Set individual parameters
    ind.phenotype, ind.nodes = phenotype, nodes
    ind.depth, ind.used_codons, ind.invalid = depth, used_cod, invalid

    # Generate random tail for genome.
    ind.genome = genome + [randint(0, params['CODON_SIZE']) for
                           _ in range(int(ind.used_codons / 2))]

    return ind


def generate_PI_ind_tree(max_depth):
    """
    Generate an individual using a given Position Independent subtree
    initialisation method.

    :param max_depth: The maximum depth for the initialised subtree.
    :return: A fully built individual.
    """

    # Initialise an instance of the tree class
    ind_tree = Tree(str(params['BNF_GRAMMAR'].start_rule["symbol"]), None)

    # Generate a tree
    genome, output, nodes, depth = pi_grow(ind_tree, max_depth)

    # Get remaining individual information
    phenotype, invalid, used_cod = "".join(output), False, len(genome)

    if params['BNF_GRAMMAR'].python_mode:
        # Grammar contains python code

        phenotype = python_filter(phenotype)

    # Initialise individual
    ind = individual.Individual(genome, ind_tree, map_ind=False)

    # Set individual parameters
    ind.phenotype, ind.nodes = phenotype, nodes
    ind.depth, ind.used_codons, ind.invalid = depth, used_cod, invalid

    # Generate random tail for genome.
    ind.genome = genome + [randint(0, params['CODON_SIZE']) for
                           _ in range(int(ind.used_codons / 2))]

    return ind


def load_population(target,otherDir=getcwd()):
    """
    Given a target folder, read all files in the folder and load/parse
    solutions found in each file.
    
    :param target: A target folder stored in the "seeds" folder.
    :return: A list of all parsed individuals stored in the target folder.
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
    
    path_2 = path.join(path_1, target)

    if not path.isdir(path_2):
        # Target folder does not exist.
        makedirs(path_2)
        """
        s = "scripts.seed_PonyGE2.load_population\n" \
            "Error: target folder " + target + \
            " does not exist in seeds directory."
        raise Exception(s)
        """
    
    # Get list of all target individuals in the target folder.
    target_inds = [i for i in listdir(path_2) if i.endswith(".txt")]
       
    # Initialize empty list for seed individuals.
    seed_inds = []

    for ind in target_inds:
        # Loop over all target individuals.

        # Get full file path.
        file_name = path.join(path_2, ind)

        # Initialise None data for ind info.
        genotype, phenotype, fitness = None, None, None
        
        if params["VERBOSE"]:
            logging.info("Loading seeds population...")
        
        # Open file.
        with open(file_name, "r") as f:
            
            # Read file.
            raw_content = f.read()
            
            # Read file.
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
            i = import_module(params['LAMARCK_MAPPER'])
            genotype = i.get_genome_from_dt_idf(phenotype)
            # Generate individual from genome.
            ind = Individual(genotype, None)
        # Set individual's fitness
        if fitness and not hasattr(params['FITNESS_FUNCTION'], 'multi_objective'):
            ind.fitness = float(fitness)
        # Add new ind to the list of seed individuals.
        seed_inds.append(ind)
    
    if params["VERBOSE"]:
            logging.info("Population loaded.")

    return seed_inds


def LTGE_initialisation(size):
    """Initialise a population in the LTGE representation."""

    pop = []
    for _ in range(size):

        # Random genotype
        g, ph = latent_tree_random_ind(params['BNF_GRAMMAR'],
                                       params['MAX_TREE_DEPTH'])

        # wrap up in an Individual and fix up various Individual attributes
        ind = individual.Individual(g, None, False)

        ind.phenotype = ph

        # number of nodes is the number of decisions in the genome
        ind.nodes = ind.used_codons = len(g)

        # each key is the length of a path from root
        ind.depth = max(len(k) for k in g)

        # in LTGE there are no invalid individuals
        ind.invalid = False

        pop.append(ind)
    return pop

# Set ramping attributes for ramped initialisers.
PI_grow.ramping = True
rhh.ramping = True
