import numpy as np
import random
from random import sample
import re


from algorithm.mapper import map_genome_from_ind_DTRAW
from algorithm.mapper import map_tree_from_genome
from operators.lamarck import tree_to_code
from algorithm.mapper import mapper
from algorithm.parameters import params
from utilities.fitness.get_data import get_data
from utilities.fitness.math_functions import sigmoid, inverse, symmetric, reLu, leakyReLu, sqrt
from operators.lamarck import applyLamarckToPhen, applyLamarckToPhen2
import math


class Individual(object):
    """
    A GE individual.
    """

    def __init__(self, genome, ind_tree, map_ind=True):
        """
        Initialise an instance of the individual class (i.e. create a new
        individual).

        :param genome: An individual's genome.
        :param ind_tree: An individual's derivation tree, i.e. an instance
        of the representation.tree.Tree class.
        :param map_ind: A boolean flag that indicates whether or not an
        individual needs to be mapped.
        """

        if map_ind:
            # The individual needs to be mapped from the given input
            # parameters.
            self.phenotype, self.genome, self.tree, self.nodes, self.invalid, \
                self.depth, self.used_codons = mapper(genome, ind_tree)

        else:
            # The individual does not need to be mapped.
            self.genome, self.tree = genome, ind_tree

        self.fitness = params['FITNESS_FUNCTION'].default_fitness
        self.runtime_error = False
        self.name = None

    def __lt__(self, other):
        """
        Set the definition for comparison of two instances of the individual
        class by their fitness values. Allows for sorting/ordering of a
        population of individuals. Note that numpy NaN is used for invalid
        individuals and is used by some fitness functions as a default fitness.
        We implement a custom catch for these NaN values.

        :param other: Another instance of the individual class (i.e. another
        individual) with which to compare.
        :return: Whether or not the fitness of the current individual is
        greater than the comparison individual.
        """

        if np.isnan(self.fitness): return True
        elif np.isnan(other.fitness): return False
        else: return self.fitness < other.fitness if params['FITNESS_FUNCTION'].maximise else other.fitness < self.fitness

    def __le__(self, other):
        """
        Set the definition for comparison of two instances of the individual
        class by their fitness values. Allows for sorting/ordering of a
        population of individuals. Note that numpy NaN is used for invalid
        individuals and is used by some fitness functions as a default fitness.
        We implement a custom catch for these NaN values.

        :param other: Another instance of the individual class (i.e. another
        individual) with which to compare.
        :return: Whether or not the fitness of the current individual is
        greater than or equal to the comparison individual.
        """

        if np.isnan(self.fitness): return True
        elif np.isnan(other.fitness): return False
        else: return self.fitness <= other.fitness if params['FITNESS_FUNCTION'].maximise else other.fitness <= self.fitness

    def __str__(self):
        """
        Generates a string by which individuals can be identified. Useful
        for printing information about individuals.

        :return: A string describing the individual.
        """
        return ("Individual: " +
                str(self.phenotype) + "; " + str(self.fitness) + ";" + str(self.genome))

    def deep_copy(self):
        """
        Copy an individual and return a unique version of that individual.

        :return: A unique copy of the individual.
        """

        if not params['GENOME_OPERATIONS']:
            # Create a new unique copy of the tree.
            new_tree = self.tree.__copy__()

        else:
            new_tree = None

        # Create a copy of self by initialising a new individual.
        new_ind = Individual(self.genome.copy(), new_tree, map_ind=False)

        # Set new individual parameters (no need to map genome to new
        # individual).
        new_ind.phenotype, new_ind.invalid = self.phenotype, self.invalid
        new_ind.depth, new_ind.nodes = self.depth, self.nodes
        new_ind.used_codons = self.used_codons
        new_ind.runtime_error = self.runtime_error

        return new_ind

    def evaluate(self):
        """
        Evaluates phenotype in using the fitness function set in the params
        dictionary. For regression/classification problems, allows for
        evaluation on either training or test distributions. Sets fitness
        value.

        :return: Nothing unless multicore evaluation is being used. In that
        case, returns self.
        """
        # Evaluate fitness using specified fitness function.
        self.fitness = params['FITNESS_FUNCTION'](self)

        if params['MULTICORE']:
            return self
    
    def applyLamarck(self):
        """
        ....
        ....
        :param ind: an individual to be changed.
        :returns: The changed individual.
        """
        from sklearn import tree
        try:
            ind = self.deep_copy()
            #print(type(ind))
            phenotype = ind.phenotype
            # Check number of nodes (np.where)
            nrNodes = phenotype.count("np.where")
            #if nrNodes > 3: # it must have at least 3 nodes
            #print('I\'m in!')
            if nrNodes == 0: # Expressions without root node, only 1 leaf, are replaced by a traditional decision tree
                x, y, x_test, y_test = \
                get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])
                if params['N_SAMPLING'] > 0:
                    N = params['N_SAMPLING']
                    pos = np.where(y == 'Sale')[0]
                    neg = np.where(y == 'NoSale')[0]
                    randPos = sample(list(pos), N)
                    randNeg = sample(list(neg), N)
                    x = x.iloc[(randPos+randNeg),:]
                    y = y.iloc[(randPos+randNeg)]
                dt = tree.DecisionTreeClassifier()
                dt = dt.fit(x, y)
                # get the tree rules
                rules = tree_to_code(dt, x.columns.tolist() + ['target'])
                ind.phenotype = rules
                ind.evaluate()
                if hasattr(params['FITNESS_FUNCTION'], 'multi_objective'):
                    if ind.fitness[0] < self.fitness[0] or math.isnan(self.fitness[0]):
                        if params['DATA_TRANSF'] == 'IDF':
                            from algorithm.dt2.dt2_idf import get_genome_from_dt_idf
                            genome = get_genome_from_dt_idf(ind.phenotype)
                        elif params['DATA_TRANSF'] == 'RAW':
                            genome = map_genome_from_ind_DTRAW(ind.phenotype)
                        ind.genome = genome
                        mapped = map_tree_from_genome(genome)
                        ind.tree = mapped[2]
                        ind.nodes = mapped[3]
                        ind.evaluate()
                        return ind
                    else:
                        return self
                else:
                    if ind.fitness > self.fitness or math.isnan(self.fitness):
                        #print('It improved!!!')
                        # replace genome
                        if params['DATA_TRANSF'] == 'IDF':
                            from algorithm.dt2.dt2_idf import get_genome_from_dt_idf
                            genome = get_genome_from_dt_idf(ind.phenotype)
                        elif params['DATA_TRANSF'] == 'RAW':
                            genome = map_genome_from_ind_DTRAW(ind.phenotype)
                        ind.genome = genome
                        mapped = map_tree_from_genome(genome)
                        ind.tree = mapped[2]
                        ind.nodes = mapped[3]
                        return ind
                    else:
                        return self
            else:
                # select a random node (can't be the first)
                randNode = random.randint(0, nrNodes-1) # -1 due to the index starting at 0
                # get all nodes' positions
                allNodes = [node.start() for node in re.finditer('np.where', phenotype)]
                # get node position
                nodePosition = allNodes[randNode]    
                # get the expression
                #   get opened and closed brackets' positions
                openBrPos = [m.start() for m in re.finditer('\(', phenotype[nodePosition:])]
                closeBrPos = [m.start() for m in re.finditer('\)', phenotype[nodePosition:])]
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
                        break
                # expression to be replaced:
                replaceExp = phenotype[nodePosition:(position+nodePosition+1)]
                #   subtree leafs, to identify records that fall there
                toReplace = re.findall("\([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?\)", replaceExp)
                for val in toReplace:
                    replaceExp = replaceExp.replace(val, '100')
               # new expression with this values
                newExp = phenotype[0:nodePosition] + replaceExp + phenotype[(position+nodePosition+1):]
                # get data
                x, y, x_test, y_test = \
                get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])
                """
                if params['N_SAMPLING'] > 0:
                    N = params['N_SAMPLING']
                    pos = np.where(y == 'Sale')[0]
                    neg = np.where(y == 'NoSale')[0]
                    randPos = sample(list(pos), N)
                    randNeg = sample(list(neg), N)
                    x = x.iloc[(randPos+randNeg),:]
                    y = y.iloc[(randPos+randNeg)]
                    #print(x.shape)
                """
                # index of records that fall on the changed expression
                index = np.where(eval(newExp) == 100)[0]
                if len(index) > 0:
                    x = x.iloc[index,:]
                    y = y.iloc[index]
                    # train a traditional decision tree
                    dt = tree.DecisionTreeClassifier(max_depth=10)
                    dt = dt.fit(x, y)
                    # get the tree rules
                    rules = tree_to_code(dt, x.columns.tolist() + ['target'])
                    # add the rules to the old phenotype
                    newPhenotype = phenotype[0:nodePosition] + rules + phenotype[(position+nodePosition+1):]
                    #print(phenotype)
                    #print(newPhenotype)
                    
                    ind.phenotype = newPhenotype
                    ind.evaluate()
                    #print(ind.fitness)
                    #print(self.fitness)
                    if hasattr(params['FITNESS_FUNCTION'], 'multi_objective'):
                        if ind.fitness[0] < self.fitness[0] or math.isnan(self.fitness[0]):
                            if params['DATA_TRANSF'] == 'IDF':
                                #print(newPhenotype)
                                from algorithm.dt2.dt2_idf import get_genome_from_dt_idf
                                genome = get_genome_from_dt_idf(ind.phenotype)
                            elif params['DATA_TRANSF'] == 'RAW':
                                genome = map_genome_from_ind_DTRAW(newPhenotype)
                            ind.genome = genome
                            mapped = map_tree_from_genome(genome)
                            ind.tree = mapped[2]
                            ind.nodes = mapped[3]
                            ind.evaluate()
                            return ind
                        else:
                            return self
                    else:
                        if ind.fitness > self.fitness or math.isnan(self.fitness):
                            #print('It improved!!!')
                            # replace genome
                            if params['DATA_TRANSF'] == 'IDF':
                                #print(newPhenotype)
                                from algorithm.dt2.dt2_idf import get_genome_from_dt_idf
                                genome = get_genome_from_dt_idf(ind.phenotype)
                            elif params['DATA_TRANSF'] == 'RAW':
                                genome = map_genome_from_ind_DTRAW(newPhenotype)
                            ind.genome = genome
                            mapped = map_tree_from_genome(genome)
                            ind.tree = mapped[2]
                            ind.nodes = mapped[3]
                            return ind
                        else:
                            return self
                else:
                    return self
        except Exception as e:
            #print(e)
            return self
    
    def applyLamarckSYMB(self):
        try:
            ind = self.deep_copy()
            x, y, x_test, y_test = get_data(params['DATASET_TRAIN'], False)
            #yhat0 = eval(phenotype)
            #auc0 = auc_metric(y, yhat0)
            phenotype = ind.phenotype
            # Choose a random position and get 2 numerical values to optimize
            #   First, get the numeric values
            #numVals = re.findall("[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", phenotype)
            numVals = re.findall("[-]?\d*\.\d+|[-]?\d+", phenotype)
            #   Some expressions do not have numeric values, so they need to be introduced in a random position
            #   something like: sigmoid(a + b * expression)
            l = len(numVals)
            if l > 0: # in case there's already at least 2 numeric values...
                # select 2 random numeric values
                #vals = random.sample(numVals, 2)
                # replace the numeric values in the phenotype for 2 variables
                phenotype1 = phenotype
                #print(phenotype1, "1")
                #if l > 50:
                #    l = 50
                #    numVals = random.choices(numVals, k=l)
                for i in range(l):
                    phenotype1 = phenotype1.replace(numVals[i], "a[" + str(i) + "]", 1)
                numVals = [float(i) for i in numVals]
                #print(phenotype1, "2")
                # extract numbers from the string...
                #vals2 = [x.replace('(', '') for x in vals]
                #vals2 = [x.replace(')', '') for x in vals2]
                #vals2 = np.array(list(map(float, vals2)))
                #vals2 = [x * 1e-10 for x in vals2]
                
            else: # add a + b * <phenotype> at a random position
                # get '(' positions and choose a random
                parenth = [m.start() for m in re.finditer('\(', phenotype)]
                pos = random.sample(parenth, 1)[0]
                # insert a + b * ... in the phenotype
                phenotype1 = phenotype[0:pos+1] + ' a[1] * ' + phenotype[pos+1:-1] + " + a[0])"
                numVals = [0, 1]
            #print("num vals: ", len(numVals))
            phenotype3 = applyLamarckToPhen(phenotype1, numVals)
            #yhat1 = eval(phenotype3)
            #auc1 = auc_metric(y, yhat1)
            ind.phenotype = phenotype3
            ind.evaluate()
            #print(phenotype3)
            #print(ind.fitness)
            if ind.fitness > self.fitness:
                #print('Improved!', self.fitness, ind.fitness, phenotype3)
                if params["MODEL"] == "SYMB":
                    from algorithm.symb.symb import get_genome_from_symb_expression
                else:
                    from algorithm.net.net import get_genome_from_symb_expression
                #print(ind.phenotype)
                #print(self.phenotype)
                genome = get_genome_from_symb_expression(ind.phenotype)
                ind.genome = genome
                mapped = map_tree_from_genome(genome)
                ind.tree = mapped[2]
                ind.nodes = mapped[3]
                return ind
            else:
                #print("Didn't improve!  ", auc1, auc0, phenotype3)
                return self
        except Exception as e:
        #    print(e)
            return self
