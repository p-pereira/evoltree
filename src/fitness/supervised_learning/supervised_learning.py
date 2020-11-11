import numpy as np
np.seterr(all="raise")
from random import sample

from algorithm.parameters import params
from utilities.fitness.get_data import get_data
from utilities.fitness.optimize_constants import optimize_constants

from fitness.base_ff_classes.base_ff import base_ff

from utilities.utils import treat_phenotype
from stats.stats import stats

class supervised_learning(base_ff):
    """
    Fitness function for supervised learning, ie regression and
    classification problems. Given a set of training or test data,
    returns the error between y (true labels) and yhat (estimated
    labels).

    We can pass in the error metric and the dataset via the params
    dictionary. Of error metrics, eg RMSE is suitable for regression,
    while F1-score, hinge-loss and others are suitable for
    classification.

    This is an abstract class which exists just to be subclassed:
    should not be instantiated.
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Get training and test data
        self.training_in, self.training_exp, self.test_in, self.test_exp = \
            get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])

        # Find number of variables.
        self.n_vars = np.shape(self.training_in)[0]

        # Regression/classification-style problems use training and test data.
        if params['DATASET_TEST']:
            self.training_test = True

    def evaluate(self, ind, **kwargs):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.

        :param ind: An individual to be evaluated.
        :param kwargs: An optional parameter for problems with training/test
        data. Specifies the distribution (i.e. training or test) upon which
        evaluation is to be performed.
        :return: The fitness of the evaluated individual.
        """
        dist = kwargs.get('dist', 'training')

        if dist == "training":
            # Set training datasets.
            x = self.training_in
            y = self.training_exp
            # If data sampling is used
            if params['N_SAMPLING'] == 'max':
                none = list(np.where(y == 'none')[0])
                very_low = list(np.where(y == 'verylow')[0])
                low = list(np.where(y == 'low')[0])
                medium = list(np.where(y == 'medium')[0])
                high = list(np.where(y == 'high')[0])
                
                M = max([len(very_low), len(low), len(medium), len(high)])
                
                randNone = sample(list(none), M)
                x = x.iloc[(randNone+very_low + low + 
                            medium + high),:]
                y = y.iloc[(randNone+very_low + low + 
                            medium + high)]
                
            elif params['N_SAMPLING'] > 0 and stats['gen'] % 10 != 0:
                N = params['N_SAMPLING']
                none = np.where(y == 'none')[0]
                very_low = np.where(y == 'verylow')[0]
                low = np.where(y == 'low')[0]
                medium = np.where(y == 'medium')[0]
                high = np.where(y == 'high')[0]
                
                if N > len(list(none)):
                    randNone = sample(list(none), len(list(none)))
                else:
                    randNone = sample(list(none), N)
                if N > len(list(very_low)):
                    randVery_low = sample(list(very_low), len(list(very_low)))
                else:
                    randVery_low = sample(list(very_low), N)
                if N > len(list(low)):
                    randLow = sample(list(low), len(list(low)))
                else:
                    randLow = sample(list(low), N)
                if N > len(list(medium)):
                    randMedium = sample(list(medium), len(list(medium)))
                else:
                    randMedium = sample(list(medium), N)
                if N > len(list(high)):
                    randhigh = sample(list(high), len(list(high)))
                else:
                    randhigh = sample(list(high), N)
                x = x.iloc[(randNone+randVery_low + randLow + 
                            randMedium + randhigh),:]
                y = y.iloc[(randNone+randVery_low + randLow + 
                            randMedium + randhigh)]
            
        elif dist == "test":
            # Set test datasets.
            x = self.test_in
            y = self.test_exp

        else:
            raise ValueError("Unknown dist: " + dist)

        if params['OPTIMIZE_CONSTANTS']:
            # if we are training, then optimize the constants by
            # gradient descent and save the resulting phenotype
            # string as ind.phenotype_with_c0123 (eg x[0] +
            # c[0] * x[1]**c[1]) and values for constants as
            # ind.opt_consts (eg (0.5, 0.7). Later, when testing,
            # use the saved string and constants to evaluate.
            if dist == "training":
                return optimize_constants(x, y, ind)

            else:
                # this string has been created during training
                phen = ind.phenotype_consec_consts
                c = ind.opt_consts
                # phen will refer to x (ie test_in), and possibly to c
                try: # NEW 19/06/2020: treat leading zeros problem
                    yhat = eval(phen)
                except:
                    phen = treat_phenotype(phen)
                    yhat = eval(phen)
                assert np.isrealobj(yhat)

                # let's always call the error function with the
                # true values first, the estimate second
                return params['ERROR_METRIC'](y, yhat)

        else:
            # phenotype won't refer to C
            phenotype = ind.phenotype
            try: # NEW 19/06/2020: treat leading zeros problem
                yhat = eval(phenotype)
                assert np.isrealobj(yhat)
    
                # let's always call the error function with the true
                # values first, the estimate second
                return params['ERROR_METRIC'](y, yhat)
            except:
                phenotype = treat_phenotype(phenotype)
                yhat = eval(phenotype)
                ind.phenotype = phenotype
            
                assert np.isrealobj(yhat)
    
                # let's always call the error function with the true
                # values first, the estimate second
                return params['ERROR_METRIC'](y, yhat)
