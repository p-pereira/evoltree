import numpy as np
np.seterr(all="raise")

from ...algorithm.parameters import params
from ...utilities.fitness.get_data import get_data
from ...utilities.fitness.optimize_constants import optimize_constants
from random import sample
from ..base_ff_classes.base_ff import base_ff
import numpy as np
import pandas as pd
import time

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

    def evaluate(self, ind, savePred=False, **kwargs):
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
            if params['N_SAMPLING'] > 0:
                N = params['N_SAMPLING']
                pos = np.where(y == 'Sale')[0]
                neg = np.where(y == 'NoSale')[0]
                randPos = sample(list(pos), N)
                randNeg = sample(list(neg), N)
                x = x.iloc[(randPos+randNeg),:]
                y = y.iloc[(randPos+randNeg)]
            
        elif dist == "test":
            # Set test datasets.
            x = self.test_in
            y = self.test_exp

        else:
            raise ValueError("Unknown dist: " + dist)

        """
        if params['OPTIMIZE_CONSTANTS']:
            from utilities.utils import remove_leading_zeros
            ind.phenotype = remove_leading_zeros(ind.phenotype)
            # if we are training, then optimize the constants by
            # gradient descent and save the resulting phenotype
            # string as ind.phenotype_with_c0123 (eg x[0] +
            # c[0] * x[1]**c[1]) and values for constants as
            # ind.opt_consts (eg (0.5, 0.7). Later, when testing,
            # use the saved string and constants to evaluate.
            if dist == "training":
            #    #return optimize_constants(x, y, ind)
                fitness, ind =  optimize_constants(x, y, ind)
                return fitness, ind
            else:
                try:
                    # this string has been created during training
                    phen = ind.phenotype_consec_consts
                    c = ind.opt_consts
                except:
                    phen = ind.phenotype
                # phen will refer to x (ie test_in), and possibly to c
                yhat = eval(phen)
                assert np.isrealobj(yhat)

                # let's always call the error function with the
                # true values first, the estimate second
                return params['ERROR_METRIC'](y, yhat), None

        else:"""
        # phenotype won't refer to C
        try:
            start = time.time()
            yhat = ind.predict(x)
            end = time.time()
        except SyntaxError: # invalid solutions due to leading zeros
            from ...utilities.utils import remove_leading_zeros
            ind.phenotype = remove_leading_zeros(ind.phenotype)
            from importlib import import_module
            i = import_module(params["LAMARCK_MAPPER"])
            ind.genotype = i.get_genome_from_dt_idf(ind.phenotype)
            start = time.time()
            yhat = ind.predict(x)
            end = time.time()
        if savePred:
            from ...utilities.fitness.error_metric import AUC
            auc_ = AUC(y, yhat)
            d = pd.DataFrame({'y': y, 'pred':yhat, 'auc':auc_},index=None)
            d.to_csv(params['FILE_PATH']+'/preds.csv', sep=";",index=False)
            f = open(params['FILE_PATH']+'/time_preds.txt', 'w')
            f.write(str((end-start)/len(y)))
            f.close()
            
        assert np.isrealobj(yhat)

        # let's always call the error function with the true
        # values first, the estimate second
        return params['ERROR_METRIC'](y, yhat)
            
            
