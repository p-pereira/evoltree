import scipy

import re

from algorithm.parameters import params
from utilities.fitness.math_functions import *


def optimize_constants(x, y, ind):
    """
    Use gradient descent to search for values for the constants in
    ind.phenotype which minimise loss.
    
    :param x: Input (an array of x values).
    :param y: Expected output (expected y values for given inputs).
    :param ind: A GE individual.
    :return: The value of the error metric at those values
    """

    # Save the old phenotype (has c[0] etc, but may not be consecutive from
    # zero)
    ind.phenotype_original = ind.phenotype

    # Parse the phenotype to make the constants consecutive.
    s, n_consts, init = make_consts_consecutive(ind.phenotype)
    
    # Create new consecutive constant attribute for individual.
    ind.phenotype_consec_consts = s

    # Eval the phenotype.
    f = eval("lambda x, c: " + s)

    # Pre-load the error metric fitness function.
    loss = params['ERROR_METRIC']

    if n_consts == 0:
        # ind doesn't refer to c: no need to optimize
        c = []
        fitness = loss(y, f(x, c))
        ind.opt_consts = c
        return fitness

    obj = lambda c: loss(y, f(x, c))
    # obj is now a function of c only for L-BFGS-B. Using 0 as the init seems a
    # reasonable choice. But for scipy.curve_fit we might use [1.0] * n_consts.
    # Maybe other minimizers do better with some other choices? There are other
    # methods to try out. document
    #init = [1.0] * n_consts
    
    res = scipy.optimize.minimize(obj, init, method="L-BFGS-B")
    
    # the result is accessed like a dict
    ind.opt_consts = [int(x) if (float(x)-int(x) == 0) else float(x) for x in res['x']]  # the optimum values of the constants

    # the most useful form of the phenotype: c[0], c[1] etc replaced
    # with actual numbers, so can be eval'd directly
    ind.phenotype = replace_consts_with_values(s, ind.opt_consts)

    return res['fun']


def make_consts_consecutive(s):
    """
    The given phenotype will have zero or more occurrences of each const c[0],
    c[1], etc. But eg it might have c[7], but no c[0]. We need to remap, eg:
        7 -> 0
        9 -> 1
    so that we just have c[0], c[1], etc.
    
    :param s: A given phenotype string.
    :return: The phenotype string but with consecutive constants.
    """
    
    ### NEW 06-12-2020: constants are not in the same format of original ponyge2 implementation.
    # Therefore, we replace constants by 'c[k]', with k in [0, len(constants)-1].
    # Later, 'c[k]' is replaced by new values, got from optimization method.
    # Since we have 2 types of constants (comparison and probabilities) represented
    # differently, we'll replace them seperately.
    #p = r"c\[(\d+)\]"
    p1 = r' \d+(?:\.\d+)?,' # new pattern for comparison constants
    # find the consts, extract idxs as ints, unique-ify and sort
    #const_idxs = sorted(map(int, set(re.findall(p, s))))
    const_idxs = re.findall(p1, s) # NEW: this now contains the constants.
    cont = 0
    for k, j in enumerate(const_idxs):
        ci = " c[%d]," % k
        #cj = "c[%d]" % j
        # NEW: we replace 1st ocurrence of constant by 'c[k]'
        s = s.replace(j, ci, 1)
        cont+=1
    
    p2 = r'\(\d+(?:\.\d+)?\)' # new pattern for probability constants
    # find the consts, extract idxs as ints, unique-ify and sort
    #const_idxs = sorted(map(int, set(re.findall(p, s))))
    const_idxs2 = re.findall(p2, s) # NEW: this now contains the constants.
    for k, j in enumerate(const_idxs2):
        ci = "(c[%d])" % (k+cont)
        #cj = "c[%d]" % j
        # NEW: we replace 1st ocurrence of constant by 'c[k]'
        s = s.replace(j, ci, 1)
    
    init_guess = [float(var[1:-1]) if '.' in var else int(var[1:-1]) for var in const_idxs+const_idxs2]
    
    return s, len(const_idxs)+len(const_idxs2), init_guess


def replace_consts_with_values(s, c):
    """
    Replace the constants in a given string s with the values in a list c.
    
    :param s: A given phenotype string.
    :param c: A list of values which will replace the constants in the
    phenotype string.
    :return: The phenotype string with the constants replaced.
    """
    
    for i in range(len(c)):
        s = s.replace("c[%d]" % i, str(c[i]))
    
    return s
