# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:10:22 2020

@author: pedro
"""

from algorithm.parameters import set_params, params
from operators.initialisation import initialisation
from utilities.fitness.get_data import get_data
from utilities.fitness.math_functions import *
import scipy
import re

set_params('')
pop = initialisation(10)
ind = pop[6]

training_in, training_exp, test_in, test_exp = get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])
x = training_in
y = training_exp

s = ind.phenotype

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

n_consts = len(const_idxs)+len(const_idxs2)
init = [float(var[1:-1]) if '.' in var else int(var[1:-1]) for var in const_idxs+const_idxs2]

ind.phenotype_consec_consts = s

# Eval the phenotype.
f = eval("lambda x, c: " + s)

# Pre-load the error metric fitness function.
loss = params['ERROR_METRIC']

obj = lambda c: loss(y, f(x, c))


res = scipy.optimize.minimize(obj, init, method="L-BFGS-B")
res['x'] = [int(x) if (float(x)-int(x) == 0) else float(x) for x in res['x']]

