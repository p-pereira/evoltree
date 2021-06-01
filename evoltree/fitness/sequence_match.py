from ..algorithm.parameters import params
from ..fitness.base_ff_classes.base_ff import base_ff

import editdistance  # https://pypi.python.org/pypi/editdistance
import lzstring  # https://pypi.python.org/pypi/lzstring/
import dtw  # https://pypi.python.org/pypi/dtw

"""

This fitness function is for a sequence-match problem: we're given
an integer sequence target, say [0, 5, 0, 5, 0, 5], and we try to synthesize a
program (loops, if-statements, etc) which will *yield* that sequence,
one item at a time.

There are several components of the fitness:

1. concerning the program:
    i. length of the program (shorter is better)
    ii. compressibility of the program (non-compressible, ie DRY, is better)

2. concerning distance from the target:
    i. dynamic time warping distance from the program's output to the target
    (lower is better).
    ii. Levenshtein distance from the program's output to the target
    (lower is better).

"""


# available for use in synthesized programs
def succ(n, maxv=6):
    """
    Available for use in synthesized programs.
    
    :param n:
    :param maxv:
    :return:
    """
    
    return min(n+1, maxv)


def pred(n, minv=0):
    """
    Available for use in synthesized programs.
    
    :param n:
    :param minv:
    :return:
    """
    
    return max(n-1, minv)


def truncate(n, g):
    """
    the program will yield one item at a time, potentially forever. We only
    up to n items.
    
    :param n:
    :param g:
    :return:
    """
    
    for i in range(n):
        yield next(g)


def dist(t0, x0):
    """
    numerical difference, used as a component in DTW.
    
    :param t0:
    :param x0:
    :return:
    """
    
    return abs(t0 - x0)


def dtw_dist(s, t):
    """
    Dynamic time warping distance between two sequences.
    
    :param s:
    :param t:
    :return:
    """
    
    s = list(map(int, s))
    t = list(map(int, t))
    d, M, C, path = dtw.dtw(s, t, dist)
    
    return d


def lev_dist(s, t):
    """
    Levenshtein distance between two sequences, normalised by length of the
    target -- hence this is *asymmetric*, not really a distance. Don't
    normalise by length of the longer, because it would encourage evolution
    to create longer and longer sequences.
    
    :param s:
    :param t:
    :return:
    """
    
    return editdistance.eval(s, t) / len(s)


def compress(s):
    """
    Convert to a string and compress. lzstring is a special-purpose compressor,
    more suitable for short strings than typical compressors.
    
    :param s:
    :return:
    """
    
    s = ''.join(map(str, s))
    return lzstring.LZString().compress(s)


def compressibility(s):
    """
    Compressability is in [0, 1]. It's high when the compressed string
    is much shorter than the original.
    
    :param s:
    :return:
    """
    
    return 1 - len(compress(s)) / len(s)


def proglen(s):
    """
    Program length is measured in characters, but in order to keep the values
    in a similar range to that of compressibility, DTW and Levenshtein, we
    divide by 100. This is a bit arbitrary.
    
    :param s: A string of a program phenotype.
    :return: The length of the program divided by 100.
    """
    
    return len(s) / 100.0


class sequence_match(base_ff):
        
    def __init__(self):
        """
        Initilise class instance
        """
        # Initialise base fitness function class.
        super().__init__()
        
        # --target will be a sequence such as (0, 5, 0, 5)
        self.target = eval(params['TARGET'])
        
        # we assume --extra_parameters is a comma-separated kv sequence, eg:
        # "alpha=0.5, beta=0.5, gamma=0.5"
        # which we can pass to the dict() constructor
        extra_fit_params = eval("dict("+params['EXTRA_PARAMETERS']+")")
        self.alpha = extra_fit_params['alpha']
        self.beta = extra_fit_params['beta']
        self.gamma = extra_fit_params['gamma']

    def evaluate(self, ind, **kwargs):
        """
        ind.phenotype will be a string incl fn defns etc. when we exec it
        will create a value XXX_output_XXX, but we exec inside an empty dict
        for safety. But we put a couple of useful primitives in the dict too.
        
        :param ind:
        :return:
        """
        
        p, d = ind.phenotype, {'pred': pred, 'succ': succ}
        exec(p, d)

        # this is the program's output: a generator
        s = d['XXX_output_XXX']

        # Truncate the generator and convert to list
        s = list(truncate(len(self.target), s))
        
        # Set target
        t = self.target

        # various weightings of four aspects of our fitness. the formula is:
        # fitness = gamma * dist + (1 - gamma) * length
        # where dist = alpha * lev_dist(t, s) + (1 - alpha) * dtw_dist(t, s)
        # and length = beta * proglen(t) + (1 - beta) * compressibility(t)
        # but when any of alpha, beta and gamma is 0 or 1, we can save some
        # calculation:

        if self.gamma > 0.0:
            if self.alpha > 0.0:
                lev_dist_v = lev_dist(t, s)
            else:
                lev_dist_v = 0.0
            if self.alpha < 1.0:
                dtw_dist_v = dtw_dist(t, s)
            else:
                dtw_dist_v = 0.0
            dist_v = self.alpha * lev_dist_v + (1 - self.alpha) * dtw_dist_v
        else:
            dist_v = 0.0

        if self.gamma < 1.0:
            if self.beta > 0.0:
                proglen_v = proglen(p)
            else:
                proglen_v = 0.0
            if self.beta < 1.0:
                compressibility_v = compressibility(p)
            else:
                compressibility_v = 0.0
            length_v = self.beta * proglen_v + (1 - self.beta) * \
                                               compressibility_v
        else:
            length_v = 0.0

        return self.gamma * dist_v + (1 - self.gamma) * length_v


if __name__ == "__main__":
    # TODO write some tests here
    pass
