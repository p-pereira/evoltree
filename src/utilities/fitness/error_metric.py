import warnings

import numpy as np
from sklearn.metrics.classification import f1_score as sklearn_f1_score
from sklearn.metrics import roc_auc_score


def mae(y, yhat):
    """
    Calculate mean absolute error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean absolute error.
    """

    return np.mean(np.abs(y - yhat))

# Set maximise attribute for mae error metric.
mae.maximise = False


def rmse(y, yhat):
    """
    Calculate root mean square error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The root mean square error.
    """

    return np.sqrt(np.mean(np.square(y - yhat)))

# Set maximise attribute for rmse error metric.
rmse.maximise = False


def mse(y, yhat):
    """
    Calculate mean square error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean square error.
    """

    return np.mean(np.square(y - yhat))

# Set maximise attribute for mse error metric.
mse.maximise = False


def hinge(y, yhat):
    """
    Hinge loss is a suitable loss function for classification.  Here y is
    the true values (-1 and 1) and yhat is the "raw" output of the individual,
    ie a real value. The classifier will use sign(yhat) as its prediction.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The hinge loss.
    """

    # NB not np.max. maximum does element-wise max
    return np.sum(np.maximum(0, 1 - y * yhat))

# Set maximise attribute for hinge error metric.
hinge.maximise = False


def f1_score(y, yhat):
    """
    The F_1 score is a metric for classification which tries to balance
    precision and recall, ie both true positives and true negatives.
    For F_1 score higher is better.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The f1 score.
    """
    # if phen is a constant, eg 0.001 (doesn't refer to x), then yhat
    # will be a constant. that will break f1_score. so convert to a
    # constant array.
    if not isinstance(yhat, np.ndarray) or len(yhat.shape) < 1:
        yhat = np.ones_like(y) * yhat

    # convert real values to boolean with a zero threshold
    yhat = (yhat > 0)
    with warnings.catch_warnings():
        # if we predict the same value for all samples (trivial
        # individuals will do so as described above) then f-score is
        # undefined, and sklearn will give a runtime warning and
        # return 0. We can ignore that warning and happily return 0.
        warnings.simplefilter("ignore")
        return sklearn_f1_score(y, yhat, average="macro")
# Set maximise attribute for f1_score error metric.
f1_score.maximise = True

def f1_score2(y, yhat):
    """
    The F_1 score is a metric for classification which tries to balance
    precision and recall, ie both true positives and true negatives.
    For F_1 score higher is better.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The f1 score.
    """
    f1s = sklearn_f1_score(y, yhat, average="macro")
    return -100 * f1s
# Set maximise attribute for f1_score error metric.
f1_score2.maximise = False

def Hamming_error(y, yhat):
    """
    The number of mismatches between y and yhat. Suitable
    for Boolean problems and if-else classifier problems.
    Assumes both y and yhat are binary or integer-valued.
    """
    return np.sum(y != yhat)
Hamming_error.maximise = False

def auc_metric(y, yhat):
    #print(yhat)
    #print(y)
    """
    Calculate the Area Under the Curve (AUC).

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean absolute error.
    """
    ### NEW ###
    # Replaces NaNs or infinite values
    yhat = np.nan_to_num(yhat)
    #if type(yhat) == np.float64:
    #    yhat = np.repeat(yhat, len(y))
    try:
        auc = roc_auc_score(y, yhat)*100
        return(auc)
    except Exception as e:
        #print(e)
        auc = 0
        return(auc)

    #print(auc)
    #return(auc)
# Set maximise attribute for auc error metric.
auc_metric.maximise = True
#auc_metric.default_fitness = 0

def auc_metric2(y, yhat):
    #print(yhat)
    #print(y)
    """
    Calculate the Area Under the Curve (AUC).

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean absolute error.
    """
    ### NEW ###
    # Replaces NaNs or infinite values
    yhat = np.nan_to_num(yhat)
    #if type(yhat) == np.float64:
    #    yhat = np.repeat(yhat, len(y))
    
    auc = roc_auc_score(y, yhat)*100
    return(-auc)

    #print(auc)
    #return(auc)
# Set maximise attribute for auc error metric.
auc_metric2.maximise = False