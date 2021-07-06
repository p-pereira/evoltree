# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:26:33 2020

@author: pedro
"""
from typing import List
import pandas as pd
from representation.individual import Individual
from sklearn.tree import _tree
import numpy as np

class evoltree(object):
    """
    evoltree object.
    
    """
    
    def __init__(self):
        self.params = {}
        self.population = []
        self.stats = {}
        self.fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series, pos_label: str, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None, pop : int =100, 
            gen : int =100, lamarck : bool =True, multicore: bool =True, 
            **extra_params):
        """
        Build a population of Evolutionary Decision Trees from the training set (X, y).
        It uses AUC and Tree complexity as metrics.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.
        y : pd.Series
            The target values (class labels) as integers or strings.
        pos_label : str
            Positive class label. Used to compute AUC.
        X_val : pd.DataFrame, optional
            Validation input samples, used to sort the population. The default is None.
        y_val : pd.Series, optional
            Validation target values (class values), used to sort the population. The default is None.
        pop : int, optional
            Number of Evolutionary Decision Trees in the population. The default is 100.
        gen : int, optional
            Number of generations (iterations) of training. The default is 100.
        lamarck : bool, optional
            If Lamarckian Evolutiona is used. The default is True.
        multicore : bool, optional
            If parallelism is used. The default is True.
        **extra_params : TYPE
            Extra parameters. For details, please check: https://github.com/PonyGE/PonyGE2/wiki/Evolutionary-Parameters.

        Returns
        -------
        self
            Fitted Evolutionary Decision Trees.

        """
        from .algorithm.parameters import params, set_params
        from .stats.stats import get_stats, stats
        from .operators.initialisation import initialisation
        from .fitness.evaluation import evaluate_fitness
        from tqdm import tqdm
        from multiprocessing import Pool
        from .utilities.algorithm.initialise_run import pool_init
        
        new_params = {'X_train': X, 'y_train': y,
                      'X_test': X_val, 'y_test': y_val,
                      'POPULATION_SIZE': pop, 'GENERATIONS': gen,
                      'LAMARCK': lamarck, 'MULTICORE': multicore,
                      'POS_LABEL': pos_label}
        params.update(new_params)
        
        param_list = list_params(pop, gen, lamarck, multicore, **extra_params)
        set_params(param_list)
        
        if params["MULTICORE"]:
            if "POOL" in params.keys():
                params["POOL"].close()
                params["POOL"] = None
            # initialize pool once, if mutlicore is enabled
            params['POOL'] = Pool(processes=params['CORES'], 
                                  initializer=pool_init,
                                  initargs=(params,))
        self.params.update(params)
        
        # Initialise population
        self.population = initialisation(params['POPULATION_SIZE'])
        # Evaluate initial population
        self.population = evaluate_fitness(self.population)
        stats['gen'] = 0
        # Generate statistics for run so far
        get_stats(self.population)
        population = self.population
        
        mlflow = get_mlflow(params['EXPERIMENT_NAME'])
        total_gens = params['GENERATIONS']+1
        range_generations = tqdm(range(1, total_gens))
        
        population = evolve(params, range_generations, mlflow, population)
        get_stats(population, end=True)
        store_pop(population)
        
        self.stats = stats
        self.population = population
        self.fitted = True
    
    def refit(self, gen: int):
        """
        Continues the training process for <gen> iterations.

        Parameters
        ----------
        gen : int
            Number of generations (iterations).

        Raises
        ------
        Exception
            If Evolutionary Decision Trees are not trained yet, please use evoltree.fit.

        Returns
        -------
        self
            Re-fitted Evolutionary Decision Trees.

        """
        if not self.fitted:
            raise Exception("evoltree needs to be fitted first. Use evoltree.fit")
        from .algorithm.parameters import params
        from .stats.stats import get_stats, stats
        from tqdm import tqdm
        
        population = self.population
        # Generate statistics for run so far
        stats['gen'] = params['GENERATIONS']
        get_stats(population)
        
        mlflow = get_mlflow(params['EXPERIMENT_NAME'])
        
        total_gens = params['GENERATIONS'] + 1 + gen
        range_generations = tqdm(range(params['GENERATIONS'] + 1, total_gens))
        population = evolve(params, range_generations, mlflow, 
                            population, refit=True)
        
        get_stats(population, end=True)
        store_pop(population)
        self.stats = stats
        self.population = population
    
    def fit_new_data(self, X, y, X_val=None, y_val=None, pop=100, gen=100, 
                     lamarck=True, multicore=True, **extra_params) -> None:
        """
        Fit Evolutionary Decision Trees to a new data set (X,Y), considering the same positive label.
        It uses the previous solutions as starting point.
        Function used for Online Learning environments.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.
        y : pd.Series
            The target values (class labels) as integers or strings.
        X_val : pd.DataFrame, optional
            Validation input samples, used to sort the population. The default is None.
        y_val : pd.Series, optional
            Validation target values (class values), used to sort the population. The default is None.
        pop : int, optional
            Number of Evolutionary Decision Trees in the population. The default is 100.
        gen : int, optional
            Number of generations (iterations) of training. The default is 100.
        lamarck : bool, optional
            If Lamarckian Evolutiona is used. The default is True.
        multicore : bool, optional
            If parallelism is used. The default is True.
        **extra_params : TYPE
            Extra parameters. For details, please check: https://github.com/PonyGE/PonyGE2/wiki/Evolutionary-Parameters.

        Raises
        ------
        Exception
            If Evolutionary Decision Trees are not trained yet, please use evoltree.fit.

        Returns
        -------
        self
            Evolutionary Decision Trees re-fitted to the new data set.

        """
        if not self.fitted:
            raise Exception("evoltree needs to be fitted first. Use evoltree.fit")
        from .algorithm.parameters import params, set_params
        from .stats.stats import get_stats, stats
        from tqdm import tqdm
        new_params = {'X_train': X, 'y_train': y,
                      'X_test': X_val, 'y_test': y_val,
                      'POPULATION_SIZE': pop, 'GENERATIONS': gen,
                      'LAMARCK': lamarck, 'MULTICORE': multicore}
        params.update(new_params)
        param_list = list_params(pop, gen, lamarck, multicore, **extra_params)
        set_params(param_list)
        
        self.params = params
        
        population = self.population
        
        mlflow = get_mlflow(params['EXPERIMENT_NAME'])
        
        total_gens = params['GENERATIONS']+1 + gen
        range_generations = tqdm(range(params['GENERATIONS']+1, total_gens))
        population = evolve(params, range_generations, mlflow, population)
        
        get_stats(population, end=True)
        store_pop(population)
        self.stats = stats
        self.population = population
    
    def predict(self, x: pd.DataFrame, mode: str="best") -> np.array:
        """
        Predict class probabilities of the input samples X.

        Parameters
        ----------
        x : pd.DataFrame
            The input samples.
        mode : str, optional
            Specifies which Evolutionar Decision Tree is used to perform predictions. 
            The default is "best". Possibilities:
                - "best": uses the tree with highest AUC estimated using validation data.
                - "simplest": uses the tree with lowest complexity.
                - "all": uses all the trees and returns all the predicted probabilities.
                - "balanced": chooses the tree with the greatest Euclidean Distance\
                    to the reference point (0, 1), where 0 is the worst possible AUC\
                        and 1 is the worst possible complexity (normalized values).

        Returns
        -------
        preds : np.array
            The class probabilities of the input samples.

        """
        if mode == "all":
            preds = [ind.predict(x) for ind in self.population]
        elif mode == "best":
            best = min(self.population, key=lambda x: x.fitness[0])
            preds = best.predict(x)
        elif mode == "simplest":
            simplest = min(self.population, key=lambda x: x.fitness[1])
            preds = simplest.predict(x)
        elif mode == "balanced":
            from math import log10
            min_y = log10(min(self.population, 
                              key=lambda x: x.fitness[1]).fitness[1])
            max_y = log10(max(self.population, 
                              key=lambda x: x.fitness[1]).fitness[1])
            # get individual with greater distance to point (0, 1)
            balanced = max(self.population,
                           key=lambda x: get_distance(x, min_y, max_y))
            preds = balanced.predict(x)
        
        return preds
    
    def evaluate_all(self, X_test: pd.DataFrame, y_test: pd.Series) -> List:
        """
        Evaluate all Evolutionary Decision Trees in terms of AUC and Complexity,\
            based on data set (X_test, y_test)

        Parameters
        ----------
        X_test : pd.DataFrame
            The test input samples..
        y_test : pd.Series
            The test target values (class labels) as integers or strings.

        Returns
        -------
        List
            List containing [AUC, Complexity] for each Evolutionary Decision Tree in the population.

        """
        import pandas as pd
        from .utilities.fitness.error_metric import AUC
        aucs = [-1*AUC(y_test, ind.predict(X_test)) for ind in self.population]
        nodes = [ind.fitness[1] for ind in self.population]
        ev = pd.DataFrame([aucs, nodes]).T
        ev.columns=['auc','node']
        ev2 = ev.groupby('auc').agg({'node':min}).reset_index().values
        return [ev2[:,0], ev2[:,1]]
    
    def __get_tree_complexity__(self, dt: _tree, columns) -> int:
        """
        Estimates the complexity of an sklearn Decision Tree.
        For advanced users only.

        Parameters
        ----------
        dt : _tree
            sklearn Decision Tree.
        columns : TYPE
            Data attributes.

        Returns
        -------
        int
            Tree complexity.

        """
        nodes = get_nodes_from_tree(dt, columns, self.params)
        return nodes
    
    def __get_randForest_complexity__(self, rf, columns) -> int:
        """
        Estimates the complexity of an sklearn Random Forest.
        For advanced users only.

        Parameters
        ----------
        rf : TYPE
            sklearn Random Forest object.
        columns : TYPE
            Data attributes.

        Returns
        -------
        TYPE
            Sum of Random Forest Trees complexities.

        """
        nodes = [get_nodes_from_tree(dt, columns, self.params) 
                 for dt in rf.estimators_]
        return sum(nodes)
    
    def load_offline_data(val=True) -> List:
        """
        Returns an example dataset [(X_train, Y_train),
                                    (X_test, Y_test)].
        If validation = True, returns:  [(X_train, Y_train),
                                         (X_validation, Y_validation),
                                         (X_test, Y_test)].
        Used for static environment (Offline Learning).

        Parameters
        ----------
        val : TYPE, optional
            If validation set is returned. The default is True.

        Returns
        -------
        List
            List of datasets in format: [(X_train, Y_train), (X_test, Y_test)].

        """
        import pandas as pd
        from os import path
        from sklearn.model_selection import train_test_split
        import pkg_resources
        DATA_PATH = pkg_resources.resource_filename('evoltree', 'data')
        dtr_filename = path.join(DATA_PATH, "example1_tr.csv")
        dts_filename = path.join(DATA_PATH, "example1_ts.csv")
        dtrain = pd.read_csv(dtr_filename, sep=";")
        dts = pd.read_csv(dts_filename, sep=";")
        
        if val:
            dtr, dval = train_test_split(dtrain, test_size=0.1, 
                                         random_state=1234, 
                                         stratify=dtrain['target'])
            dtr = dtr.reset_index(drop=True)
            dval = dval.reset_index(drop=True)
            return [dtr.drop('target', axis=1), dtr['target'],
                    dval.drop('target', axis=1), dval['target'],
                    dts.drop('target', axis=1), dts['target']]
        else:
            return [dtr.drop('target', axis=1), dtr['target'],
                    dts.drop('target', axis=1), dts['target']]
    
    def load_online_data(val: bool=True) -> List:
        """
        Returns two example datasets, ordered in time.
        The format the following, where 1 and 2 denote two different data sets:
            [(X_train1, Y_train1),
            (X_test1, Y_test1),
            (X_train2, Y_train2),
            (X_test2, Y_test2)].
        If validation = True, returns:  [(X_train1, Y_train1),
                                         (X_validation1, Y_validation1),
                                         (X_test1, Y_test1),
                                         (X_train2, Y_train2),
                                         (X_validation2, Y_validation2),
                                         (X_test2, Y_test2)].
        Parameters
        ----------
        val : bool, optional
            If validation set is returned. The default is True.

        Returns
        -------
        List
            List of datasets.

        """
        import pandas as pd
        from os import path
        from sklearn.model_selection import train_test_split
        import pkg_resources
        DATA_PATH = pkg_resources.resource_filename('evoltree', 'data')
        dtr_filename1 = path.join(DATA_PATH, "example1_tr.csv")
        dts_filename1 = path.join(DATA_PATH, "example1_ts.csv")
        dtrain1 = pd.read_csv(dtr_filename1, sep=";")
        dts1 = pd.read_csv(dts_filename1, sep=";")
        
        dtr_filename2 = path.join(DATA_PATH, "example2_tr.csv")
        dts_filename2 = path.join(DATA_PATH, "example2_ts.csv")
        dtrain2 = pd.read_csv(dtr_filename2, sep=";")
        dts2 = pd.read_csv(dts_filename2, sep=";")
        
        if val:
            dtr1, dval1 = train_test_split(dtrain1, test_size=0.1, 
                                           random_state=1234,
                                           stratify=dtrain1['target'])
            dtr1 = dtr1.reset_index(drop=True)
            dval1 = dval1.reset_index(drop=True)
            
            dtr2, dval2 = train_test_split(dtrain2, test_size=0.1,
                                           random_state=1234,
                                           stratify=dtrain2['target'])
            dtr2 = dtr2.reset_index(drop=True)
            dval2 = dval2.reset_index(drop=True)
            return [dtr1.drop('target', axis=1), dtr1['target'],
                    dval1.drop('target', axis=1), dval1['target'],
                    dts1.drop('target', axis=1), dts1['target'],
                    dtr2.drop('target', axis=1), dtr2['target'],
                    dval2.drop('target', axis=1), dval2['target'],
                    dts2.drop('target', axis=1), dts2['target']]
        else:
            return [dtr1.drop('target', axis=1), dtr1['target'],
                    dts1.drop('target', axis=1), dts1['target'],
                    dtr2.drop('target', axis=1), dtr2['target'],
                    dts2.drop('target', axis=1), dts2['target']]


def store_pop(population: List):
    """
    Stores the evolved population. For advanced users only.

    Parameters
    ----------
    population : List
        The population to be stored

    Returns
    -------
    None.

    """
    from os import path, getcwd, makedirs
    from .algorithm.parameters import params
    SEEDS_PATH = path.join('evoltree', 'seeds')
    makedirs(path.join(getcwd(), SEEDS_PATH, params['TARGET_SEED_FOLDER']),
             exist_ok=True)
    for cont, item in enumerate(population):
        if item.phenotype != None:
            fname = path.join(SEEDS_PATH, params['TARGET_SEED_FOLDER'],
                              "{0}.txt".format(str(cont)))
            with open(fname, 'w+', 
                      encoding="utf-8") as f:
                f.write("Phenotype:\n")
                f.write("%s\n" % item.phenotype)
                f.write("Genotype:\n")
                f.write("%s\n" % item.genome)
                f.write("Tree:\n")
                f.write("%s\n" % str(item.tree))
                f.write("Training fitness:\n")
                f.write("%s\n" % item.fitness)
                f.close()
    
def get_distance(ind: Individual, min_y: float, max_y: float) -> float:
    """
    Compute Euclidean Distance from a solution to the reference point (0, 1).
    Data is normalized, 0 is AUC and 1 is tree complexity.
    For advanced users only.

    Parameters
    ----------
    ind : Individual
        Evolutionary Decision Tree.
    min_y : float
        The normalized minimum complexity from the population.
    max_y : float
        The normalized maximum complexity from the population..

    Returns
    -------
    float
        The Euclidean Distance.

    """
    import math
    auc = ind.fitness[0] / -100 # auc (positive, from 0 to 1)
    comp = math.log10(ind.fitness[1]) #complexity
    # scale complexity to [0, 1]
    comp = (comp - min_y) / (max_y - min_y)
    # worst result: (0, 1)
    x = 0
    y = 1
    # get distance:
    dist = math.hypot(auc-x, comp-y)
    return dist
    
def list_params(pop: list, gen: int, lamarck: bool, multicore: bool, 
                **extra_params: dict) -> List:
    """
    Internal function to list execution parameters.
    For advanced users only.

    Parameters
    ----------
    pop : list
        List of individuals.
    gen : int
        Number of generations.
    lamarck : bool
        If Lamarckian Evolution is used.
    multicore : bool
        If parallelism is used.
    **extra_params : dict
        Extra parameters. For details, please check: https://github.com/PonyGE/PonyGE2/wiki/Evolutionary-Parameters.

    Returns
    -------
    param_list : List
        List of parameters.

    """
    param_list = []
    if 'population_size' not in extra_params.keys():
        param_list.append('--population_size={0}'.format(str(pop)))
    
    if 'generations' not in extra_params.keys():
        param_list.append('--generations={0}'.format(str(gen)))
    
    if multicore and 'multicore' not in extra_params.keys():
        param_list.append("--multicore")
    if lamarck and 'lamarck' not in extra_params.keys():
        param_list.append("--lamarck")
    
    for (key, val) in extra_params.items():
        if val == "True":
            param_list.append("--"+key)
        elif val=="False" or val=="":
            continue
        else:
            param_list.append("--{0}={1}".format(key, val))
    return param_list

def get_mlflow(experiment_name: str):
    """
    Internal function for the use of mlflow.
    For advanced users only.

    Parameters
    ----------
    experiment_name : str
        The mlflow experiment name.

    Returns
    -------
    mlflow : mlflow
        The mlflow execution to store metrics and parameters.

    """
    import mlflow
    from os import path
    URI = path.join("evoltree", "results", "mlruns")
    mlflow.set_tracking_uri(URI)
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    try:
        _ = mlflow.create_experiment(experiment_name)
    except:
        _ = client.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)
    return mlflow

def evolve(params: dict, range_generations: range, mlflow, population: list, 
           refit: bool=False) -> List:
    """
    Internal function for the evolutionary process.
    For advanced users only.

    Parameters
    ----------
    params : dict
        The dictionary with execution parameters..
    range_generations : range
        Range of generations.
    mlflow : mlflow
        The mlflow execution.
    population : list
        List of individuals to evolve.
    refit : bool, optional
        If the population is being re-trained. The default is False.

    Returns
    -------
    List
        Fitted population.

    """
    import numpy as np
    from .stats.stats import stats
    from .utilities.fitness.error_metric import AUC
    
    with mlflow.start_run():
        mlflow.log_param("REFIT", refit)
        for key, val in params.items():
            if key in ['X_train', 'y_train', 'X_test', 'y_test', 'POOL',
                       'BNF_GRAMMAR', 'SEED_INDIVIDUALS']:
                continue
            else:
                mlflow.log_param(key, val)
        # Traditional GE
        for generation in range_generations:
            stats['gen'] = generation
            
            population = params['STEP'](population)
            all_auc = [ind.fitness[0] for ind in population]
            all_nodes = [ind.fitness[1] for ind in population]
            mlflow.log_metrics(metrics={"1st ind AUC" : -min(all_auc),
                                        "1st ind NODES" : max(all_nodes),
                                        "last ind AUC" : -max(all_auc),
                                        "last ind NODES" : min(all_nodes),
                                        "mean AUC" : -np.mean(all_auc),
                                        "mean nodes" : np.mean(all_nodes)},
                               step=generation)
        val_aucs = [AUC(params['y_test'], 
                        ind.predict(params['X_test']))\
                    for ind in population]
        mlflow.log_metric("best val AUC", -min(val_aucs))
    return population

def get_nodes_from_tree(tree: _tree, feature_names: list, params: dict) -> int:
    """
    Returns a sklearn Decision Tree complexity, based on the GE grammar.
    For advanced users only.

    Parameters
    ----------
    tree : _tree
        sklearn Decision Tree.
    feature_names : list
        Data attributes.
    params : dict
        The execution parameters.

    Returns
    -------
    int
        Number of GE tree nodes.

    """
    from importlib import import_module
    from .algorithm.mapper import map_tree_from_genome
    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    def recurse(node, depth, rule=""):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            rule = rule + "np.where(x['{}'] <= {}, ".format(name, threshold)
            rule = recurse(tree_.children_left[node], depth + 1, rule)
            rule = rule + ", "
            rule = recurse(tree_.children_right[node], depth + 1, rule)
            rule = rule + ")"
            return rule
        else:
            prob = 1 - round(tree_.value[node][0][0] / 
                             (tree_.value[node][0][0] + 
                              tree_.value[node][0][1]), 3)
            rule = rule + "({})".format(prob)
            return rule
    
    rules = recurse(0, 1)
    i = import_module(params['LAMARCK_MAPPER'])
    genome = i.get_genome_from_dt_idf(rules)
    nodes = map_tree_from_genome(genome)[3]
    return nodes
