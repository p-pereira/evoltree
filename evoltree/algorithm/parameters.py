from multiprocessing import cpu_count
from os import path, makedirs, getcwd
from socket import gethostname
import pkg_resources

hostname = gethostname().split('.')
machine_name = hostname[0]


"""Algorithm parameters"""
params = {
        # Set default step and search loop functions
        'SEARCH_LOOP': 'search_loop',
        'STEP': 'step',

        # Evolutionary Parameters
        'POPULATION_SIZE': 100,
        'GENERATIONS': 100,
        'HILL_CLIMBING_HISTORY': 100000,
        'SCHC_COUNT_METHOD': "count_all",

        # Set optional experiment name
        'EXPERIMENT_NAME': "evoltree",
        # Set default number of runs to be done.
        # ONLY USED WITH EXPERIMENT MANAGER.
        'RUNS': 1,

        # Class of problem
        'FITNESS_FUNCTION': "supervised_learning.supervised_learning, minimise_nodes", #"supervised_learning.classification",#

        # Select problem dataset
        'DATASET_TRAIN': "",
        'DATASET_TEST': "",
        'DATASET_DELIMITER': ";",

        # Set grammar file
        # if '' or 'auto', it is automatically created based on data
        'GRAMMAR_FILE': "auto",

        # Set the number of depths permutations are calculated for
        # (starting from the minimum path of the grammar).
        # Mainly for use with the grammar analyser script.
        'PERMUTATION_RAMPS': 5,

        # Select error metric
        'ERROR_METRIC': "AUC",

        # Optimise constants in the supervised_learning fitness function.
        # TODO - NEW 06-12-2020: it's not completely working!
        'OPTIMIZE_CONSTANTS': False,

        # Specify target for target problems
        'TARGET': "target",

        # Set max sizes of individuals
        'MAX_TREE_DEPTH': 50,  # SET TO 90 DUE TO PYTHON EVAL() STACK LIMIT.
                               # INCREASE AT YOUR OWN RISK.
        'MAX_TREE_NODES': None,
        'CODON_SIZE': 300,
        'MAX_GENOME_LENGTH': None,
        'MAX_WRAPS': 0,

        # INITIALISATION
        # Set initialisation operator.
        'INITIALISATION': "operators.initialisation.PI_grow",
        # Set the maximum geneome length for initialisation.
        'INIT_GENOME_LENGTH': 2000,#100,
        # Set the maximum tree depth for initialisation.
        'MAX_INIT_TREE_DEPTH': 20,#20,
        # Set the minimum tree depth for initialisation.
        'MIN_INIT_TREE_DEPTH': None,

        # SELECTION
        # Set selection operator.
        'SELECTION': "operators.selection.nsga2_selection",#"operators.selection.truncation",#
        # For tournament selection
        'TOURNAMENT_SIZE': 2,
        # For truncation selection
        'SELECTION_PROPORTION': 0.5,#0.25,
        # Allow for selection of invalid individuals during selection process.
        'INVALID_SELECTION': False,

        # OPERATOR OPTIONS
        # Boolean flag for selecting whether or not mutation is confined to
        # within the used portion of the genome. Default set to True.
        'WITHIN_USED': True,

        # CROSSOVER
        # Set crossover operator.
        'CROSSOVER': "operators.crossover.subtree",#"operators.crossover.variable_onepoint",#
        # Set crossover probability.
        'CROSSOVER_PROBABILITY': 0.75,# 0.64,
        # Prevents crossover from generating invalids.
        'NO_CROSSOVER_INVALIDS': True,

        # MUTATION
        # Set mutation operator.
        'MUTATION': "operators.mutation.subtree",#"operators.mutation.int_flip_per_ind",
        # Set mutation probability (None defaults to 1 over the length of
        # the genome for each codon)
        'MUTATION_PROBABILITY': None,
        # Set number of mutation events
        'MUTATION_EVENTS': 1,
        # Prevents mutation from generating invalids.
        'NO_MUTATION_INVALIDS': True, #True, #

        # REPLACEMENT
        # Set replacement operator.
        'REPLACEMENT': "operators.replacement.nsga2_replacement",#"operators.replacement.generational",#
        # Set elite size.
        'ELITE_SIZE': 30,

        # DEBUGGING
        # Use this to turn on debugging mode. This mode doesn't write any files
        # and should be used when you want to test new methods.
        'DEBUG': False,

        # PRINTING
        # Use this to print out basic statistics for each generation to the
        # command line.
        'VERBOSE': False,
        # Use this to prevent anything being printed to the command line.
        'SILENT': True,

        # SAVING
        # Save the phenotype of the best individual from each generation. Can
        # generate a lot of files. DEBUG must be False.
        'SAVE_ALL': True,
        # Save a plot of the evolution of the best fitness result for each
        # generation.
        'SAVE_PLOTS': True,

        # MULTIPROCESSING
        # Multi-core parallel processing of phenotype evaluations.
        'MULTICORE': False,
        # Set the number of cpus to be used for multiprocessing
        'CORES': cpu_count(),

        # STATE SAVING/LOADING
        # Save the state of the evolutionary run every generation. You can
        # specify how often you want to save the state with SAVE_STATE_STEP.
        'SAVE_STATE': False,
        # Specify how often the state of the current evolutionary run is
        # saved (i.e. every n-th generation). Requires int value.
        'SAVE_STATE_STEP': 1,
        # Load an evolutionary run from a saved state. You must specify the
        # full file path to the desired state file. Note that state files have
        # no file type.
        'LOAD_STATE': None,

        # SEEDING
        # Specify a list of PonyGE2 individuals with which to seed the initial
        # population.
        'SEED_INDIVIDUALS': [],
        # Specify a target seed folder in the 'seeds' directory that contains a
        # population of individuals with which to seed a run.
        'TARGET_SEED_FOLDER': "",
        # Set a target phenotype string for reverse mapping into a GE
        # individual
        'REVERSE_MAPPING_TARGET': None,
        # Set Random Seed for all Random Number Generators to be used by
        # PonyGE2, including the standard Python RNG and the NumPy RNG.
        'RANDOM_SEED': 1234,

        # CACHING
        # The cache tracks unique individuals across evolution by saving a
        # string of each phenotype in a big list of all phenotypes. Saves all
        # fitness information on each individual. Gives you an idea of how much
        # repetition is in standard GE/GP.
        'CACHE': True,
        # Uses the cache to look up the fitness of duplicate individuals. CACHE
        # must be set to True if you want to use this.
        'LOOKUP_FITNESS': False,
        # Uses the cache to give a bad fitness to duplicate individuals. CACHE
        # must be True if you want to use this (obviously)
        'LOOKUP_BAD_FITNESS': False,
        # Removes duplicate individuals from the population by replacing them
        # with mutated versions of the original individual. Hopefully this will
        # encourage diversity in the population.
        'MUTATE_DUPLICATES': True,

        # MULTIAGENT Parameters
        # True or False for Multiagent
        'MULTIAGENT': False,
        # Agent Size. Number of agents having their own copy of genetic material
        'AGENT_SIZE': 100,
        # Interaction Probablity. How frequently the agents can interaction with each other
        'INTERACTION_PROBABILITY': 0.5,
        
        # OTHER
        # Set machine name (useful for doing multiple runs)
        'MACHINE': machine_name,
        
        # N value for data sampling
        # if a value is assigned, N*2 random records will be used to train the model
        'N_SAMPLING': 1000,

        # Folder name to store the results
        # if None, timestamp is used
        'FOLDER_NAME': "",
        
        # If LAMARCKIAN approach is used or not
        # True or False
        'LAMARCK': False,
        ### NEW 23-11-2020: probability of applying Lamarck
        # Set lamarck probability.
        'LAMARCK_PROBABILITY': 0.5,
        ### NEW 13-11-2020: parameter used to create the mapper automatically
        # Lamarck special mapper operator (created automatically if '' or 'auto')
        'LAMARCK_MAPPER' : 'auto',
        
        # Save initial and final populations: only for testing purposes
        # True or False
        'SAVE_POP': True,
        
        ### NEW 29-11-2020: dataset can be passed as argument
        'X_train' : None,
        'y_train' : None,
        'X_test' : None,
        'y_test' : None,
        ### New 01-06-2021: positive class should be passed for computing AUC
        'POS_LABEL': "",
        
}

def load_params(file_name):
    """
    Load in a params text file and set the params dictionary directly.

    :param file_name: The name/location of a parameters file.
    :return: Nothing.
    """

    try:
        open(file_name, "r")
    except FileNotFoundError:
        s = "algorithm.paremeters.load_params\n" \
            "Error: Parameters file not found.\n" \
            "       Ensure file extension is specified, e.g. 'regression.txt'."
        raise Exception(s)

    with open(file_name, 'r') as parameters:
        # Read the whole parameters file.
        content = parameters.readlines()

        for line in [l for l in content if not l.startswith("#")]:

            # Parameters files are parsed by finding the first instance of a
            # colon.
            split = line.find(":")

            # Everything to the left of the colon is the parameter key,
            # everything to the right is the parameter value.
            key, value = line[:split], line[split+1:].strip()

            # Evaluate parameters.
            try:
                value = eval(value)

            except:
                # We can't evaluate, leave value as a string.
                pass

            # Set parameter
            params[key] = value
        

def set_params(command_line_args, create_files=True):
    """
    This function parses all command line arguments specified by the user.
    If certain parameters are not set then defaults are used (e.g. random
    seeds, elite size). Sets the correct imports given command line
    arguments. Sets correct grammar file and fitness function. Also
    initialises save folders and tracker lists in utilities.trackers.

    :param command_line_args: Command line arguments specified by the user.
    :return: Nothing.
    """

    from ..utilities.algorithm.initialise_run import initialise_run_params
    from ..utilities.algorithm.initialise_run import set_param_imports
    from ..utilities.fitness.math_functions import return_one_percent
    from ..utilities.algorithm.command_line_parser import parse_cmd_args
    from ..utilities.stats import trackers, clean_stats
    from ..representation import grammar

    cmd_args, unknown = parse_cmd_args(command_line_args)

    if unknown:
        # We currently do not parse unknown parameters. Raise error.
        s = "algorithm.parameters.set_params\nError: " \
            "unknown parameters: %s\nYou may wish to check the spelling, " \
            "add code to recognise this parameter, or use " \
            "--extra_parameters" % str(unknown)
        raise Exception(s)

    # LOAD PARAMETERS FILE
    # NOTE that the parameters file overwrites all previously set parameters.
    if 'PARAMETERS' in cmd_args:
        load_params(path.join("parameters", cmd_args['PARAMETERS']))

    # Join original params dictionary with command line specified arguments.
    # NOTE that command line arguments overwrite all previously set parameters.
    params.update(cmd_args)
    
    ### NEW 29-11-2020: dataset is loaded once only!
    if params["X_train"] is None or params["y_test"] is None:
        if params["DATASET_TRAIN"] == "" or params["TARGET"] == "":
            s = "algorithm.parameters.set_params\nError: " \
            "dataset or target was not provided!" \
            "Specify dataset path and target using 'DATASET_TRAIN' and 'TARGET' parameters, respectively."\
            "Alternatively, pass the data directly to 'X_train' and 'y_train' parameters."
            raise Exception(s)
        else:
            from ..utilities.fitness.get_data import get_Xy_train_test_separate as get_d
            train_set = path.join("datasets", params["DATASET_TRAIN"])
            if params["DATASET_TEST"] != "":
                # Get the path to the testing dataset.
                test_set = path.join("datasets", params["DATASET_TEST"])
            else:
                # There is no testing dataset used.
                test_set = None
            (params["X_train"], params["y_train"],
             params["X_test"], params["y_test"]) = get_d(train_set, 
                                                         test_set, 
                                                         skip_header=1)
    
    if params['EXPERIMENT_NAME'] == '':
        params['EXPERIMENT_NAME'] = "evoltree"
    if params['FOLDER_NAME'] == '':
        import time
        params['FOLDER_NAME'] = "{0}".format(time.strftime("%Y%m%d%H%M%S"))
        

    if params['LOAD_STATE']:
        # Load run from state.
        from ..utilities.algorithm.state import load_state

        # Load in state information.
        individuals = load_state(params['LOAD_STATE'])

        # Set correct search loop.
        from ..algorithm.search_loop import search_loop_from_state
        params['SEARCH_LOOP'] = search_loop_from_state

        # Set population.
        setattr(trackers, "state_individuals", individuals)

    else:
        if isinstance(params['REPLACEMENT'], str):
            if params['REPLACEMENT'].split(".")[-1] == "steady_state":
                # Set steady state step and replacement.
                params['STEP'] = "steady_state_step"
                params['GENERATION_SIZE'] = 2
    
            else:
                # Elite size is set to either 1 or 1% of the population size,
                # whichever is bigger if no elite size is previously set.
                if params['ELITE_SIZE'] is None:
                    params['ELITE_SIZE'] = return_one_percent(1, params[
                        'POPULATION_SIZE'])
    
                # Set the size of a generation
                params['GENERATION_SIZE'] = params['POPULATION_SIZE'] - \
                                            params['ELITE_SIZE']

        # Initialise run lists and folders before we set imports.r
        initialise_run_params(create_files)

        # Set correct param imports for specified function options, including
        # error metrics and fitness functions.
        set_param_imports()

        # Clean the stats dict to remove unused stats.
        clean_stats.clean_stats()

        try:
            # Set GENOME_OPERATIONS automatically for faster linear operations.
            if (params['CROSSOVER'].representation == "subtree" or
                params['MUTATION'].representation == "subtree"):
                params['GENOME_OPERATIONS'] = False
            else:
                params['GENOME_OPERATIONS'] = True
        except:
            pass

        # Ensure correct operators are used if multiple fitness functions used.
        if hasattr(params['FITNESS_FUNCTION'], 'multi_objective'):

            # Check that multi-objective compatible selection is specified.
            if not hasattr(params['SELECTION'], "multi_objective"):
                s = "algorithm.parameters.set_params\n" \
                    "Error: multi-objective compatible selection " \
                    "operator not specified for use with multiple " \
                    "fitness functions."
                raise Exception(s)

            if not hasattr(params['REPLACEMENT'], "multi_objective"):

                # Check that multi-objective compatible replacement is
                # specified.
                if not hasattr(params['REPLACEMENT'], "multi_objective"):
                    s = "algorithm.parameters.set_params\n" \
                        "Error: multi-objective compatible replacement " \
                        "operator not specified for use with multiple " \
                        "fitness functions."
                    raise Exception(s)
        ### NEW 11-11-2020: Generate grammar automatically, based on the dataset
        if params['GRAMMAR_FILE'] == '' or params['GRAMMAR_FILE'] == 'auto':
            """gramm_filename = path.join(params['EXPERIMENT_NAME'], 
                                       params['FOLDER_NAME'],
                                       "grammar.bnf")
            import os
            os.makedirs(path.join(".evoltree", "grammars", params['EXPERIMENT_NAME'], 
                                  params['FOLDER_NAME']), exist_ok=True)
            """
            # Get base grammar
            GRAMMAR_PATH = pkg_resources.resource_filename('evoltree', 'grammars')
            default_filepath = path.join(GRAMMAR_PATH, "base.bnf")
            bnf = open(default_filepath, 'r')
            content = bnf.read() + "\n"
            bnf.close()
            
            import os
            gramm_filename = path.join(GRAMMAR_PATH, params['EXPERIMENT_NAME'], 
                                       params['FOLDER_NAME'], "grammar.bnf")
            os.makedirs(path.join(GRAMMAR_PATH, params['EXPERIMENT_NAME'], 
                                  params['FOLDER_NAME']), exist_ok=True)
            # Get data headers
            if params["X_train"] is None: # TODO: with the package, this directory may change
                data_file = open(path.join('datasets', 
                                           params['DATASET_TRAIN']), 'r')
                headers = data_file.readline()[:-1] # ignore last character: '\n'.
                data_file.close()
                headers = headers.replace(";" + params['TARGET'], "")
                headers = headers.replace(";", "'\" | \"'")
                # Build grammar file based on base
                idx = "<idx>\t\t\t::= \"'" + headers + "'\""
            else:
                columns = params["X_train"].columns
                headers=""
                for n, col in enumerate(columns):
                    if n==0:
                        headers += "\"'{0}'\"".format(col)
                    else:
                        headers += " | \"'{0}'\"".format(col)
                # Build grammar file based on base
                idx = "<idx>\t\t\t::= " + headers
            
            new_f = open(gramm_filename, 'w')
            new_f.write(content)
            new_f.write(idx)
            new_f.close()
            params['GRAMMAR_FILE'] = gramm_filename
        
        # Parse grammar file and set grammar class.
        params['BNF_GRAMMAR'] = grammar.Grammar(params['GRAMMAR_FILE'])
        ### NEW 20-11-2020: Generate Lamarck mapper automatically
        """if (params['LAMARCK_MAPPER'] == '' or 
            params['LAMARCK_MAPPER'] == 'auto'):
            from ..utilities.utils import create_lamarck_mapper
            params['LAMARCK_MAPPER'] = create_lamarck_mapper(params)
        """
        ### NEW 27-01-2021: Generate Lamarck mapper automatically ALWAYS!
        from ..utilities.utils import create_lamarck_mapper
        params['LAMARCK_MAPPER'] = create_lamarck_mapper(params)
        # Population loading for seeding runs (if specified)
        if params['TARGET_SEED_FOLDER'] != '':

            # Import population loading function.
            from ..operators.initialisation import load_population
            SEEDS_PATH = path.join(getcwd(), 'evoltree', 'seeds')
            makedirs(path.join(SEEDS_PATH, params['TARGET_SEED_FOLDER']),
                        exist_ok=True)
            # A target folder containing seed individuals has been given.
            params['SEED_INDIVIDUALS'] = load_population(
                params['TARGET_SEED_FOLDER'], SEEDS_PATH)

        elif params['REVERSE_MAPPING_TARGET']:
            # A single seed phenotype has been given. Parse and run.

            # Import GE LR Parser.
            from ..scripts import GE_LR_parser

            # Parse seed individual and store in params.
            params['SEED_INDIVIDUALS'] = [GE_LR_parser.main()]
        
        
