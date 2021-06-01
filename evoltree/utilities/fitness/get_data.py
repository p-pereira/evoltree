import logging
import pandas as pd

from ...algorithm.parameters import params


def get_Xy_train_test_separate(train_filename, test_filename, skip_header=0):
    """
    Read in training and testing data files, and split each into X
    (all columns up to last) and y (last column).
    
    :param train_filename: The file name of the training dataset.
    :param test_filename: The file name of the testing dataset.
    :param skip_header: The number of header lines to skip.
    :return: Parsed numpy arrays of training and testing input (x) and
    output (y) data.
    """

    if params['DATASET_DELIMITER']:
        # Dataset delimiter has been explicitly specified.
        delimiter = params['DATASET_DELIMITER']
    
    else:
        # Try to auto-detect the field separator (i.e. delimiter).
        f = open(train_filename)
        for line in f:
            if line.startswith("#") or len(line) < 2:
                # Skip excessively short lines or commented out lines.
                continue
                
            else:
                # Set the delimiter.
                if "\t" in line:
                    delimiter = "\t"
                    break
                elif "," in line:
                    delimiter = ","
                    break
                elif ";" in line:
                    delimiter = ";"
                    break
                elif ":" in line:
                    delimiter = ":"
                    break
                else:
                    logging.warning("utilities.fitness.get_data.get_Xy_train_test_separate\n"
                                    "Warning: Dataset delimiter not found. "
                                    "Defaulting to whitespace delimiter.")
                    delimiter = " "
                    break
        f.close()
    
    ################ NEW: changed (again) 13-11-2020
    # Read in all training data.
    trainData = pd.read_csv(train_filename, delimiter = delimiter)
    ##### OLD:
    # Read in all training data.
    #train_Xy = np.genfromtxt(train_filename, skip_header=skip_header, delimiter=delimiter)
    
    try:
        # Separate out input (X) and output (y) data.
        # NEW 13-11-2020: select target attribute based on parameters
        ### all columns except target
        train_X = trainData.drop(params['TARGET'], axis=1)
        ### target
        train_y = trainData[params['TARGET']]
    
    except IndexError:
        s = "utilities.fitness.get_data.get_Xy_train_test_separate\n" \
            "Error: specified delimiter '%s' incorrectly parses training " \
            "data." % delimiter
        raise Exception(s)

    if test_filename:
        testData = pd.read_csv(test_filename, delimiter = delimiter)
        # Separate out input (X) and output (y) data.
        # NEW 13-11-2020: select target attribute based on parameters
        ### all columns except target
        test_X = testData.drop(params['TARGET'], axis=1)
        ### target
        test_y = testData[params['TARGET']]

    else:
        test_X, test_y = None, None

    return train_X, train_y, test_X, test_y


def get_data(train, test):
    """
    Return the training and test data for the current experiment.
    
    :param train: The desired training dataset.
    :param test: The desired testing dataset.
    :return: The parsed data contained in the dataset files.
    """
    ### NEW 29-11-2020: dataset is loaded once only!
    """
    # Get the path to the training dataset.
    train_set = path.join("datasets", train)
     
    if test:
        # Get the path to the testing dataset.
        test_set = path.join("datasets", test)
    
    else:
        # There is no testing dataset used.
        test_set = None
    """
    
    # Read in the training and testing datasets from the specified files.
    training_in, training_out, test_in, test_out = \
        (params["X_train"], params["y_train"], 
         params["X_test"], params["y_test"])
    #get_Xy_train_test_separate(train_set, test_set, skip_header=1)
    
    return training_in, training_out, test_in, test_out
