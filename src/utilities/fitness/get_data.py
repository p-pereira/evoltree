from os import path

import numpy as np
import pandas as pd
import shelve

from algorithm.parameters import params


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
                    print("utilities.fitness.get_data.get_Xy_train_test_separate\n"
                          "Warning: Dataset delimiter not found. "
                          "Defaulting to whitespace delimiter.")
                    delimiter = " "
                    break
        f.close()
    
    ################NEW #### NEW AGAIN
    if params['DATA_TRANSF'] == "1HOT":
        f = shelve.open(train_filename)
        trainData = f['train']
        f.close()
    else:
        trainData = pd.read_csv(train_filename, delimiter = delimiter)
        if params['DATA_TRANSF'] == 'Factor':
            for col in trainData.columns:
                trainData[col] = trainData[col].astype(str)
    # Read in all training data.
    #train_Xy = np.genfromtxt(train_filename, skip_header=skip_header, delimiter=delimiter)
    
    try:
        # Separate out input (X) and output (y) data.
        #train_X = train_Xy[:, :-1].transpose()  # all columns but last
        #train_y = train_Xy[:, -1].transpose()  # last column
        train_X = trainData.drop('target', axis=1)  # all columns but last
        train_y = trainData['target'] # last column
    
    except IndexError:
        s = "utilities.fitness.get_data.get_Xy_train_test_separate\n" \
            "Error: specified delimiter '%s' incorrectly parses training " \
            "data." % delimiter
        raise Exception(s)

    if test_filename:
        if params['DATA_TRANSF'] == "1HOT":
            f = shelve.open(test_filename)
            testData = f['test']
            f.close()
        else:
            # Read in all testing data.
            #test_Xy = np.genfromtxt(test_filename, skip_header=skip_header, delimiter=delimiter)
            testData = pd.read_csv(test_filename, delimiter = delimiter)
            if params['DATA_TRANSF'] == 'Factor':
                for col in testData.columns:
                    testData[col] = testData[col].astype(str)
        # Separate out input (X) and output (y) data.
        #test_X = test_Xy[:, :-1].transpose()  # all columns but last
        #test_y = test_Xy[:, -1].transpose()  # last column
        test_X = testData.drop('target', axis=1)  # all columns but last
        test_y = testData['target'] # last column

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
    
    # Get the path to the training dataset.
    train_set = path.join("..", "datasets", train)
        
    if test:
        # Get the path to the testing dataset.
        test_set = path.join("..", "datasets", test)
    
    else:
        # There is no testing dataset used.
        test_set = None
    
    # Read in the training and testing datasets from the specified files.
    training_in, training_out, test_in, \
    test_out = get_Xy_train_test_separate(train_set, test_set, skip_header=1)
    
    return training_in, training_out, test_in, test_out
