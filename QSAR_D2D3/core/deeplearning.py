# define environment
import os, sys
import time
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
conf_dir = core_dir+"/conf"
sys.path.insert(0, core_dir)
sys.path.insert(0, conf_dir)

os.environ['OPENBLAS_NUM_THREADS'] = '1'

from buildmodel import *
from misc import *
from filters import *
from descriptor_setup import dnames, dlist
import copy
import numpy as np
from numpy import random
import itertools as it
from itertools import product
import random
import time
import pickle

# Keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, RepeatedStratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

# statistics and matplot
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.colors

# Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Optimizes GPU usage and allocates memory accordingly
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# turn off system warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# keeping environment variable fixed to prevent unwanted randomness
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.autograph.set_verbosity(0)

# print('precision', tf.keras.backend.floatx())

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer


def check_nans(train_descs, test_descs):
    # Check for NaN or infinity values
    if np.isnan(train_descs).any() or np.isinf(train_descs).any() or np.isnan(test_descs).any() or np.isinf(test_descs).any():
        print("Error: NaN or infinity values detected. Terminating the job.")
        sys.exit(1)  # Exit with a non-zero status code to indicate an error
    else:
        print("No NaN or infinity values detected. Proceeding with the job.")


def check_nans2(descs):
    # Check for NaN or infinity values
    if np.isnan(descs).any() or np.isinf(descs).any():
        print("Error: NaN or infinity values detected. Terminating the job.")
        sys.exit(1)  # Exit with a non-zero status code to indicate an error
    else:
        print("No NaN or infinity values detected. Proceeding with the job.")


def convert_to_zscore(train_descs, test_descs):
    '''Normalizing function - converts raw data to z-scores.
    Uses the mean and stdev of train_descs (only, not test_descs) to normalize both train_descs and test_descs '''
    # get mean and stdev of train_descs
    print("Using convert_to_zscore")
    combined_mean = np.mean(train_descs, axis=0)
    combined_std = np.std(train_descs, axis=0)

    # Check for standard deviation values that are very close to 0 and change them to 1. Without this, it can lead to explosion issues.
    ## (new changes)
    zero_std_mask = np.isclose(combined_std, 0.0)
    combined_std = np.where(zero_std_mask, 1.0, combined_std)
    ## (new changes)

    # layer it up -- same mean and stdev across the column
#     for i in range(train_descs.shape[0]-1):
#         combined_mean = np.vstack((combined_mean, x))
#         combined_std = np.vstack((combined_std, y))
    # apply it to train_descs
    tmp = np.subtract(train_descs, combined_mean)
    train_descs_norm = np.divide(tmp, combined_std)
    # apply it to test_descs using train_descs' mean and stdev
    tmp = np.subtract(test_descs, combined_mean)
    test_descs_norm = np.divide(tmp, combined_std)

    # change nans and infs to 0 (happens when stdev is 0... meaning raw data column is entirely 0)
    train_descs_norm[np.isnan(train_descs_norm)] = 0
    test_descs_norm[np.isnan(test_descs_norm)] = 0
    train_descs_norm[~np.isfinite(train_descs_norm)] = 0
    test_descs_norm[~np.isfinite(test_descs_norm)] = 0

    # get indexes of columns with 0s only... removes them
    # idx = np.argwhere(np.all(train_descs_norm == 0, axis=0))
    # idx2 = np.argwhere(np.all(test_descs_norm == 0, axis=0))
    # total_idx = np.unique(np.concatenate((idx, idx2)))
    # train_descs_norm = np.delete(train_descs_norm, total_idx, axis=1)
    # test_descs_norm = np.delete(test_descs_norm, total_idx, axis=1)

    check_nans(train_descs_norm, test_descs_norm)

    return train_descs_norm, test_descs_norm


def convert_to_zscore2(train_descs, test_descs=None):
    '''Normalizing function - converts raw data to z-scores.
        Uses the mean and stdev of train_descs (only, not test_descs) to normalize both train_descs and test_descs '''
    # Identify columns with continuous values based on the number of unique values.
    # We apply normalization only to these columns.
    print("Using convert_to_zscore2")
    non_binary_columns = np.apply_along_axis(lambda x: not set(np.unique(x)).issubset({0, 1}), axis=0, arr=train_descs)

    # get mean and stdev of train_descs for continuous columns
    combined_mean = np.mean(train_descs[:, non_binary_columns], axis=0)
    combined_std = np.std(train_descs[:, non_binary_columns], axis=0)

    # Check for standard deviation values that are very close to 0 and change them to 1. Without this, it can lead to explosion issues.
    zero_std_mask = np.isclose(combined_std, 0.0)
    combined_std = np.where(zero_std_mask, 1.0, combined_std)

    # Apply it to train_descs for continuous columns
    train_descs_norm = train_descs.copy()
    train_descs_norm[:, non_binary_columns] = (train_descs[:, non_binary_columns] - combined_mean) / combined_std
    # Change nans and infs to 0 (happens when stdev is 0... meaning raw data column is entirely 0)
    train_descs_norm[np.isnan(train_descs_norm)] = 0
    train_descs_norm[~np.isfinite(train_descs_norm)] = 0
    # Apply it to test_descs for continuous columns using train_descs' mean and stdev
    if test_descs is not None:
        test_descs_norm = test_descs.copy()
        test_descs_norm[:, non_binary_columns] = (test_descs[:, non_binary_columns] - combined_mean) / combined_std
        test_descs_norm[np.isnan(test_descs_norm)] = 0
        test_descs_norm[~np.isfinite(test_descs_norm)] = 0
    else:
        test_descs_norm = 0

    check_nans(train_descs_norm, test_descs_norm)

    return train_descs_norm, test_descs_norm


def convert_to_zscore3(train_descs, test_descs):
    '''Normalizing function - converts raw data to z-scores.
    non_decimal_columns = discrete columns
    Normalizes ONLY the non-discrete columns. If there is a single decimal number in the column, it is deemed non-discrete
    Uses the mean and stdev of train_descs (only, not test_descs) to normalize both train_descs and test_descs '''
    print("Using convert_to_zscore3")

    # Identify columns with decimal numbers
    discrete_columns = np.apply_along_axis(
        lambda x: np.issubdtype(x.dtype, np.floating) and not np.any(np.mod(x, 1) != 0), axis=0, arr=train_descs)
    decimal_columns = np.apply_along_axis(
        lambda x: np.issubdtype(x.dtype, np.floating) and np.any(np.mod(x, 1) != 0), axis=0, arr=train_descs)

    # get mean and stdev of train_descs for decimal columns
    combined_mean = np.mean(train_descs[:, decimal_columns], axis=0)
    combined_std = np.std(train_descs[:, decimal_columns], axis=0)

    # Check for standard deviation values that are very close to 0 and change them to 1. Without this, it can lead to explosion issues.
    zero_std_mask = np.isclose(combined_std, 0.0)
    combined_std = np.where(zero_std_mask, 1.0, combined_std)

    # Apply it to train_descs for decimal columns
    train_descs_norm = train_descs.copy()
    test_descs_norm = test_descs.copy()
    train_descs_norm[:, decimal_columns] = (train_descs[:, decimal_columns] - combined_mean) / combined_std
    # Apply it to test_descs for decimal columns using train_descs' mean and stdev
    test_descs_norm[:, decimal_columns] = (test_descs[:, decimal_columns] - combined_mean) / combined_std

    # Change nans and infs to 0 (happens when stdev is 0... meaning raw data column is entirely 0)
    train_descs_norm[np.isnan(train_descs_norm)] = 0
    test_descs_norm[np.isnan(test_descs_norm)] = 0
    train_descs_norm[~np.isfinite(train_descs_norm)] = 0
    test_descs_norm[~np.isfinite(test_descs_norm)] = 0

    check_nans(train_descs_norm, test_descs_norm)

    return train_descs_norm, test_descs_norm


def convert_to_zscore4(train_descs, test_descs, topo_names):
    '''Normalizing function - converts raw data to z-scores.
    non_decimal_columns = discrete columns
    Normalizes ONLY the non-discrete columns. If there is a single decimal number in the column, it is deemed non-discrete
    Uses the mean and stdev of train_descs (only, not test_descs) to normalize both train_descs and test_descs '''

    print("Using convert_to_zscore4")
    ls_features = ["HeavyAtomCount", "NOCount", "NumHAcceptors", "NumRotatableBonds", "NumValenceElectrons",
                   "NumHeteroatoms", "RingCount", "NumSaturatedRings", "NumSaturatedHeterocycles"]
    # Get the indexes of non-decimal columns specified in ls_features
    non_decimal_indexes = [topo_names.index(feature) for feature in ls_features]


    # Identify columns with decimal numbers
    decimal_columns = np.apply_along_axis(
        lambda x: np.issubdtype(x.dtype, np.floating) and np.any(np.mod(x, 1) != 0), axis=0, arr=train_descs)
    # Set the corresponding entries in decimal_columns to True
    decimal_columns[non_decimal_indexes] = True


    # get mean and stdev of train_descs for decimal columns
    combined_mean = np.mean(train_descs[:, decimal_columns], axis=0)
    combined_std = np.std(train_descs[:, decimal_columns], axis=0)

    # Check for standard deviation values that are very close to 0 and change them to 1. Without this, it can lead to explosion issues.
    zero_std_mask = np.isclose(combined_std, 0.0)
    combined_std = np.where(zero_std_mask, 1.0, combined_std)

    # Apply it to train_descs for decimal columns
    train_descs_norm = train_descs.copy()
    test_descs_norm = test_descs.copy()
    train_descs_norm[:, decimal_columns] = (train_descs[:, decimal_columns] - combined_mean) / combined_std
    # Apply it to test_descs for decimal columns using train_descs' mean and stdev
    test_descs_norm[:, decimal_columns] = (test_descs[:, decimal_columns] - combined_mean) / combined_std

    # Change nans and infs to 0 (happens when stdev is 0... meaning raw data column is entirely 0)
    train_descs_norm[np.isnan(train_descs_norm)] = 0
    test_descs_norm[np.isnan(test_descs_norm)] = 0
    train_descs_norm[~np.isfinite(train_descs_norm)] = 0
    test_descs_norm[~np.isfinite(test_descs_norm)] = 0

    check_nans(train_descs_norm, test_descs_norm)

    return train_descs_norm, test_descs_norm


def convert_to_zscore5(train_descs, test_descs, topo_names, normalize = "both"):
    '''Normalizing function - converts raw data to z-scores.
    non_decimal_columns = discrete columns
    Normalizes ONLY the non-discrete columns. If there is a single decimal number in the column, it is deemed non-discrete
    Uses the mean and stdev of train_descs (only, not test_descs) to normalize both train_descs and test_descs '''

    print("Using convert_to_zscore5")
    ls_features = ["HeavyAtomCount", "NumValenceElectrons"]
    # Get the indexes of non-decimal columns specified in ls_features
    non_decimal_indexes = [topo_names.index(feature) for feature in ls_features]

    # Identify columns with decimal numbers
    decimal_columns = np.apply_along_axis(
        lambda x: np.issubdtype(x.dtype, np.floating) and np.any(np.mod(x, 1) != 0), axis=0, arr=train_descs)
    # Set the corresponding entries in decimal_columns to True
    decimal_columns[non_decimal_indexes] = True

    # get mean and stdev of train_descs for decimal columns
    combined_mean = np.mean(train_descs[:, decimal_columns], axis=0)
    combined_std = np.std(train_descs[:, decimal_columns], axis=0)

    # Check for standard deviation values that are very close to 0 and change them to 1. Without this, it can lead to explosion issues.
    zero_std_mask = np.isclose(combined_std, 0.0)
    combined_std = np.where(zero_std_mask, 1.0, combined_std)

    # Apply it to train_descs for decimal columns
    train_descs_norm = train_descs.copy()
    train_descs_norm[:, decimal_columns] = (train_descs[:, decimal_columns] - combined_mean) / combined_std

    # Change nans and infs to 0 (happens when stdev is 0... meaning raw data column is entirely 0)
    train_descs_norm[np.isnan(train_descs_norm)] = 0
    train_descs_norm[~np.isfinite(train_descs_norm)] = 0

    if normalize == "both":
        test_descs_norm = test_descs.copy()
        test_descs_norm[:, decimal_columns] = (test_descs[:, decimal_columns] - combined_mean) / combined_std
        test_descs_norm[np.isnan(test_descs_norm)] = 0
        test_descs_norm[~np.isfinite(test_descs_norm)] = 0
        check_nans(train_descs_norm, test_descs_norm)
        return train_descs_norm, test_descs_norm

    elif normalize == "train_descs": # normalizing only train_descs
        check_nans2(train_descs_norm)

        return train_descs_norm


def convert_to_minmax(train_descs, test_descs, unique_threshold = 1):
    '''Scaling function - converts raw data to min-max scaled values.
        Uses the range of train_descs (only, not test_descs) to scale both train_descs and test_descs
        NEVER keep unique_threshold as 0 (unless for good reason). This will lead to nans and infs and cause issues'''
    print("Using convert_to_minmax")
    # Identify columns with continuous values based on the number of unique values.
    # We apply scaling only to these columns.
    continuous_columns = np.apply_along_axis(lambda x: len(np.unique(x)) > unique_threshold, axis=0, arr=train_descs)

    # get min and max of train_descs for continuous columns
    min_vals = np.min(train_descs[:, continuous_columns], axis=0)
    max_vals = np.max(train_descs[:, continuous_columns], axis=0)

    # Apply it to train_descs for continuous columns
    train_descs_scaled = train_descs.copy()
    test_descs_scaled = test_descs.copy()

    # Scale train_descs
    train_descs_scaled[:, continuous_columns] = (train_descs[:, continuous_columns] - min_vals) / (max_vals - min_vals)

    # Scale test_descs using train_descs' min and max
    test_descs_scaled[:, continuous_columns] = (test_descs[:, continuous_columns] - min_vals) / (max_vals - min_vals)

    check_nans(train_descs_scaled, test_descs_scaled)

    return train_descs_scaled, test_descs_scaled


def convert_to_zscore_features(train_descs, test_descs, topo_names):
    '''Normalizing function - converts raw data to z-scores for specified features in ls_features.'''
    print("Using convert_to_zscore_features")

    ls_features = ["HeavyAtomCount", "NumValenceElectrons",
                   "BertzCT", "ExactMolWt", "HeavyAtomMolWt", "LabuteASA", "MolWt", "PEOE_VSA1",
                   "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SlogP_VSA2",
                   "SlogP_VSA5", "SlogP_VSA6", "TPSA"
                   ]

    # Get the indexes of features in ls_features
    feature_indexes = [topo_names.index(feature) for feature in ls_features]

    # get mean and stdev of train_descs for specified features
    combined_mean = np.mean(train_descs[:, feature_indexes], axis=0)
    combined_std = np.std(train_descs[:, feature_indexes], axis=0)

    # Check for standard deviation values that are very close to 0 and change them to 1.
    zero_std_mask = np.isclose(combined_std, 0.0)
    combined_std = np.where(zero_std_mask, 1.0, combined_std)

    # Apply it to train_descs for continuous columns
    train_descs_norm = train_descs.copy()
    test_descs_norm = test_descs.copy()

    # Apply it to train_descs for specified features
    train_descs_norm[:, feature_indexes] = (train_descs[:, feature_indexes] - combined_mean) / combined_std
    # Apply it to test_descs for specified features using train_descs' mean and stdev
    test_descs_norm[:, feature_indexes] = (test_descs[:, feature_indexes] - combined_mean) / combined_std

    # Change nans and infs to 0 (happens when stdev is 0, meaning raw data column is entirely 0)
    train_descs_norm[np.isnan(train_descs_norm)] = 0
    test_descs_norm[np.isnan(test_descs_norm)] = 0
    train_descs_norm[~np.isfinite(train_descs_norm)] = 0
    test_descs_norm[~np.isfinite(test_descs_norm)] = 0

    check_nans(train_descs_norm, test_descs_norm)

    return train_descs_norm, test_descs_norm


def convert_to_combined_norm(train_descs, test_descs):
    '''Normalizing function - converts raw data to z-scores.
    non_decimal_columns = discrete columns
    Normalizes ONLY the non-discrete columns. If there is a single decimal number in the column, it is deemed non-discrete
    Uses the mean and stdev of train_descs (only, not test_descs) to normalize both train_descs and test_descs '''
    print("Using convert_to_combined_norm")

    continuous_columns = np.apply_along_axis(lambda x: len(np.unique(x)) > 2, axis=0, arr=train_descs)

    # Identify decimal columns among continuous columns
    decimal_columns = np.apply_along_axis(lambda x: np.issubdtype(x.dtype, np.floating) and np.any(np.mod(x, 1) != 0),
                                          axis=0, arr=train_descs)
    decimal_columns_continuous = np.logical_and(decimal_columns, continuous_columns)

    # Identify discrete columns among continuous columns
    discrete_columns = np.logical_and(continuous_columns, ~decimal_columns_continuous)

    ### Z-score normalization (decimal_columns)
    # get mean and stdev of train_descs for decimal columns
    combined_mean = np.mean(train_descs[:, decimal_columns], axis=0)
    combined_std = np.std(train_descs[:, decimal_columns], axis=0)
    # Check for standard deviation values that are very close to 0 and change them to 1. Without this, it can lead to explosion issues.
    zero_std_mask = np.isclose(combined_std, 0.0)
    combined_std = np.where(zero_std_mask, 1.0, combined_std)

    ### Min-Max normalization (discrete_columns)
    # get min and max of train_descs for continuous columns
    min_vals = np.min(train_descs[:, discrete_columns], axis=0)
    max_vals = np.max(train_descs[:, discrete_columns], axis=0)

    ### Apply normalization for both cases
    train_descs_norm = train_descs.copy()
    test_descs_norm = test_descs.copy()
    # Z-score
    train_descs_norm[:, decimal_columns] = (train_descs[:, decimal_columns] - combined_mean) / combined_std
    test_descs_norm[:, decimal_columns] = (test_descs[:, decimal_columns] - combined_mean) / combined_std

    # Min-max scale
    train_descs_norm[:, discrete_columns] = (train_descs[:, discrete_columns] - min_vals) / (max_vals - min_vals)
    test_descs_norm[:, discrete_columns] = (test_descs[:, discrete_columns] - min_vals) / (max_vals - min_vals)

    # Change nans and infs to 0 (happens when stdev is 0... meaning raw data column is entirely 0)
    train_descs_norm[np.isnan(train_descs_norm)] = 0
    test_descs_norm[np.isnan(test_descs_norm)] = 0
    train_descs_norm[~np.isfinite(train_descs_norm)] = 0
    test_descs_norm[~np.isfinite(test_descs_norm)] = 0

    check_nans(train_descs_norm, test_descs_norm)

    return train_descs_norm, test_descs_norm


def len_dictionary(test_dict):
    '''
    :param test_dict: Input dictionary, such as the parameter dictionary
    :return: Returns the NUMBER of total combination in test_dict
    '''
    temp = list(test_dict.keys())
    res = dict()
    cnt = 0

    # making key-value combinations using product
    for combs in product (*test_dict.values()):
        # zip used to perform cross keys combinations.
        res[cnt] = [[ele, cnt] for ele, cnt in zip(test_dict, combs)]
        cnt += 1
    return cnt


def set_tf_seed(n):
    """
    Sets all necessary random seeds with a randomly generated number n
    :param n: int, random number generated from a random number generator
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(n)
    random.seed(n)
    np.random.seed(n)
    tf.random.set_seed(n)


def get_training_dataset(mode, tp, stage, train_dataset_prefix, output_dir, output_ext, rand_state,
                         random_split, all_descs=False, extra_features = False, remove_ipc = False, chiral_descs=False):
    """
    Prepares training dataset by splitting dataset based on tp
    :param all_descs: True or False. True: include all 200 descriptors; False: prune the features.
    :param extra_features: True or False. True: include Ph2D features; False: exclude Ph2D features
    :param remove_ipc: True or False. True: Remove feature Ipc (index 33 if you start from 0)
    """
    print(f'(rand_state)_(random_split): {rand_state}_{random_split}')

    mols, acts, deletes, changes = read_data4buildmodel(train_dataset_prefix, mode)
    mols, acts = curate_mols(mols, acts, deletes, changes)
    if tp > 0:
        # Split data into 85:15 ratio
        train_mols, train_names, train_acts, \
        test_mols, test_names, test_acts = split_data(mols, acts, tp, random_split)
        # Calculate AD domain
        ad_fps, ad_rad = calc_appdom(train_mols, output_dir, output_ext)
        test_mols, test_acts, test_names, test_mols_reject, test_acts_reject, test_names_reject \
            = check_appdom(ad_fps, ad_rad, test_mols, test_names, test_acts, step=stage)
    else:
        train_mols, train_names, train_acts = all_data(mols, acts)
        ad_fps, ad_rad = calc_appdom(train_mols, output_dir, output_ext)


    # Get indices for 200 descriptors. Load dnames from descriptor_setup.py (see core/conf/descriptor_setup.py)
    # If pruned, these are updated later on
    topo_names = copy.copy(dnames)
    if remove_ipc:
        topo_names.remove("Ipc")
    topo_index = []
    for desc in topo_names:
        i = dnames.index(desc)
        topo_index.append(i)
    if chiral_descs:
        topo_names.append('nChiral')
        topo_names.append('ChirIdx')
        max_index = max(topo_index)
        topo_index.append(max_index + 1)
        topo_index.append(max_index + 2)

    # Using Ph2D features
    if extra_features == True:
        train_topo_descs = calc_topo_descs(train_mols, topo_index)
        if chiral_descs:
            train_chir_descs = calc_chir_descs(train_mols)
            train_topo_descs = np.concatenate((train_topo_descs, train_chir_descs), axis=1)
        train_phore_descs, phore_sigbits, phore_names = prune_phore_descs(calc_phore_descs(train_mols), output_dir, output_ext)
        train_descs = np.concatenate((train_topo_descs, train_phore_descs), axis=1)
    else:
        train_descs = calc_topo_descs(train_mols, topo_index)

    # Pruning 200 features and updating topo_index and topo_names
    if not all_descs:
        train_descs, topo_index, topo_names = prune_topo_descs(mode, train_descs, train_acts, output_dir)


    # Getting other variables
    train_names = np.array(train_names)
    train_acts = np.array(train_acts)
    if tp > 0:
        test_descs = calc_topo_descs(test_mols, topo_index)
        if extra_features == True: # add on Ph2D descriptors
            if chiral_descs:
                test_chir_descs = calc_chir_descs(test_mols)
                test_descs = np.concatenate((test_descs, test_chir_descs), axis=1)
            test_phore_descriptors = calc_phore_descs(test_mols, phore_sigbits)
            test_descs = np.concatenate((test_descs, test_phore_descriptors), axis=1)
        test_acts = np.array(test_acts)
    else:
        test_descs, test_acts, test_names = 0, 0, 0  # These are not used if tp == 0
    print("Training dataset preparation is done. train_descs.shape:", train_descs.shape)

    return train_names, test_names, train_descs, train_acts, test_descs, test_acts, topo_names, phore_names


def retrieve_training_data(filename, tp):
    with open(filename, 'rb') as handle:
        dict_data = pickle.load(handle)

    train_names = dict_data['train_names']
    train_descs = dict_data['train_descs']
    train_descs_norm = dict_data['train_descs_norm']
    train_acts = dict_data['train_acts']
    topo_names = dict_data["topo_names"]
    phore_names = dict_data["phore_names"]
    feature_names = dict_data["feature_names"]

    if tp > 0:
        test_acts = dict_data['test_acts']
        test_descs = dict_data['test_descs']
        test_descs_norm = dict_data['test_descs_norm']
        test_names = dict_data["test_names"]
    elif tp == 0:
        test_acts, test_descs, test_descs_norm, test_names = 0, 0, 0, 0

    return train_descs, test_descs, train_descs_norm, test_descs_norm, train_acts, test_acts, train_names, test_names, \
           topo_names, phore_names, feature_names


def get_validation_dataset(method, datadir, topo_index):
    '''
    Prepares validation set. We are using dataset gathered from Dr. Newman's group within our branch.
    :param method: str, dnn (deep neural network)
    :param datadir: str, path to the data directory
    :param topo_index: list, list of indexes that are used for descriptors, e.g. [0 1 2 .... 199] is the original list
    :return: mol_names, standard_mol_smiles, mol_act, validation_descs
    '''
    moldatafile = str(datadir + ".smi")
    actdatafile = str(datadir + ".act")

    # Read in molecules_name/simles from validation set
    mol_simles, mol_names = [], []
    molfh = open(moldatafile)
    for molline in molfh:
        line = molline.split()
        mol_simles.append(Chem.MolFromSmiles(line[0]))
        mol_names.append(line[1])
    molfh.close()

    # Read in molecules_name/act from validation set
    mol_names_check, mol_act = [], []
    molfh = open(actdatafile)

    for molline in molfh:
        line = molline.split()
        mol_names_check.append(line[0])
        mol_act.append(float(line[1]))
    molfh.close()

    # make sure the input files have the same order
    assert mol_names == mol_names_check

    # Standardize structures
    s = Standardizer()
    standard_mol_smiles = []
    for imol in mol_simles:
        standard_mol_smiles.append(s.standardize(imol))

    # we need topo_index from training dataset
    validation_descs = calc_topo_descs(standard_mol_smiles, topo_index)
    mol_act = np.array(mol_act)

    if method.startswith('cnn'):
        # change dimensions
        validation_descs = np.expand_dims(validation_descs, axis=2)

    print("Validation set preparation is done")

    return mol_names, standard_mol_smiles, mol_act, validation_descs

#
# def build_dnn_model(mode, rand_state, hidden_layers, neurons, dropout, learning_rate):
#     """
#     Note that input layer is not needed. Keras already takes that into consideration
#     Order of layers:
#         normalization layer (for normalizing, prevents gradient explosion, which would cause NaN values)
#         input layer
#         hidden layers (this is where # of neurons go)
#         output layer (this is where reg vs class happens with loss function)
#     :param mode: str, 'reg' or 'class' (regression or classification)
#     :param random_split: int, randomly generated number for fixing random seed in function set_tf_seed
#     :param hidden_layers: int, number of hidden layers in the DNN architecture
#     :param neurons: int, number of neurons in the DNN architecture
#     :param dropout: float, proportion of neurons that is skipped randomly to prevent overlearning
#     :param learning_rate: float, the rate at which models are being learned
#     :return: model
#     """
#     set_tf_seed(rand_state)
#     model = keras.Sequential()
#
#     # hidden layers
#     for i in range(hidden_layers):
#         model.add(layers.Dense(neurons, activation='relu'))  # tanh? relu?
#         model.add(layers.Dropout(rate=dropout, seed = rand_state))
#
#     # output layer
#     if mode == 'reg':
#         model.add(layers.Dense(1))
#         model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
#     elif mode == 'class':
#         model.add(layers.Dense(1, activation="sigmoid"))
#         model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#                       metrics=['accuracy'])
#
#     return model

def build_dnn_model(mode, rand_state, hidden_layers, neurons, dropout, learning_rate, num_features):
    set_tf_seed(rand_state)
    model = keras.Sequential()

    model.add(layers.InputLayer(input_shape=(num_features,)))

    # hidden layers
    for i in range(hidden_layers):
        model.add(layers.Dense(neurons, activation='relu'))  # tanh? relu?
        if i < hidden_layers - 1:
            model.add(layers.Dropout(rate=dropout, seed = rand_state))
    # output layer
    if mode == 'reg':
        model.add(layers.Dense(1))
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    elif mode == 'class':
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])
    return model


def buildmodel_RandomSearchCV(mode, rand_state, train_descs, train_acts, output_ext):
    '''
    RandomSearchCV is used. Instead of waiting until all iterations are complete, we are running them one at a time and
    save progress. This is saved with the 'iter_restart' files.
    :param mode: str, either regression (raw data) or classification (binary data)
    :param rand_state: int, random seed is set depending on this number.
    :param train_descs: 2D array, Descriptor matrix of the training dataset
    :param train_acts: 2D array, Binding affinity matrix of training dataset
    :param output_dir: str, folder name where model contents are stored e.g., reg_dnn_0.15
    :param output_ext: str, output_dir name followed by rand_state and rand_split, e.g. reg_dnn_0.15_87_864
    :return: best_model, best_params, best_score, model_time
    '''
    # initial values
    n_jobs = 4
    cv = 10  # Cross Validation, number of cross-folds to build models to internally validate
    num_features =  train_descs.shape[1]

    dict_para = {'epochs': [160, 640, 1280],
     'hidden_layers': [1, 2],
     'neurons': [6144, 7168, 8192],
     'learning_rate': [1**(-5.5), 1e-05, 1**(-4.5)],
     'batch_size': [256, 512, 1024],
     'dropout': [0.0, 0.1]}


    # takes forever to run
    # dict_para = {'batch_size': [64, 128, 512],
    #  'dropout': [0.0, 0.1],
    #  'epochs': [640, 2560, 5120],
    #  'hidden_layers': [5, 6, 8],
    #  'learning_rate': [1e-06, 10**(-5.5), 1e-05],
    #  'neurons': [5120, 7168, 8192]}


    # dict_para = {'epochs': [2560, 5120, 10240],
    #  'hidden_layers': [1, 2, 3],
    #  'neurons': [6144, 7168, 8192],
    #  'learning_rate': [10**(-5.5), 1e-05, 10**(-4.5)],
    #  'batch_size': [16, 32, 64],
    #  'dropout': [0.0,0.1]}
    # dict_para = {'epochs': [10],
    #  'hidden_layers': [5],
    #  'neurons': [32,64],
    #  'learning_rate': [1e-02],
    #  'batch_size': [128],
    #  'dropout': [0.0]}

    n_iter = len_dictionary(dict_para)

    _dict_para = copy.deepcopy(dict_para)
    print('n_jobs', n_jobs, 'cv', cv, 'n_iter', n_iter)
    print(dict_para)
    ################## CODE ##################
    output_dir = output_ext.rpartition('_')[0].rpartition('_')[0]
    build_model = build_dnn_model

    if mode == 'reg':
        KerasModel = KerasRegressor(build_fn=build_model, verbose=0)
        scoring = 'neg_mean_squared_error'
    elif mode == 'class':
        KerasModel = KerasClassifier(build_fn=build_model, verbose=0)
        scoring = 'accuracy'
    # searchCV - build model takes place here
    # read iter_restart from a file
    my_file = Path(f'{output_dir}/iter_restart_{output_ext}.txt')
    if my_file.is_file():
        with open(f'{output_dir}/iter_restart_{output_ext}.txt', 'r') as f:
            x = f.readlines()
        x = [s.strip('\n') for s in x]
        # retrieving previous run's information
        iter_restart = int(x[0]) + 1  # one more than where we last finished
        print(iter_restart)
        best_score = float(x[1])
        best_params = eval(x[2])
        best_iter = int(x[3])
        best_model = keras.models.load_model(f"{output_dir}/model_{output_ext}")
    else:
        iter_restart = 1
        print(iter_restart)
        try:
            print("Deleting preexisting score_i_iter.txt ")
            os.remove(f'{output_dir}/score_i_iter_{output_ext}.txt')
        except FileNotFoundError:
            print("score_i_iter file is not in the folder, nothing to delete.")

    total_time = 0
    # creating a huge list of dictionary combinations (in tuple form, ls_combinations. This comes out alphabetically)
    allNames = sorted(dict_para)
    combinations = it.product(*(dict_para[Name] for Name in allNames))
    ls_combinations = list(combinations)
    random.Random(rand_state).shuffle(ls_combinations)

    # if
    if n_iter > len(ls_combinations):
        n_iter = len(ls_combinations)

    for i_iter in range(iter_restart, n_iter + 1):
        print(i_iter)
        batch_size = ls_combinations[i_iter-1][0]
        dropout = ls_combinations[i_iter-1][1]
        epochs = ls_combinations[i_iter-1][2]
        hidden_layers = ls_combinations[i_iter-1][3]
        learning_rate = ls_combinations[i_iter-1][4]
        neurons = ls_combinations[i_iter-1][5]

        # all arguments in build_dnn_model function are required to be listed here, i.e. mode and rand_state
        params = {"batch_size": [batch_size],
                  "dropout": [dropout],
                  "epochs": [epochs],
                  "hidden_layers": [hidden_layers],
                  "learning_rate": [learning_rate],
                  "neurons": [neurons],
                  "mode": [mode],
                  "rand_state": [rand_state],
                  "num_features": [num_features]
                  }

        start = time.time()
        # Search is being done with one parameter each time
        search_models = RandomizedSearchCV(KerasModel, params, n_iter=1, scoring=scoring, n_jobs=n_jobs, cv=cv,
                                           random_state=rand_state, verbose = 1)
        search_models.fit(train_descs, train_acts)
        end = time.time()
        model_time = (end - start) / 3600
        total_time = total_time + model_time
        # current model information
        _model = search_models.best_estimator_
        _score = search_models.best_score_
        _params = search_models.best_params_
        for key in ['mode', 'rand_state', "num_features"]:
            _params.pop(key)
        with open(f'{output_dir}/score_i_iter_{output_ext}.txt', 'a') as f:
            f.write(f'{output_ext} i_iter {i_iter} score {_score} batch_size {batch_size} '
                    f'dropout {dropout} epochs {epochs} hidden_layers {hidden_layers} '
                    f'learning_rate {learning_rate} neurons {neurons}\n')
        if i_iter == 1:
            best_model = _model
            best_score = _score
            best_params = _params
            best_iter = 1
            best_model.model.save(f"{output_dir}/model_{output_ext}")
        else:
            # if current score is better than best score, then we need to rewrite everything
            if _score > best_score:
                best_model = _model
                best_score = _score
                best_params = _params
                best_iter = i_iter
                best_model.model.save(f"{output_dir}/model_{output_ext}")
            # if not, then we don't rewrite anything.... continue as it is.
        with open(f'{output_dir}/iter_restart_{output_ext}.txt', 'w') as f:
            f.write(f'{i_iter}\n{best_score}\n{best_params}\n{best_iter}\n')
        with open(f"{output_dir}/Compiled_{output_ext}.log", 'w') as f:
            f.write(f"{output_ext} {best_iter} of {i_iter}  BestScore {best_score} TotalHours {total_time} BestParameters {best_params}\n")
        with open(f"{output_dir}/parameter_used_{output_ext}.txt", 'w') as f:
            f.write(f"{output_ext} n_iter {n_iter} n_jobs {n_jobs} cv {cv} descs {train_descs.shape} \
                        Parameters: {_dict_para}\n")
    print("Search model done, Hours took:", total_time)

    return best_model, best_params, best_score, total_time


def benchmark_plot(true, pred, output_dir, output_ext):
    '''
    Plots predicted scatterplot, true vs predicted values.
    :param true: 2D-array, true, experimental value (from ChEMBL)
    :param pred: 2D-array, predicted values (from the model)
    :param output_ext: str, contains full information of the current model being run, e.g. reg_dnn_0.15_87_864
    '''
    # plot and save external benchmark
    plt.xlim(1, 10)
    plt.ylim(1, 10)
    plt.scatter(true, pred)
    plt.title(f"Test Data Results")
    plt.xlabel("True Output")
    plt.ylabel("Predicted Output")
    outputfile = output_dir + "/" + "benchmark" + "_" + output_ext
    plt.savefig(outputfile + ".png")
    plt.clf()


def stats(y_true, y_pred, verbose = False):
    '''
    Given the predicted and true values, calculates the statistical values
    :param y_true: 2D-array, true, experimental value (from ChEMBL)
    :param y_pred: 2D-array, predicted values (from the model)
    :return: R2 (Coefficient of Determination), Rp (Pearson), Rs (Spearman), MSE (Mean Squared Error)
    '''
    #print(y_true.shape, y_pred.shape)
    Rs = spearmanr(y_true, y_pred)[0]
    Rp = pearsonr(y_true, y_pred)[0]
    R2 = r2_score(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = sqrt(MSE)
    if verbose == True:
        print('R2', R2, 'Rp', Rp, 'Rs', Rs, 'MSE', MSE, 'RMSE', RMSE)
    return R2, Rp, Rs, MSE, RMSE


def predict_model(model, test_names, test_descs, test_acts, output_ext):
    """
    Using the model built from SearchCV, predict using the testing descriptors and calculate the statistical benchmarks.
    :param model: model itself
    :param test_descs: training descriptors (features)
    :param test_acts: final descriptors (features); true value
    """
    output_dir = output_ext.rpartition('_')[0].rpartition('_')[0]
    start_time = time.time()
    y_true, y_pred = test_acts, model.predict(test_descs).flatten()
    end_time = time.time()
    elapsed_time = end_time - start_time
    R2, Rp, Rs, MSE, RMSE = stats(y_true, y_pred)
    benchmark_plot(y_true, y_pred, output_dir, output_ext)
    split = output_ext.split('_')[-1]
    with open(f"{output_dir}/stat_{split}.txt", 'w') as f:
        f.write(f"{output_ext} R2 {R2} Rp {Rp} Rs {Rs} MSE {MSE} RMSE {RMSE}\n")
    with open(f"{output_dir}/pred_{split}.csv", 'w') as f:
        f.write(f"compound\texp_mean\tpred_{split}\n")
        for i in range(len(y_pred)):
            f.write(f"{test_names[i]}\t{y_true[i]}\t{y_pred[i]}\n")

    print("Benchmark done")
    print(f"Elapsed time: {elapsed_time} seconds")


def read_mols_dnn(mode, method, output_ext, basename, datadir='Default', modeldir='Default'):
    currworkdir = os.getcwd()
    if datadir == 'Default':
        datadir = os.path.join(currworkdir, 'data')
    else:
        if not os.path.isdir(datadir):
            print("error: ", datadir, " is not a directory. exiting.")
            exit(2)

    if modeldir == 'Default':
        modeldir = os.path.join(currworkdir, 'models')
    else:
        if not os.path.isdir(modeldir):
            print("error: ", modeldir, " is not a directory. exiting.")
            exit(2)
        else:
            print('setting modeldir to ', modeldir, '.')
            print('Have you set the random splits to be correct for the model?')

    mol_data_filename = basename + '.smi'
    act_data_filename = basename + '.act'
    moldatafile = os.path.join(datadir, mol_data_filename)
    actdatafile = os.path.join(datadir, act_data_filename)

#     output_ext = "%s_%s_%d_%d" % (mode, method, int(rand_split), int(rand_state))
#     model_filename = "model_%s" % output_ext
    # index_filename = "indices_%s.dat" % output_ext
    appdom_fp_filename = "training-FPs_%s.dat" % output_ext
    appdom_rad_filename = "AD-radius_%s.dat" % output_ext

    ls_acts = []
    if mode.startswith('class'):
        bits_filename = "sigbits_%s.dat" % output_ext
        bits_file = os.path.join(modeldir, bits_filename)
        with open(bits_file, 'rb') as f:
            significant_bits = pickle.load(f)
        if os.path.isfile(actdatafile):
            actfh = open(actdatafile)

            activities = []  # array of tuples: (activity, molecule name)

            for actline in actfh:
                line = actline.split()
                act = float(line[1])
                actname = line[0]
                activities.append((act, actname))

            actfh.close()

    elif mode.startswith('reg'):
        bits_filename = "sigbits_%s.dat" % output_ext
        bits_file = os.path.join(modeldir, bits_filename)
        with open(bits_file, 'rb') as f:
            significant_bits = pickle.load(f)

#     model_file = os.path.join(modeldir, model_filename)
    # loaded_model = pickle.load(open(model_file, "rb"))
#     print('modeldir', modeldir)
#     print('model_file', model_file)
#     loaded_model = keras.models.load_model(model_file)
    #f'{output_dir}/model_{output_ext}'

    # index_file = os.path.join(modeldir, index_filename)
    # with open(index_file, 'rb') as f:
    #     indexes = pickle.load(f)

    appdom_fp_file = os.path.join(modeldir, appdom_fp_filename)
    with open(appdom_fp_file, 'rb') as f:
        appdom_fps = pickle.load(f)

    appdom_rad_file = os.path.join(modeldir, appdom_rad_filename)
    with open(appdom_rad_file, 'rb') as f:
        appdom_radius = pickle.load(f)

    # Read in molecules from test set
    molfh = open(moldatafile)

    molecules = []  # array of tuples: (molecule, molecule name)

    for molline in molfh:
        line = molline.split()
        mol = Chem.MolFromSmiles(line[0])
        molname = line[1]
        molecules.append((mol, molname))

    molfh.close()

    mols_train = []
    molnames_train = []

    if 'activities' in locals():
        acts_train = []
        actnames_train = []

    for i in range(len(molecules)):
        mols_train.append(molecules[i][0])
        molnames_train.append(molecules[i][1])
        if mode.startswith('class') and 'activities' in locals():
            acts_train.append(activities[i][0])
            actnames_train.append(activities[i][1])

    # Standardize structures
    s = Standardizer()
    standard_mols_train = []
    for mol in mols_train:
        standard_mols_train.append(s.standardize(mol))

    return_dict = {}

    return_dict['molnames'] = molnames_train
    return_dict['molecules'] = standard_mols_train
#     return_dict['model'] = loaded_model
    # return_dict['inds'] = indexes
    return_dict['sigbits'] = significant_bits
    if mode.startswith('class') and 'activities' in locals():
        return_dict['activities'] = acts_train
    return_dict['ad_fps'] = appdom_fps
    return_dict['ad_radius'] = appdom_radius

    return return_dict


# def build_cnn_model(mode, train_descs, rand_state, convlayers, neurons, learning_rate, filters, kernel_size, pool_size):
#     set_tf_seed(rand_state)
#     model = keras.Sequential()
#     normalizer = preprocessing.Normalization(axis=-1)
#     normalizer.adapt(np.array(train_descs))  # Normalizing train_descs (training matrix with features)
#     model.add(normalizer)
#     model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='tanh',
#                         input_shape=(train_descs.shape[1], train_descs.shape[2]), padding='same'))
#     model.add(layers.MaxPooling1D(pool_size=pool_size))
#
#     for i_convlayers in range(convlayers):
#         model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='tanh', padding='same'))
#         model.add(layers.MaxPooling1D(pool_size=pool_size))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(neurons, activation='relu'))
#
#     if mode == "reg":
#         model.add(layers.Dense(1))
#         model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
#     elif mode == "class":
#         model.add(layers.Dense(1, activation='sigmoid'))
#         model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#                       metrics=['accuracy'])
#
#     return model
#
def predict_class(model, test_names, train_acts, test_descs, test_acts, output_dir, output_ext):
    raw_predictions = model.predict(test_descs)
    # test_results = raw_predictions.argmax(axis=-1)
    # # test_results = np.argmax(predict_test, axis=1)
    test_results = np.where(raw_predictions > 0.5, 1, 0)
    acc = accuracy_score(test_acts, test_results)
    sens = recall_score(test_acts, test_results, pos_label=1)
    spec = recall_score(test_acts, test_results, pos_label=0)
    print('Accuracy: %5.3f' % acc)
    print('Sensitivity: %5.3f ' % sens)
    print('Specificity: %5.3f ' % spec)
    confmat = confusion_matrix(test_acts, test_results, labels=[1, 0])
    print('')
    print(confmat)
    print('')
    train_pos = 0
    train_neg = 0
    test_pos = 0
    test_neg = 0

    for i in train_acts:
        if i:
            train_pos += 1
        else:
            train_neg += 1

    for i in test_acts:
        if i:
            test_pos += 1
        else:
            test_neg += 1
    print('Training Positives: ', train_pos)
    print('Training Negatives: ', train_neg)
    print('Testing Positives:', test_pos)
    print('Testing Negatives: ', test_neg)

    sample = open(f"{output_dir}/stats_class_{output_ext}.txt", "w")
    print(output_ext)
    print('Accuracy: %5.3f' % acc, file=sample)
    print('Sensitivity: %5.3f ' % sens, file=sample)
    print('Specificity: %5.3f ' % spec, file=sample)
    print('', file=sample)
    print(confmat, file=sample)
    print('', file=sample)
    print('Training Positives: ', train_pos, file=sample)
    print('Training Negatives: ', train_neg, file=sample)
    print('Testing Positives:', test_pos, file=sample)
    print('Testing Negatives: ', test_neg, file=sample)
    sample.close()
    print("Internal and External benchmark are done\n")

    split = output_ext.split('_')[-1]
    with open(f"{output_dir}/pred_{split}.csv", 'w') as f:
        f.write(f"compound\texp_mean\tpred_{split}\n")
        for i in range(len(test_results)):
            f.write(f"{test_names[i]}\t{test_acts[i]}\t{test_results[i][0]}\n")

    return test_results, acc, sens, spec
