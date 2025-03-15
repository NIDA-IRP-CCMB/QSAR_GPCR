"""
Program: buildmodel.py

Written: Andrew Fant, Joslyn Jung

This is a python program to generate a QSAR regression model using rdkit and scipy. The molecular descriptor data is
    read and processed with extraneous descriptors removed, and the XGBoost model determines the best-performing
    parameters to be used.

input: four files.  1) named training.smi. This is the Molecules training set.
                       SMILES formatted holding the structure of the molecules in the model training set.

                    2) named training.act. This is the Activities training set.  each line has the name of one molecule
                       and its activity as a pIC50.

                    3) named testset-2.smi.
                    4) named testset-2.act.

                   N.B. molecules and activities need to be in the same order in both files.

output: generates and saves XGBoost regression model with optimized parameters in build.pickle.dat
"""

## define enviroment
import sys, os
from pathlib import Path
home = str(Path.home())
base_dir = home+'/repositories/ai-x'
core_dir = base_dir+'/core'
conf_dir = core_dir+"/keywords"
sys.path.insert(0, core_dir)
sys.path.insert(0, conf_dir)



import os
import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from molvs import Standardizer
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, confusion_matrix, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import operator
import copy
from descriptor_setup import dnames, dlist
from xgboost import XGBRegressor, XGBClassifier
from math import sqrt
import pickle
import matplotlib.pyplot as plt
import matplotlib
from misc import *

matplotlib.use('Agg')

# Initialize global variables
output_ext = ''


def read_data4buildmodel(input_basename, mode='regression'):

    # Read in molecules and activities
    molfh = open(input_basename+'.smi')
    actfh = open(input_basename+'.act')

    datadir = get_dir(input_basename)

    input_molecules = []
    input_activities = []

    delete_list = []
    change_dict = {}

    act = []

    for molline in molfh:
        line = molline[:-1].split()
        mol = Chem.MolFromSmiles(line[0])
        molname = line[1]
        input_molecules.append((mol, molname))

    for actline in actfh:
        line = actline[:-1].split()
        if mode.startswith('reg'):
            act = float(line[1])
        elif mode.startswith('class'):
            act = int(float(line[1]))
        actname = line[0]
        input_activities.append((act, actname))

    molfh.close()
    actfh.close()

    if mode.startswith('reg'):

        # Append AmyCompounds to model building
        try:
            molfh2 = open(datadir + "/AmyCompounds.smi")
            actfh2 = open(datadir + "/AmyCompounds.act")

            for molline in molfh2:
                line = molline.split()
                mol = Chem.MolFromSmiles(line[0])
                molname = line[1]
                input_molecules.append((mol, molname))

            for actline in actfh2:
                line = actline.split()
                act = float(line[1])
                actname = line[0]
                input_activities.append((act, actname))

            molfh2.close()
            actfh2.close()
        except:
            print("WARNING: no AmyCompounds.* used")

        # Molecules to remove the affinities of
        try:
            moldeletefh = open(datadir + "/to_remove.txt")
            for delline in moldeletefh:
                if '#' in delline:
                    line = delline.split('#')[0]
                else:
                    line = delline
                delitem = line[:-1]
                delete_list.append(delitem)
            moldeletefh.close()
        except:
            print("WARNING: no to_remove.txt used")

        # Molecules to change the affinities of
        try:
            molchangefh = open(datadir + "/to_change.txt")
            for changeline in molchangefh:
                if '#' in changeline:
                    line = changeline.split('#')
                else:
                    line = changeline
                workline = line.split()
                newvalue = float(workline[1])
                actname = workline[0]
                change_dict[actname] = newvalue
            molchangefh.close()
        except:
            print("WARNING: no to_change.txt used")

    return input_molecules, input_activities, delete_list, change_dict


def curate_mols(input_mols, input_acts, delete_list, change_dict):
    output_mols = []
    output_acts = []
    if len(input_mols) != len(input_acts):
        print("Error: Number of molecules not equal to number of activities")
        exit(1)
    for item in range(len(input_mols)):
        if input_mols[item][1] in delete_list:
            continue
        curr_name = input_acts[item][1]
        curr_mol = input_mols[item][0]
        curr_act = input_acts[item][0]
        if input_acts[item][1] in change_dict.keys():
            curr_act = change_dict[curr_name]
        output_mols.append((curr_mol, curr_name))
        output_acts.append((curr_act, curr_name))

    if len(output_mols) != len(output_acts):
        print("Error: curated molecules length not equal to curated activities length")
        exit(1)
    for i in range(len(output_mols)):
        if output_mols[i][1] != output_acts[i][1]:
            print("Error: curated molecules order is not the same as curated activities order")
            exit(1)
    return output_mols, output_acts


def split_data(mols, acts, test_percent, split):
    mols_train = []
    mols_test = []
    molnames_train = []
    molnames_test = []
    acts_train = []
    acts_test = []
    actnames_train = []
    actnames_test = []

    # Split molecules and activities training set into training and test sets
    m_train, m_test, a_train, a_test = train_test_split(mols, acts, test_size=test_percent, random_state=split)
    # Make a list of the names of all the molecules in the training list
    names_train = []

    for mol in m_train:
        names_train.append(mol[1])

    # Iterate over all the molecules we have read in
    for i in range(len(mols)):
        # assert mols[i][1] == acts[i][1]
        if mols[i][1] in names_train:  # is the molecule in the training set?
            mols_train.append(mols[i][0])
            molnames_train.append(mols[i][1])
            acts_train.append(acts[i][0])
            actnames_train.append(acts[i][1])
        else:  # the molecule is in the test set if it isn't in the the training set
            mols_test.append(mols[i][0])
            molnames_test.append(mols[i][1])
            acts_test.append(acts[i][0])
            actnames_test.append(acts[i][1])

    assert molnames_train == actnames_train
    assert molnames_test == actnames_test

    # Standardize structures of the training set and test set
    s = Standardizer()
    standard_mols_train = []

    for mol in mols_train:
        standard_mols_train.append(s.standardize(mol))

    standard_mols_test = []

    for mol in mols_test:
        standard_mols_test.append(s.standardize(mol))

    return standard_mols_train, molnames_train, acts_train, standard_mols_test, molnames_test, acts_test


def all_data(molecules, activities):
    training_molecules = []
    training_names = []
    training_activities = []
    for i in range(len(molecules)):
        training_molecules.append(molecules[i][0])
        training_names.append(molecules[i][1])
        training_activities.append(activities[i][0])
    return training_molecules, training_names, training_activities
       
    
def get_output_ext(mode, method, tp, rand_state, rand_split):
    global output_ext
    output_ext = "%s_%s_%.2f_%d_%d" % (mode, method, float(tp), int(rand_state), int(rand_split))
    return output_ext


def get_output_dir(mode, method, tp):
    global output_dir
    output_dir = "%s_%s_%.2f" % (mode, method, float(tp))
    return output_dir


def calc_appdom(training_set, out_model_dir, output_ext):
    import pickle
    '''
    calculate applicability domains
    :param training_set:
    :param out_model_dir:
    :param output_ext:
    :return:
    '''
    appdom_fps = []
    # output_ext = "%s_%s_%d_%d" % (mode, method, int(rand_split), int(rand_state))

    for mol in training_set:
        # fingerprint = Chem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
        fingerprint = FingerprintMols.FingerprintMol(mol)
        appdom_fps.append(fingerprint)

    distances = []

    for i in range(0, (len(appdom_fps) - 1)):
        for j in range(i + 1, len(appdom_fps)):
            # dist = 1.0 - (DataStructs.TanimotoSimilarity(appdom_fps[i], appdom_fps[j]))
            dist = 1.0 - (DataStructs.FingerprintSimilarity(appdom_fps[i], appdom_fps[j]))
            distances.append(dist)

    distances = np.array(distances)
    mean_distance = np.mean(distances)
    dev_distance = np.std(distances)

    # Andy Fant defined "applicability domains" as the following
    appdom_radius = mean_distance + dev_distance

    # Write fingerprints of training set and AD radius to pickle files for later prediction runs
    with open(out_model_dir + "/training-FPs_%s.dat" % output_ext, 'wb') as f:
        pickle.dump(appdom_fps, f)

    with open(out_model_dir + "/AD-radius_%s.dat" % output_ext, 'wb') as f:
        pickle.dump(appdom_radius, f)

    return appdom_fps, appdom_radius


# args: fps, rad, mols, names, acts
def check_appdom(*args, step):
    if len(args) < 4 or len(args) > 5:
        print("Error: incorrect number of arguments passed to check_appdom()")
        exit(1)

    appdom_fps = args[0]
    appdom_radius = args[1]
    pred_mols = args[2]
    pred_names = args[3]
    if len(args) == 5:
        pred_acts = args[4]

    accept_mols = []
    accept_names = []
    reject_mols = []
    reject_names = []

    if 'pred_acts' in locals():
        accept_acts = []
        reject_acts = []

    for i in range(len(pred_mols)):
        test_fp = FingerprintMols.FingerprintMol(pred_mols[i])
        distances = []
        for training_fp in appdom_fps:
            distances.append(1.0 - (DataStructs.FingerprintSimilarity(training_fp, test_fp)))

        distances = np.array(distances)
        if np.min(distances) <= appdom_radius:
            accept_mols.append(pred_mols[i])
            accept_names.append(pred_names[i])
            if 'pred_acts' in locals():
                accept_acts.append(pred_acts[i])
        else:
            reject_mols.append(pred_mols[i])
            reject_names.append(pred_names[i])
            if 'pred_acts' in locals() and len(args) > 4:
                reject_acts.append(pred_acts[i])

            print("Compound %s is out of the AD for this model", pred_names[i])

    if step.startswith('p'):
        if len(reject_names) == 0:
            print("No molecules rejected for prediction by AD")
        return_dict = {}

        return_dict['test_mols'] = accept_mols
        return_dict['test_names'] = accept_names
        return_dict['rej_mols'] = reject_mols
        return_dict['rej_names'] = reject_names

        if 'pred_acts' in locals() and len(args) > 4:
            return_dict['test_acts'] = accept_acts
            return_dict['rej_acts'] = reject_acts

        return return_dict
    elif step.startswith('b') or step == "same_buildmodel":
        if len(reject_names) == 0:
            print('All molecules in test set within applicability domain.')
        return accept_mols, accept_acts, accept_names, reject_mols, reject_acts, reject_names


def calc_topo_descs(mols, indexes=None):
    descs = np.zeros((len(mols), len(dlist)))
    for i in range(len(mols)):
        for j in range(len(dlist)):
            descs[i, j] = dlist[j](mols[i])
    if indexes is not None:
        # Select descriptors
        del_indexes = []
        for i in range(len(dlist)):
            if i not in indexes:
                del_indexes.append(i)

        del_indexes.reverse()

        for i in del_indexes:
            descs = np.delete(descs, [i], axis=1)

    return descs

def calc_chir_descs(mols):
    print('Chiral Descriptors are used.')
    def get_ChiralityIndex(m):
        '''

        :param m: for a given molecule
        :return: num_chiral_atoms, chirality_index
        num_chiral_atoms is the number of atoms that has chrial
        chirality_index is based on the following calculation:

           chirallity index =  sum (ai * ci) / sum (ai)  (See Benjoe's slide)

        '''

        m_copy = Chem.Mol(m)
        Chem.AssignStereochemistry(m_copy, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        chiral_atom_idx = []
        cip_val = []

        # In RDKit, the HasProp method is used to check whether a molecule (or an atom or a bond
        # within a molecule) has a specific property set.
        for atom in m_copy.GetAtoms():
            if atom.HasProp('_CIPCode'):
                cip_code = atom.GetProp("_CIPCode")
                # The _CIPCode is an internal property in RDKit used to store the CIP (Cahn-Ingold-Prelog) codes for
                # stereocenters in a molecule. These codes are used to describe the absolute configuration of
                # chiral centers as R (rectus) or S (sinister) for atoms and E/Z for double bonds.

                #assign and append value (R = 1, S = -1) and atom index
                cip_value = 1 if cip_code == 'R' else -1
                atom_idx = atom.GetIdx() + 1
                chiral_atom_idx.append(atom_idx)
                cip_val.append(cip_value)

        numerator = sum(index * cip for index, cip in zip(chiral_atom_idx, cip_val))
        denominator = sum(chiral_atom_idx)

        chiral_centers = Chem.FindMolChiralCenters(m_copy, force=True, includeUnassigned=True)
        num_chiral_atoms = len(chiral_centers)

        chirality_index = numerator / denominator if denominator != 0 else 0
        return num_chiral_atoms, chirality_index

    descs = np.zeros((len(mols), 2))
    for i in range(len(mols)):
        descs[i, 0], descs[i, 1] = get_ChiralityIndex(mols[i])

    return descs

def prune_topo_descs(mode, input_descs, acts_train, out_model_dir, output_ext):
    # output_ext = "%s_%s_%d_%d" % (mode, method, int(rand_split), int(rand_state))
    local_dnames = copy.copy(dnames)  # create a local copy of dnames
    del_indices = []  # array of indices of deleted entries from dnames

    hold = np.copy(input_descs)

    # Remove zero-variance descriptors
    for i in range((len(local_dnames) - 1), -1, -1):
        if min(input_descs[:, i]) == max(input_descs[:, i]):
            hold = np.delete(hold, [i], axis=1)
            del_indices.append(i)
            del local_dnames[i]
    descs = hold

    # Remove highly correlated descriptors
    correl = np.zeros((len(local_dnames), len(local_dnames)))
    hcpairs = []
    hcdescs = {}
    descs_to_del = []

    for i in range(len(local_dnames)):
        for j in range(len(local_dnames)):
            correl[i, j] = pearsonr(descs[:, i], descs[:, j])[0]

    for i in range((len(local_dnames) - 1)):
        for j in range(i + 1, len(local_dnames)):
            if correl[i, j] > 0.99:
                hcpairs.append((i, j))

    for pair in hcpairs:
        if pair[0] not in hcdescs.keys():
            hcdescs[pair[0]] = 1
        else:
            hcdescs[pair[0]] = hcdescs[pair[0]] + 1
        if pair[1] not in hcdescs.keys():
            hcdescs[pair[1]] = 1
        else:
            hcdescs[pair[1]] = hcdescs[pair[1]] + 1

    sorted_hcdescs = sorted(hcdescs.items(), key=operator.itemgetter(0))
    sorted_hcdescs.reverse()

    while len(sorted_hcdescs) > 0:
        foo = sorted_hcdescs[0][0]
        descs_to_del.append(foo)

        for i in range((len(hcpairs) - 1), -1, -1):
            if foo in hcpairs[i]:
                del hcpairs[i]

        hcdescs = {}
        for pair in hcpairs:
            if pair[0] not in hcdescs.keys():
                hcdescs[pair[0]] = 1
            else:
                hcdescs[pair[0]] = hcdescs[pair[0]] + 1
            if pair[1] not in hcdescs.keys():
                hcdescs[pair[1]] = 1
            else:
                hcdescs[pair[1]] = hcdescs[pair[1]] + 1

        sorted_hcdescs = sorted(hcdescs.items(), key=operator.itemgetter(1))
        sorted_hcdescs.reverse()

    descs_to_del.sort()
    descs_to_del.reverse()

    hold = np.copy(descs)

    del_indices = []
    for i in descs_to_del:
        hold = np.delete(hold, [i], axis=1)
        del_indices.append(i)
        del local_dnames[i]
    descs = hold
    if mode.startswith('reg'):
        # Select predictive variables for further modeling
        acts_train_binarized = copy.copy(acts_train)
        descs_to_del = []

        for foo in range(len(acts_train_binarized)):
            if float(acts_train_binarized[foo]) < 5:
                acts_train_binarized[foo] = 0
            else:
                acts_train_binarized[foo] = 1
        # print(len(set(acts_train_binarized)))

        # for selectivity. Selectivity datasets don't have compounds with pKi > 5,
        # so everything in acts_train_binarized is 0, and causes code to break
        if len(set(acts_train_binarized)) == 1:
            for foo in range(len(acts_train_binarized)):
                if float(acts_train_binarized[foo]) < 0:
                    acts_train_binarized[foo] = 0
                else:
                    acts_train_binarized[foo] = 1
        else:
            for desc_num in range(len(local_dnames)):
                aroc = roc_auc_score(acts_train_binarized, descs[:, desc_num])
                if aroc < 0.55 and aroc > 0.45:
                    descs_to_del.append(desc_num)

        descs_to_del.reverse()
        hold = np.copy(descs)

        del_indices = []
        for i in descs_to_del:
            hold = np.delete(hold, [i], axis=1)
            del_indices.append(i)
            del local_dnames[i]
    cleaned_descriptors = hold

    # Get complete list of indices of descriptors in use
    indices = []

    for desc in local_dnames:
        i = dnames.index(desc)
        indices.append(i)
    indices.sort()

    # Write list of descriptor indices and significant bits to files
    with open(out_model_dir + "/indices_%s.dat" % output_ext, 'wb') as f:
        pickle.dump(indices, f)

    return cleaned_descriptors, indices, local_dnames


def calc_phore_descs(mols, significant_bits=None, testing=False):
    fp_holding = []
    accumulated_bits_set = {}

    for mol in mols:
        fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        fp_holding.append(fp)
        if significant_bits is not None:
            bits_set = list(fp.GetOnBits())
            for fp_bit in bits_set:
                if fp_bit not in accumulated_bits_set.keys():
                    accumulated_bits_set[fp_bit] = 1
                else:
                    accumulated_bits_set[fp_bit] = accumulated_bits_set[fp_bit] + 1

    if significant_bits is not None:
        phore_descs = np.zeros((len(mols), len(significant_bits)))

        for mol_num in range(len(mols)):
            for bit_num in range(len(significant_bits)):
                if significant_bits[bit_num] in fp_holding[mol_num].GetOnBits():
                    phore_descs[mol_num, bit_num] = 1
        if testing:
            return "significant_bits: %d" % len(significant_bits), "fp_descriptors: %s" % str(phore_descs.shape)
        print("significant_bits:", len(significant_bits))
        print("fp_descriptors:", phore_descs.shape)
        return phore_descs
    else:
        return fp_holding


def prune_phore_descs(input_descs, out_model_dir, output_ext):
    import pickle
    # output_ext = "%s_%s_%d_%d" % (mode, method, int(rand_split), int(rand_state))
    # Extract pharmacophore bits that occur often enough to be useful
    accumulated_bits_set = {}

    for fp in input_descs:
        bits_set = list(fp.GetOnBits())
        for fp_bit in bits_set:
            if fp_bit not in accumulated_bits_set.keys():
                accumulated_bits_set[fp_bit] = 1
            else:
                accumulated_bits_set[fp_bit] = accumulated_bits_set[fp_bit] + 1

    significant_bits = []
    all_set_bits = list(accumulated_bits_set.keys())
    all_set_bits.sort()
    fp_names = []

    for bit in all_set_bits:
        if accumulated_bits_set[bit] >= 100:
            significant_bits.append(bit)
            bit_name = 'Ph2D_' + str(bit)
            fp_names.append(bit_name)

    nmols = np.shape(input_descs)[0]

    fp_descriptors = np.zeros((nmols, len(significant_bits)))

    for mol_num in range(nmols):
        for bit_num in range(len(significant_bits)):
            if significant_bits[bit_num] in input_descs[mol_num].GetOnBits():
                fp_descriptors[mol_num, bit_num] = 1

    print("significant_bits:", len(significant_bits))
    print("fp_descriptors:", fp_descriptors.shape)


    if out_model_dir:
        with open(out_model_dir + "/sigbits_%s.dat" % output_ext, 'wb') as f:
            pickle.dump(significant_bits, f)

    return fp_descriptors, significant_bits, fp_names


def build_model(mode, method, rand_state, training_descs, training_acts, out_model_dir, output_ext):
    # output_ext = "%s_%s_%d_%d" % (mode, method, int(rand_split), int(rand_state))
    if method == 'xgb':
        # Parameter sets as specified in Noskov and Wacker
        # For parameters, random_state is a random seed, default is 0
        # If random_state is not changed, then the results will be the same
        # parameters_xgb = dict(colsample_bytree=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        #                       subsample=[0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        #                       max_depth=[2, 3, 4, 5, 6, 7, 8],
        #                       learning_rate=[0.001, 0.01, 0.02, 0.08, 0.1])
        parameters_xgb = dict(colsample_bytree=[0.25, 0.5, 0.75, 1],
                              subsample=[0.4, 0.6, 0.8, 1],
                              max_depth=[5, 7, 9, 11],
                              learning_rate=[0.08, 0.1, 0.3, 0.5, 1])
        # parameters_xgb = dict(colsample_bytree=[1],
        #                       subsample=[ 1],
        #                       max_depth=[1],
        #                       learning_rate=[1])
        # Train XGBoost model and estimate best parameters using GridSearchCV with 10-fold cross validation
        if mode.startswith('reg'):
            mode = 'reg'
            reg = GridSearchCV(XGBRegressor(random_state=rand_state), parameters_xgb, cv=10,
                               scoring='neg_mean_squared_error', n_jobs=-1)
        elif mode.startswith('class'):
            mode = 'class'
            reg = GridSearchCV(XGBClassifier(random_state=rand_state), parameters_xgb, cv=10, scoring='accuracy',
                               n_jobs=-1)

        # Uncomment the following line for y-randomization tests only
        # random.shuffle(acts_train)

        reg.fit(training_descs, training_acts)
        best_params = reg.best_params_
        candidate_model = reg.best_estimator_
        score = reg.best_score_
        if mode.startswith('class'):
            # oob = model.oob_score_
            print("Best model params: ", best_params)
            print("Final Accuracy Score: ", score)
            # print("Out-Of-Bag error for this model is %5.3f" % oob)

        # Save model generated by GridSearchCV with XGBoost regressor to a file
        with open(str(out_model_dir) + "/model_%s.dat" % output_ext, 'wb') as file:
            pickle.dump(candidate_model, file)
        #pickle.dump(candidate_model, open(str(out_model_dir) + "/model_%s.dat" % output_ext, "wb"))
        if mode.startswith('reg'):
            print("Best parameters score:", str(best_params), abs(score))
    elif method == 'rf':
        # Train RandomForest model and estimate best parameters using GridSearchCV with 10-fold cross validation
        if mode.startswith('reg'):
            mode = 'reg'
            #parameters = {'n_estimators': [30, 100, 300, 1000, 3000, 10000, 30000, 100000]}
            parameters = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
            mod = GridSearchCV(RandomForestRegressor(random_state=rand_state,oob_score=True),
                               parameters,
                               cv=10,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1)
        elif mode.startswith('class'):
            mode = 'class'
            parameters = {'n_estimators': [30, 100, 300, 1000, 3000, 10000, 30000, 100000],
                          'max_features': ['auto', 'sqrt', 'log2']
                          }
            mod = GridSearchCV(RandomForestClassifier(random_state=rand_state,oob_score=True),
                               parameters,
                               cv=10,
                               scoring='accuracy',
                               n_jobs=-1)
        mod.fit(training_descs, training_acts)

        best_params = mod.best_params_
        candidate_model = mod.best_estimator_
        score = mod.best_score_
        oob = candidate_model.oob_score_

        print("Best model params: ", best_params)
        if mode.startswith('class'):
            print("Final Accuracy Score: ", score)
        else:
            print("Final MSE Score: ", score)
        print("Out-Of-Bag error for this model is %5.3f" % oob)

        # Save model generated by GridSearchCV with XGBoost regressor to a file
        with open(out_model_dir + "/model_%s.dat" % output_ext, 'wb') as file:
            pickle.dump(candidate_model, file)
        #pickle.dump(candidate_model, open(out_model_dir + "/model_%s.dat" % output_ext, "wb"))
    return candidate_model, score, best_params


def predict_model(candidate_model, train_names, train_acts, test_descs, test_acts, test_names, split,
                  out_model_dir, mode, method, rand_state, Verbose=False):
    if mode.startswith('reg'):
        # Make prediction on test set using model
        y_true, y_pred = test_acts, candidate_model.predict(test_descs)

        f = open(out_model_dir + "/stat_" + str(split), "w")
        f.write('random_split %s R2 %s RMSE %s MSE %s Rp %s Rs %s\n' % (
            split, R2(y_true, y_pred), RMSE(y_true, y_pred), MSE(y_true, y_pred), pearsonR(y_true, y_pred), spearmanr(y_true, y_pred)[0]))
        f.close()

        with open(f"{out_model_dir}/pred_{split}.csv", 'w') as f:
            f.write(f"compound\texp_mean\tpred_{split}\n")
            for i in range(len(y_pred)):
                f.write(f"{test_names[i]}\t{y_true[i]}\t{y_pred[i]}\n")

        if Verbose:
            # Evaluate model using measures of fit
            print("R2: %2.3f" % R2(y_true, y_pred))
            print("RMSE: %2.3f" % RMSE(y_true, y_pred))
            print("MSE: %2.3f" % MSE(y_true, y_pred))
            print("Rp: %2.3f" % pearsonR(y_true, y_pred))
            print("Rs: %2.3f" % spearmanr(y_true, y_pred))

        # Save model generated by GridSearchCV with XGBoost regressor to a file
        # pickle.dump(candidate_model,
        #             open(out_model_dir + "/model_reg_%s_%d_%d.dat" % (ml_type, split, rand_state), "wb"))

        # Plot data: y_pred vs. y_true
        plt.figure()
        npoints = len(y_pred)
        # Determine data boundary
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        bounds = (int(min(np.min(y_true_array), np.min(y_pred_array)) - 0.15 * np.min(y_pred_array)),
                  int(max(np.max(y_true_array), np.max(y_pred_array)) + 0.15 * np.max(y_pred_array)))
        plt.xlim(bounds)
        plt.ylim(bounds)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.plot([0, 1], [0, 1], color='red', linewidth=2, linestyle='-', alpha=0.25, transform=plt.gca().transAxes)
        # Annotations for metric scores
        text = f"Test Points = {npoints}\nR2 = {R2(y_true, y_pred):.3f}\nRMSE = {RMSE(y_true, y_pred):.3f}\nMSE = {MSE(y_true, y_pred):.3f}\nRp = {pearsonR(y_true, y_pred):.3f}\nRs = {spearmanr(y_true, y_pred)[0]:.3f}"
        plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
        plt.scatter(y_true, y_pred)
        plt.title("Test Data Results")
        plt.xlabel("True Output")
        plt.ylabel("Predicted Output")
        plt.savefig(out_model_dir + "/graph_%d.png" % split)
        plt.clf()

        return R2(y_true, y_pred), RMSE(y_true, y_pred), MSE(y_true, y_pred)

    elif mode.startswith('class'):
        test_results = candidate_model.predict(test_descs)

        acc = accuracy_score(test_acts, test_results)
        sens = recall_score(test_acts, test_results, pos_label=1)
        spec = recall_score(test_acts, test_results, pos_label=0)
        f1 = f1_score(test_acts, test_results)
        auc = roc_auc_score(test_acts, test_results)

        print('Accuracy: %5.3f' % acc)
        print('Sensitivity: %5.3f ' % sens)
        print('Specificity: %5.3f ' % spec)
        print('F1: %5.3f ' % f1)
        print('AUC: %5.3f ' % auc)


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

        sample = open(out_model_dir + '/prediction_class_%s_%d_%d.dat' % (method, split, rand_state), 'w')
        print('Accuracy: %5.3f' % acc, file=sample)
        print('Sensitivity: %5.3f ' % sens, file=sample)
        print('Specificity: %5.3f ' % spec, file=sample)
        print('F1: %5.3f ' % f1, file=sample)
        print('AUC: %5.3f ' % auc, file=sample)

        print('', file=sample)
        print(confmat, file=sample)
        print('', file=sample)
        print('Training Positives: ', train_pos, file=sample)
        print('Training Negatives: ', train_neg, file=sample)
        print('Testing Positives:', test_pos, file=sample)
        print('Testing Negatives: ', test_neg, file=sample)
        sample.close()

        # save the predictions themselves
        with open(f"{out_model_dir}/pred_{split}.csv", 'w') as f:
            f.write(f"compound\texp_mean\tpred_{split}\n")
            for i in range(len(test_results)):
                f.write(f"{train_names[i]}\t{test_acts[i]}\t{test_results[i]}\n")

        return test_results, acc


def MSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse


def R2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2

def pearsonR(y_true, y_pred):
    R = pearsonr(y_true,y_pred)[0]
    return R


def RMSE(y_true, y_pred):
    rmse = sqrt(MSE(y_true, y_pred))
    return rmse


def read_mols(mode, method, basename, output_ext, datadir='Default', modeldir='Default'):
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

    # output_ext = "%s_%s_%d_%d" % (mode, method, int(rand_state), int(rand_split))
    model_filename = "model_%s.dat" % output_ext
    index_filename = "indices_%s.dat" % output_ext
    appdom_fp_filename = "training-FPs_%s.dat" % output_ext
    appdom_rad_filename = "AD-radius_%s.dat" % output_ext
    feature_filename = "feature_names_%s.dat" % output_ext

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
    model_file = os.path.join(modeldir, model_filename)
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    index_file = os.path.join(modeldir, index_filename)
    with open(index_file, 'rb') as f:
        indexes = pickle.load(f)
    appdom_fp_file = os.path.join(modeldir, appdom_fp_filename)
    with open(appdom_fp_file, 'rb') as f:
        appdom_fps = pickle.load(f)
    appdom_rad_file = os.path.join(modeldir, appdom_rad_filename)
    with open(appdom_rad_file, 'rb') as f:
        appdom_radius = pickle.load(f)
    feature_file = os.path.join(modeldir, feature_filename)
    with open(feature_file, 'rb') as f:
        feature_names = pickle.load(f)

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
    return_dict['model'] = loaded_model
    return_dict['inds'] = indexes
    return_dict['sigbits'] = significant_bits
    if mode.startswith('class') and 'activities' in locals():
        return_dict['activities'] = acts_train
    return_dict['ad_fps'] = appdom_fps
    return_dict['ad_radius'] = appdom_radius
    return_dict['feature_names'] = feature_names

    return return_dict


def make_preds(*args, mode):

    if len(args) < 3 and len(args) > 4:
        print("Error: incorrect number of arguments passed to check_appdom()")
    mol_names = args[0]
    descs = args[1]
    saved_model = args[2]

    if len(args) == 4:
        if mode.startswith('reg'):
            split = args[3]
        elif mode.startswith('class'):
            y_true = args[3]

    # some of the desc elements are nans and cause errors. Let's grab the row index of these nans. There are not many.
    # indices_to_replace = []
    # x, y = np.where(~np.isfinite(descs))  # x is row number, y is col number
    # indices_to_replace = list(set(x))  # grab the index of the compound #
    # failed_cols = list(set(y))
    # descs[indices_to_replace, :] = 0
    #
    # # Failed index test (obscure metals and salts fail in the descriptor extraction)
    # print("Indexes failed: ", indices_to_replace)
    # ls_features_failed = []
    # for index in failed_cols:
    #     ls_features_failed.append(feature_names[index])
    # print("Feature columns failed: ", failed_cols, ls_features_failed)
    # ls_mols = []
    # for index in indices_to_replace:
    #     mol = mol_names[index]
    #     ls_mols.append(mol)
    # print("Compound failed: ", ls_mols)

    # predictions
    descs = descs.astype('float32')
    y_pred = saved_model.predict(descs)
    if mode.startswith("class"):
        y_pred = np.where(y_pred > 0.5, 1, 0)
    # for index in indices_to_replace:
    #     y_pred[index] = 0


    # nan_indices = (np.where(np.isnan(descs)))
    # compounds_with_nan = [mol_names[index] for index in nan_indices[0]]
    # print(compounds_with_nan)
    # 
    # nan_indices = np.isnan(descs)
    # mean_value = np.nanmean(descs)  # Calculate the mean of non-NaN values
    # descs[nan_indices] = mean_value

    prediction_results = {}
    prediction_results['predictions'] = y_pred

    if 'y_true' in locals() and mode.startswith('class'):
        print("Molecule\tActual Act.\tPredicted Act.")
        for out_line in range(len(mol_names)):
            print(mol_names[out_line], "\t", y_true[out_line], "\t\t", y_pred[out_line])
        print("")
        print("Accuracy Score:",accuracy_score(y_true,y_pred))
        print("")
        confmat = confusion_matrix(y_true,y_pred, labels=[1,0])
        print(confmat)
        prediction_results['accuracy'] = accuracy_score(y_true, y_pred)
    else:
        print("Molecule\tPredicted Act.")
        for out_line in range(len(mol_names)):
            print(mol_names[out_line], "\t", y_pred[out_line])

    print('completed')

    return prediction_results


def summarize_preds(*args):
    if len(args) != 2:
        print("Error: incorrect number of arguments passed to summarize_preds()")

    names = args[0]
    preds_list = args[1]

    nsplits = len(preds_list)
    npreds = len(preds_list[0])
    pivot_list = []
    for i in range(npreds):
        pivot_list.append([])

    for pred in range(npreds):
        for trial in range(nsplits):
            pivot_list[pred].append(preds_list[trial][pred])

    predmean_list = []
    predstd_list = []

    for i in range(npreds):
        predmean_list.append(np.mean(pivot_list[i]))
        predstd_list.append(np.std(pivot_list[i]))

    print("Compound\tPredicted\tStdDev")
    for i in range(len(names)):
        print("%s \t %2.3f \t %2.3f" % (names[i], predmean_list[i], predstd_list[i]))

    return names, predmean_list, predstd_list


#import pickle5 as pickle
# this is used to solve the "ValueError: unsupported pickle protocol: 5"
def retrieve_training_data(filename, tp):
    with open(filename, 'rb') as handle:
        dict_data = pickle.load(handle)

    train_names = dict_data['train_names']
    train_descs = dict_data['train_descs']
    train_acts = dict_data['train_acts']


    ## Jason made this function, for old models there is no following information, we need to byapss those.
    try:
        topo_names = dict_data["topo_names"]
    except:
        topo_names = 0
    try:
        phore_names = dict_data["phore_names"]
    except:
        phore_names = 0
    try:
        feature_names = dict_data["feature_names"]
    except:
        feature_names = 0

    if tp > 0:
        test_acts = dict_data['test_acts']
        test_descs = dict_data['test_descs']
        test_names = dict_data["test_names"]
    elif tp == 0:
        test_acts = 0
        test_descs = 0
        test_names = 0

    return train_descs, test_descs, train_acts, test_acts, train_names, test_names, \
           topo_names, phore_names, feature_names


def get_training_dataset_ML(in_filename, mode, method, tp, output_dir, rand_state, random_split, stage, chiral_descs=False):
    stage = "buildmodel"
    output_ext = get_output_ext(mode, method, tp, rand_state, random_split)

    mols, acts, deletes, changes = read_data4buildmodel(in_filename, mode)
    mols, acts = curate_mols(mols, acts, deletes, changes)

    # using split_data (tp!=0.00) or all_data (tp=0.00)
    if tp > 0.00:
        train_mols, train_names, train_acts, \
        test_mols, test_names, test_acts = split_data(mols, acts, tp, random_split)
    else:
        train_mols, train_names, train_acts = all_data(mols, acts)

    ad_fps, ad_rad = calc_appdom(train_mols, output_dir, output_ext)

    if tp > 0:
        test_mols, test_acts, test_names, test_mols_reject, test_acts_reject, test_names_reject \
            = check_appdom(ad_fps, ad_rad, test_mols, test_names, test_acts, step=stage)

    # feature_topo
    train_topo_descs = calc_topo_descs(train_mols)
    train_topo_descs, topo_index, topo_names = prune_topo_descs(mode, train_topo_descs, train_acts, output_dir, output_ext)

    if chiral_descs:
        # Get chirality descriptors
        train_chir_descs = calc_chir_descs(train_mols)
        train_topo_descs = np.concatenate((train_topo_descs, train_chir_descs), axis=1)
        topo_names.append('nChiral')
        topo_names.append('ChirIdx')

    train_phore_descs = calc_phore_descs(train_mols)
    train_phore_descs, phore_sigbits, phore_names = prune_phore_descs(train_phore_descs, output_dir, output_ext)
    train_descs = np.concatenate((train_topo_descs, train_phore_descs), axis=1)
    if tp > 0:
        test_topo_descs = calc_topo_descs(test_mols, topo_index)
        if chiral_descs:
            test_chir_descs = calc_chir_descs(test_mols)
            test_topo_descs = np.concatenate((test_topo_descs, test_chir_descs), axis=1)
        test_phore_descriptors = calc_phore_descs(test_mols, phore_sigbits)
        test_descs = np.concatenate((test_topo_descs, test_phore_descriptors), axis=1)

    # Save data, Export train_descs and combined feature_names data
    pickle.dump(train_descs, open(output_dir + f"/train_descs_{rand_state}_{random_split}.dat", "wb"))
    pickle.dump(train_descs, open(output_dir + f"/train_descs_{output_ext}.dat", "wb"))
    feature_names = topo_names + phore_names
    pickle.dump(feature_names, open(output_dir + f"/feature_names_{rand_state}_{random_split}.dat", "wb"))
    pickle.dump(feature_names, open(output_dir + f"/feature_names_{output_ext}.dat", "wb"))

    if tp == 0:
        test_mols, test_names, test_acts, test_descs, test_names = 0, 0, 0, 0, 0
    dict_data = {'train_names': train_names, 'train_descs': train_descs, 'train_acts': train_acts,
                 "topo_names": topo_names, "phore_names": phore_names,
                 "feature_names": feature_names,  'test_acts': test_acts,
                 'test_descs': test_descs, 'test_names': test_names}
    with open(f'{output_dir}/dict_data_{rand_state}_{random_split}.pickle', 'wb') as handle:
        pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_descs, train_acts, train_names, test_descs, test_acts, test_names
