import os, sys
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
sys.path.insert(0, core_dir)

import glob
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from statistics import mean, stdev
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from misc import check_output_dir
from deeplearning import stats


def get_target_information(target):
    if target == "SERT":
        target1 = "SERT"
        assaydefinition1 = "inhibitor"
        measurement1 = "Ki"
        target2 = "DAT"
        assaydefinition2 = "inhibitor"
        measurement2 = "Ki"

        target1_source = home+ f"/repositories/ai-SERT/datasets/chembl32/dataset_all_SERT_binding_Ki/pubdata"
        target2_source = home+ f"/repositories/ai-DAT/datasets/chembl32/dataset_all_DAT_binding_Ki/pubdata"

    elif target == "DR":
        target1 = "D3"
        assaydefinition1 = "antagonist"
        measurement1 = "Ki"
        target2 = "D2"
        assaydefinition2 = "antagonist"
        measurement2 = "Ki"

        datadir = home + "/repositories/ai-DR/datasets/chembl_datasets/"
        target1_source = datadir + "C33/dataset_D3_antagonist_Ki/pubdata"
        target2_source = datadir + "C33/dataset_D2_antagonist_Ki/pubdata"

    return target1, target2, assaydefinition1, assaydefinition2, measurement1, measurement2, target1_source, target2_source


def get_dataset_dict(target1, assaydefinition1, measurement1, target2, assaydefinition2, measurement2):
    dict_dataset = {"models_"+target1: f"dataset_{target1}_{assaydefinition1}_{measurement1}",
                   "models_"+target2: f"dataset_{target2}_{assaydefinition2}_{measurement2}",
                   f"models_{target1}_overlap": f"dataset_{target1}_overlap_{assaydefinition1}_{measurement1}",
                   f"models_{target2}_overlap": f"dataset_{target2}_overlap_{assaydefinition2}_{measurement2}",
                   f"models__ratio_{target1}{target2}": f"dataset__ratio_{target1}_{assaydefinition1}_{measurement1}_"
                                                        f"{target2}_{assaydefinition2}_{measurement2}"}
    return dict_dataset


def get_df(path):
    '''
    Given path, returns a big df containing all three pieces of information: chembl id, pki, and canonical_smiles
    :param path: str, path to the rawdata, ie. pubdata
    :return: df, dataframe containing the rawdata
    '''
    df_1 = pd.read_csv(path + '.act', sep='\t', header=None)
    df_2 = pd.read_csv(path + '.smi', sep='\t', header=None)
    df = pd.concat([df_1, df_2], axis=1, join="inner").iloc[:, :-1]
    df.columns = ['CHEMBL', 'pKi', 'canonical_smiles']
    return df


def get_mol_dup_pairs(df1, df2):
    '''
    Given two dfs, uses Tanimoto Similarity (>0.999) and returns the similar indexes as mol_dup_pairs
    :param df1: dataframe, the first rawdata, must have a column called 'canonical_smiles'
    :param df2: dataframe, the second rawdata, must have a column called 'canonical_smiles'
    :return: mol_dup_pairs, paired sets
    '''
    # Fingerprint
    fingerprints_df1 = []
    fingerprints_df2 = []
    for i in range(len(df1)):
        mol_in = Chem.MolFromSmiles(df1['canonical_smiles'][i])
        fp = Chem.GetMorganFingerprintAsBitVect(mol_in, 2, nBits=2048)  # set nbits equal to what you will use in model?
        fingerprints_df1.append(fp)

    for j in range(len(df2)):
        mol_in = Chem.MolFromSmiles(df2['canonical_smiles'][j])
        fp = Chem.GetMorganFingerprintAsBitVect(mol_in, 2, nBits=2048)  # set nbits equal to what you will use in model?
        fingerprints_df2.append(fp)

    # Overlap (middle of Venn Diagram). Get similarity comparison
    mol_dup_pairs = []
    for i in range(len(df1)):
        for j in range(len(df2)):
            if DataStructs.TanimotoSimilarity(fingerprints_df1[i], fingerprints_df2[j]) > 0.999:
                mol_dup_pairs.append((i, j))
    return mol_dup_pairs


def get_overlap(target1, target2, mol_dup_pairs):
    '''
    Gets the overlapping similar compounds between d2 and d3 by using mol_dup_pairs
    :param d2: df, contains 3 columns of rawdata, must contain a column called 'pKi_D2'
    :param d3: df, contains 3 columns of rawdata, must contain a column called 'pKi_D3'
    :param mol_dup_pairs: paired set, each pair refer to indexes of the similar compounds
    :return: df_overlap, dataframe, the overlapping similar compounds between d2 and d3
    '''
    ls_i = []
    ls_j = []
    for i, j in mol_dup_pairs:
        ls_i.append(i)
        ls_j.append(j)
    target1_overlap = target1.loc[ls_i].reset_index(drop=True)
    target2_overlap = target2.loc[ls_j].reset_index(drop=True)

    ls_target1_pki = list(target1_overlap['pKi'])
    ls_target2_pki = list(target2_overlap['pKi'])
    ls_ratio = []
    for i in range(0, len(target1_overlap)):
        ls_ratio.append(ls_target1_pki[i] - ls_target2_pki[i]) # target1 pKi - target2 pKi

    # get overlapping df
    target1_overlap = target1_overlap.add_suffix('_1') # adds suffix to every column
    target2_overlap = target2_overlap.add_suffix('_2') # adds suffix to every column
    df_overlap = pd.concat([target1_overlap, target2_overlap], axis=1)
    df_overlap['ratio'] = ls_ratio
    return df_overlap


def get_similar_indexes(mol_dup_pairs, n=200):
    '''
    Given mol_dup_pairs, returns a list of index with 'n' number of indexes (that corresponds to indexes in dataframe d2 and d3)
    :param mol_dup_pairs: paired set, the set refers to the index with similar compounds between d2 and d3
    :param n: int, 'n' number of indexes that we want to extract
    :return: list of indexes that correspond to the similar indexes among d2 and d3
    '''
    ls_d2_indexes = []
    ls_d3_indexes = []
    ls_overlap_indexes = []
    ls_indexes = list(range(0, len(mol_dup_pairs)))
    for i in range(0, n):
        index = np.random.choice(ls_indexes)
        d2_index = mol_dup_pairs[index][0]
        d3_index = mol_dup_pairs[index][1]
        ls_d2_indexes.append(d2_index)
        ls_d3_indexes.append(d3_index)
        ls_overlap_indexes.append(index)
        ls_indexes.remove(index)

    return ls_d2_indexes, ls_d3_indexes, ls_overlap_indexes


def training_validation_datasplit(df, ls_indexes):
    '''
    Splits the df into training and validation df. The indexes refer to the validation indexes.
    :param df: dataframe
    :param ls_indexes: list of indexes, refers to the validation indexes
    :return: Splits the original df into two df, training and validation df.
    '''
    df_training = df.drop(df.index[ls_indexes])
    df_validation = df.iloc[ls_indexes]

    if 'ratio' in df.columns:
        df_training['ratio'] = df_training['ratio'].astype(float).round(2)
        df_validation['ratio'] = df_validation['ratio'].astype(float).round(2)

    df_training.reset_index(drop=True)
    df_validation.reset_index(drop=True)

    return df_training, df_validation


def split_overlapped_df(df_training, df_validation):
    # df_training_overlap1 = df_training[['CHEMBL_1', 'pKi_1', 'canonical_smiles_1', "classification_1"]]
    # df_validation_overlap1 = df_validation[['CHEMBL_1', 'pKi_1', 'canonical_smiles_1', "classification_1"]]
    # df_training_overlap2 = df_training[['CHEMBL_2', 'pKi_2', 'canonical_smiles_2', "classification_2"]]
    # df_validation_overlap2 = df_validation[['CHEMBL_2', 'pKi_2', 'canonical_smiles_2', "classification_2"]]
    df_training_overlap1 = df_training[['CHEMBL_1', 'pKi_1', 'canonical_smiles_1']]
    df_validation_overlap1 = df_validation[['CHEMBL_1', 'pKi_1', 'canonical_smiles_1']]
    df_training_overlap2 = df_training[['CHEMBL_2', 'pKi_2', 'canonical_smiles_2']]
    df_validation_overlap2 = df_validation[['CHEMBL_2', 'pKi_2', 'canonical_smiles_2']]

    # remove "_1" and "_2" from every column names
    dfs = [df_training_overlap1, df_validation_overlap1, df_training_overlap2, df_validation_overlap2]
    for df in dfs:
        df.columns = ['CHEMBL', 'pKi', 'canonical_smiles']

    return df_training_overlap1, df_validation_overlap1, df_training_overlap2, df_validation_overlap2


def save_file(df, filename, mode):
    if mode == "reg":
        df_act = df[["CHEMBL",'pKi']]
    elif mode == "class":
        # df = df[df.classification != 'nan']
        df = df.dropna(subset=['classification']).reset_index(drop=True)
        df_act = df[["CHEMBL", 'classification']]
    df_smi = df[['canonical_smiles','CHEMBL']]
    df_act.to_csv(filename+'.act', sep='\t', index=False, header = False)
    df_smi.to_csv(filename+'.smi', sep='\t', index=False, header = False)


def save_one_dataset(output_dir, df_training, df_validation, suffix):
    """
    Used to save individual models and overlapping datasets.
    NOT used to save ratio datasets.
    """
    mode = "reg"
    # training dataset
    filename = output_dir+'pubdata'+suffix
    save_file(df_training, filename, mode)
    # validation dataset
    filename = output_dir+'val'+suffix
    save_file(df_validation, filename, mode)

    # mode = "class"
    # # training dataset
    # filename = output_dir+'pubdata_class'+suffix
    # save_file(df_training, filename, mode)
    # # validation dataset
    # filename = output_dir+'val_class'+suffix
    # save_file(df_validation, filename, mode)


def classify(df):
    '''
    Using 'ratio' as a column, returns a df with a column 'class' with classification values. This is specific to D3.
    Below -1 is D3 selective (1)
    Above -0.5 is D3 nonselective (0)
    In between are 'nan'
    :param df: dataframe, requires a column called "ratio" - this is D2 pKi minus D3 pKi.
    :return: df with a column called "class", with 1s and 0s for D3 selective, D3 nonselective.
    Reference: https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/
    '''
    conditions = [
        (df['ratio'] <= -1),  # below -1 is D3 selective - '1'
        (df['ratio'] >= -0.5),  # above -0.5 is D3 nonselective - '0'
        (df['ratio'] < -0.5) & (df['ratio'] > -1)  # in between those two are 'nan' - we remove these
    ]

    # create a list of the values we want to assign for each condition
    values = [1, 0, 'nan']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['ratio_class'] = np.select(conditions, values)
    df = df[pd.to_numeric(df['ratio_class'], errors='coerce').notnull()]  # removes rows with 'nan', only keeps 1 and 0s

    return df


def save_ratios(output_dir, df_training, df_validation, suffix):
    def save_file_ratio(df, filename, mode):
        if mode == "reg":
            df_act = df[['CHEMBL_1', 'ratio']]
        elif mode == "class":
            df_act = df[['CHEMBL_1', 'ratio_class']]
        df_smi = df[['canonical_smiles_1', 'CHEMBL_1']]
        df_act.to_csv(filename + '.act', sep='\t', index=False, header=False)
        df_smi.to_csv(filename + '.smi', sep='\t', index=False, header=False)

    # regression datasets
    save_file_ratio(df_training, output_dir+'pubdata'+suffix, "reg")
    save_file_ratio(df_validation, output_dir + 'val' + suffix, "reg")

    # classification datasets
    df_train_class = classify(df_training)
    df_val_class = classify(df_validation)
    save_file_ratio(df_train_class, output_dir+'pubdata_class'+suffix, "class")
    save_file_ratio(df_val_class, output_dir + 'val_class' + suffix, "class")


def check_similarity_within_df(df):
    fingerprints_df = []
    for i in range(len(df)):
        mol_in = Chem.MolFromSmiles(df['canonical_smiles'][i])
        fp = Chem.GetMorganFingerprintAsBitVect(mol_in, 2, nBits=2048)  # set nbits equal to what you will use in model?
        fingerprints_df.append(fp)
    ls_indexes=[]
    for i in range(len(fingerprints_df)):
        for j in range(i+1, len(fingerprints_df)):
            if DataStructs.TanimotoSimilarity(fingerprints_df[i], fingerprints_df[j]) > 0.999:
                print(i)
                ls_indexes.append(i)
    df = df.drop(df.index[ls_indexes])
    return df


def stats_df(df):
    # print('len(df) =', len(df))
    y_pred = list(df['pred_mean'])
    y_true = list(df['exp_mean'])
    R2, Rp, Rs, MSE, RMSE = stats(y_true, y_pred)
    return R2, Rp, Rs, MSE, RMSE


def plot_validation(df, xymin = '', xymax = ''):
    labels = True
    x = df[f'exp_mean']
    xerr = df[f'exp_stdev']
    y = df['pred_mean']
    yerr = df['pred_stdev']

    R2, Rp, Rs, MSE, RMSE = stats_df(df)

    # plt.set_aspect('equal', adjustable='box')
    fig, axs = plt.subplots()
    axs.errorbar(x, y, yerr, xerr, fmt='d', markersize='3',
                 label=f'R2: {round(R2, 3)}; Rp: {round(Rp, 3)}; Rs: {round(Rs, 3)}', elinewidth=1,
                 color='green')
    axs.set_aspect('equal', adjustable='box')
    if xymin == '' and xymax == '':
        pass
    else:
        axs.set_xlim([xymin, xymax])
        axs.set_ylim([xymin, xymax])

    if labels == True:
        axs.tick_params(top=True, bottom=True, left=True, right=True,
                        labelleft=True, labelbottom=True)
        #         axs.set_title(title,  fontsize=14)
        axs.set_xlabel('Experimental pKi', fontsize=15)
        axs.set_ylabel('Predicted pKi', fontsize=15)
        axs.annotate(f'R2: {round(R2, 3)}; Rp: {round(Rp, 3)}; Rs: {round(Rs, 3)}', xy=(170, 210),
                     xycoords='axes points',
                     size=10, ha='right', va='top', bbox=dict(boxstyle='square', fc='w', lw=0.175))
        axs.tick_params(top=True, bottom=True, left=True, right=True,
                        labelleft=True, labelbottom=True)
    else:
        axs.set_xticklabels([])
        axs.set_yticklabels([])
        axs.tick_params(top=True, bottom=True, left=True, right=True,
                        labelleft=False, labelbottom=False)
        axs.margins(x=0)
        axs.margins(y=0)
# fig.savefig(filename, bbox_inches='tight', transparent=True, pad_inches=0)


def predict_dnn(path):
    '''
    Returns the predicted data in the path given as a df
    :param path:
    :return:
    '''
    df = pd.read_csv(path, sep = '\t')
    df.columns = ['compound', 'pred_mean', 'pred_stdev', 'exp_mean']
    df['exp_stdev'] = 0

    return df


def predict_d2d3(path, d2, d3):
    '''
    Returns df of D2 minus D3 predicted values
    :param path:
    :param d2:
    :param d3:
    :return:
    '''
    df = pd.read_csv(path, sep = '\t', header = None)
    df.columns = ['CHEMBL', 'exp_mean']
    df['pred_mean'] = d2['pred_mean'] - d3['pred_mean']
    df['pred_stdev'] = d2['pred_stdev'] - d3['pred_stdev']
    df['exp_stdev'] = 0

    return df


def predict_xgb(path):
    df = pd.read_csv(path, sep = '\t')
    df.columns = ['compound', 'pred_mean', 'pred_stdev']
    return df


def get_consensus(path1, path2):
    df1 = predict_dnn(path1)
    df2 = predict_dnn(path2)

    df = df1.merge(df2, on=['compound', 'exp_mean', 'exp_stdev'], how='right')
    df['new_pred_mean'] = (df['pred_mean_x'] * 5 + df['pred_mean_y'] * 5) / 10
    df['new_stdev_mean'] = (df['pred_stdev_x'] + df['pred_stdev_y']) / 2

    # preparing proper columns to be used for plot_validation function
    df = df[['compound', 'exp_mean', 'exp_stdev', 'new_pred_mean', 'new_stdev_mean']]
    df.columns = ['compound', 'exp_mean', 'exp_stdev', 'pred_mean', 'pred_stdev']

    return df


def get_training_df(dataset):
    df1 = pd.read_csv(dataset+'.act', sep = '\t', header = None)
    df1.columns = ['chembl', 'pKi']
    df2 = pd.read_csv(dataset+'.smi', sep = '\t', header = None)
    df2.columns = ['smiles', 'chembl']
    df_training = df1.merge(df2, on = 'chembl')
    return df_training


def remove_similar_amy(df_training, df_amy):
    '''Compares the similarity between the two df. Returns the df containing amy compounds with no similar compounds'''

    # Fingerprint
    fingerprints_df_training = []
    fingerprints_df_amy = []

    for i in range(len(df_training)):
        mol_in = Chem.MolFromSmiles(df_training['smiles'][i])
        fp = Chem.GetMorganFingerprintAsBitVect(mol_in, 2, nBits=2048)  # set nbits equal to what you will use in model?
        fingerprints_df_training.append(fp)

    for j in range(len(df_amy)):
        mol_in = Chem.MolFromSmiles(df_amy['smiles'][j])
        fp = Chem.GetMorganFingerprintAsBitVect(mol_in, 2, nBits=2048)  # set nbits equal to what you will use in model?
        fingerprints_df_amy.append(fp)

    # # Overlap (middle of Venn Diagram). Get similarity comparison
    ls_drop_amy = []
    for i in range(len(df_amy)):
        for j in range(len(fingerprints_df_training)):
            if DataStructs.TanimotoSimilarity(fingerprints_df_training[j], fingerprints_df_amy[i]) > 0.999:
                ls_drop_amy.append(i)
    df = df_amy.drop(ls_drop_amy)

    return df


def get_smiles_in_amy(df, dataset):
    df_amy = pd.read_csv(dataset, sep = '\t', header = None)
    df_amy.columns = ['smiles', 'compound']
    df_amy = df.merge(df_amy, on = 'compound')
    return df_amy


# %matplotlib inline
def summarize_model(datadir, chembl_version, method, model_name):
    """
    Summarizing model information from individual target predictions
    """
    modeldir = f'{datadir}/models_C{chembl_version}_200'
    data = pd.DataFrame(columns=['R2', 'Rp', 'Rs', 'RMSE'])
    n_models = 10
    for i in range(0, n_models):
        path = modeldir+f'/{model_name}/model_{i}/pred_reg_{method}_0.00/reg_{method}_0.00'
        df = pd.read_csv(path, sep = '\t')
        df.columns = ['compound', 'pred_mean', 'pred_stdev', 'exp_mean']
        df['exp_stdev'] = 0
        plot_validation(df)
        R2, Rp, Rs, MSE, RMSE = stats_df(df)
        data.loc[i] = [R2, Rp, Rs, RMSE]
    stat = {'ChEMBL version': [chembl_version],
            "method": [method],
            'model': [model_name],
            "n_models": [n_models],
            'Rp(mean)': [mean(data["Rp"])],
            'Rs(mean)': [mean(data["Rs"])],
            'R2(mean)': [mean(data["R2"])],
            'RMSE(mean)': [mean(data["RMSE"])],
            'Rp(stdev)': [np.std(data["Rp"])],
            'Rs(stdev)': [np.std(data["Rs"])],
            'R2(stdev)': [np.std(data["R2"])],
            'RMSE(stdev)': [np.std(data["RMSE"])],
            'Rp(max)': [max(data["Rp"])],
            'Rs(max)': [max(data["Rs"])],
            'R2(max)': [max(data["R2"])],
            'RMSE(min)': [min(data["RMSE"])]}
    df = pd.DataFrame(stat)
    df = df.round(2)
    return df


def summarize_models(datadir, method, chembl_version, output_dir):
    """
    Combining all four individual target information into one df
    """
    d2 = summarize_model(datadir, chembl_version, method, "models_D2")
    d3 = summarize_model(datadir, chembl_version, method, "models_D3")
    d2_overlap = summarize_model(datadir, chembl_version, method, "models_D2_overlap")
    d3_overlap = summarize_model(datadir, chembl_version, method, "models_D3_overlap")
    df = pd.concat([d2, d3, d2_overlap, d3_overlap], axis=0)
    df.to_csv(output_dir + f"/C{chembl_version}.csv")
    return df


def summarize_two_model(datadir, chembl_version, method, n_models, model1, model2):
    """
    Subtracting two models' predictions against each other to determine selectivity
    """
    modeldir = f'{datadir}/models_C{chembl_version}_200'
    data = pd.DataFrame(columns=['R2', 'Rp', 'Rs', 'RMSE'])
    n_models = 10
    for i in range(0, n_models):
        path = modeldir + f'/{model1}/model_{i}/pred_reg_{method}_0.00/reg_{method}_0.00'
        df1 = pd.read_csv(path, sep='\t')
        df1.columns = ['compound', 'pred_mean', 'pred_stdev', 'exp_mean']
        df1['exp_stdev'] = 0

        path = modeldir + f'/{model2}/model_{i}/pred_reg_{method}_0.00/reg_{method}_0.00'
        df2 = pd.read_csv(path, sep='\t')
        df2.columns = ['compound', 'pred_mean', 'pred_stdev', 'exp_mean']
        df2['exp_stdev'] = 0

        df = pd.DataFrame(columns=['compound', 'pred_mean', 'pred_stdev', 'exp_mean', 'exp_stdev'])
        df['compound'] = df1['compound']
        df['pred_mean'] = df1['pred_mean'] - df2['pred_mean']
        df['pred_stdev'] = df1['pred_stdev'] - df2['pred_stdev']
        df['exp_mean'] = df1['exp_mean'] - df2['exp_mean']
        df['exp_stdev'] = df1['exp_stdev'] - df2['exp_stdev']

        plot_validation(df)
        R2, Rp, Rs, MSE, RMSE = stats_df(df)
        data.loc[i] = [R2, Rp, Rs, RMSE]

    stat = {'ChEMBL version': [chembl_version],
            "method": [method],
            'model': [f"{model1}_{model2}"],
            "n_models": [n_models],
            'Rp(mean)': [mean(data["Rp"])],
            'Rs(mean)': [mean(data["Rs"])],
            'R2(mean)': [mean(data["R2"])],
            'RMSE(mean)': [mean(data["RMSE"])],
            'Rp(stdev)': [np.std(data["Rp"])],
            'Rs(stdev)': [np.std(data["Rs"])],
            'R2(stdev)': [np.std(data["R2"])],
            'RMSE(stdev)': [np.std(data["RMSE"])],
            'Rp(max)': [max(data["Rp"])],
            'Rs(max)': [max(data["Rs"])],
            'R2(max)': [max(data["R2"])],
            'RMSE(min)': [min(data["RMSE"])]}
    df = pd.DataFrame(stat)
    df = df.round(2)
    return df


def summarize_selectivities(datadir, method, chembl_version, n_models, output_dir):
    """ Combines all 3 selectivity approaches together
    1. D2 Ki predictions - D3 Ki predictions
    2. D2 overlap Ki predictions - D3 overlap Ki predictions
    3. building selectivity model directly from D2 pKi-D3 pKi dataset
    """
    df_diff = summarize_two_model(datadir, chembl_version, method, n_models, "models_D2", "models_D3")
    df_diff_overlap = summarize_two_model(datadir, chembl_version, method, n_models, "models_D2_overlap",
                                          "models_D3_overlap")
    d2d3_diff = summarize_model(datadir, chembl_version, method, "models__ratio_D2D3")
    df = pd.concat([df_diff, df_diff_overlap, d2d3_diff], axis=0)
    df.to_csv(output_dir + f"/C{chembl_version}_diff.csv")
    return df



def summarize_all_200(model_dir, chembl_version, n_models, ls_models, output_dir):
    df1 = summarize_model(model_dir, chembl_version, n_models, ls_models[0])
    df2 = summarize_model(model_dir, chembl_version, n_models, ls_models[1])
    df1_overlap = summarize_model(model_dir, chembl_version, n_models, ls_models[2])
    df2_overlap = summarize_model(model_dir, chembl_version, n_models, ls_models[3])
    df_final = pd.concat([df1, df2, df1_overlap, df2_overlap], axis=1)
    df_final.to_csv(output_dir + f"/{chembl_version}_200.csv")

    return df_final


def summarize_all_200_selectivity(model_dir, chembl_version, n_models, ls_models, output_dir):
    df_diff = summarize_two_models(model_dir, chembl_version, n_models, ls_models[0], ls_models[1])
    df_diff_overlap = summarize_two_models(model_dir, chembl_version, n_models, ls_models[2], ls_models[3])
    df_diff_ratio = summarize_model(model_dir, chembl_version, n_models, ls_models[4])
    df_final = pd.concat([df_diff, df_diff_overlap, df_diff_ratio], axis=1)
    df_final.to_csv(output_dir+f"/{chembl_version}_200_selectivity.csv")
    return df_final


def write_script_do_one_200(path, mode, dataset, i):
    with open(path + f'/do_{mode}.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('set -x\n')
        f.write('module load python/anaconda3-2020.02-py3.7.6\n')
        f.write('\n')
        f.write(dataset)
        f.write("COREPATH=$HOME/repositories/ai-x/core\n")
        f.write(f"i={i}\n\n")
        f.write("echo $DATASET\n")
        f.write("echo $COREPATH\n\n")
        f.write("echo $i\n")
        if mode == 'buildmodel':
            f.write(f"#python $COREPATH/run_buildmodel.py -s {mode} -m reg -x xgb -t 0.15 -r 1 -n 1 -i ${{DATASET}}$i \n")
            f.write(f"#python $COREPATH/run_buildmodel.py -s {mode} -m reg -x rf -t 0.15 -r 1 -n 1 -i ${{DATASET}}$i \n")
            f.write(f"python $COREPATH/run_buildmodel.py -s {mode} -m reg -x xgb -t 0 -r 1 -n 1 -i ${{DATASET}}$i \n")
            f.write(f"python $COREPATH/run_buildmodel.py -s {mode} -m reg -x rf -t 0 -r 1 -n 1 -i ${{DATASET}}$i \n")

            f.write(f"#python $COREPATH/run_buildmodel.py -s {mode} -m class -x xgb -t 0.15 -r 1 -n 1 -i ${{DATASET}}$i \n")
            f.write(f"#python $COREPATH/run_buildmodel.py -s {mode} -m class -x rf -t 0.15 -r 1 -n 1 -i ${{DATASET}}$i \n")
            f.write(f"#python $COREPATH/run_buildmodel.py -s {mode} -m class -x xgb -t 0 -r 1 -n 1 -i ${{DATASET}}$i \n")
            f.write(f"#python $COREPATH/run_buildmodel.py -s {mode} -m class -x rf -t 0 -r 1 -n 1 -i ${{DATASET}}$i \n")
        elif mode == 'prediction':
            f.write(f"python $COREPATH/run_buildmodel.py -s {mode} -m reg -x xgb -t 0 -r 1 -n 1 -d ${{DATASET}}$i \n")
            f.write(f"python $COREPATH/run_buildmodel.py -s {mode} -m reg -x rf -t 0 -r 1 -n 1 -d ${{DATASET}}$i \n")
            f.write(f"#python $COREPATH/run_buildmodel.py -s {mode} -m class -x xgb -t 0 -r 1 -n 1 -d ${{DATASET}}$i \n")
            f.write(f"#python $COREPATH/run_buildmodel.py -s {mode} -m class -x rf -t 0 -r 1 -n 1 -d ${{DATASET}}$i \n")


def write_script_do_all_200(path, mode):
    with open(path + f'/all_{mode}s.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write("for i in {0..9}\n")
        f.write("do\n")
        f.write("echo $i\n")
        f.write("cd model_$i\n")
        f.write(f"bash do_{mode}.sh\n")
        f.write("cd ..\n")
        f.write("done\n")


def write_script_do_all_predictions(model_dir, ls_models):
    with open(f'{model_dir}chembl_predictions.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        for model in ls_models:
            f.write(f"cd {model}\n")
            f.write("bash all_predictions.sh\n")
            f.write("cd ..\n")


def get_C33_dataset(datadir, dataset, filename):
    print(dataset)
    path1 = datadir+f"C31_0/{dataset}/pubdata0"
    path2 = datadir+f"C33_0/{dataset}/pubdata0"
    df_C31 = get_df(path1)
    df_C33 = get_df(path2)
    df_val = remove_similar_amy(df_C31, df_C33) # adds a column of smi
    df = df_val

    output_dir = '../selectivity_datasets/new_dataset_selectivity_C33_only/'
    df_act = df[['compound', 'act']]
    df_smi = df[['smiles', 'compound']]
    check_output_dir(output_dir, keep_old = False)
    df_act.to_csv(output_dir + filename+'.act', sep='\t', index=False, header = False)
    df_smi.to_csv(output_dir + filename+'.smi', sep='\t', index=False, header = False)

    return df


def get_ls_R_and_R2(path, method, ls_Rp, ls_Rs, ls_R2, ls_RMSE, ls_MSE):
    '''
    returns list of calculated Rs and R2s for all of models
    '''
    ls_filename = glob.glob(f'{path}/reg_{method}_0.15/stat_*')
    ls_lines = []
    for filename in ls_filename:
        with open(filename) as f:
            lines = f.readlines()
            ls_lines.append(lines)
    for line in ls_lines:
        line = line[0].replace('\n', '')
        line_split = line.split(' ')
        for i in range(len(line_split)):
            if line_split[i] == 'Rp':
                ls_Rp.append(float(line_split[i + 1]))
            if line_split[i] == 'Rs':
                ls_Rs.append(float(line_split[i + 1]))
            if line_split[i] == 'R2':
                ls_R2.append(float(line_split[i + 1]))
            if line_split[i] == 'RMSE':
                ls_RMSE.append(float(line_split[i + 1]))
            if line_split[i] == 'MSE':
                ls_MSE.append(float(line_split[i + 1]))
    return ls_Rp, ls_Rs, ls_R2, ls_RMSE, ls_MSE


def internal_benchmark(datadir, ls_chembls, ls_methods, ls_models, save2png):
    df = pd.DataFrame()
    for model in ls_models:
        for chembl in ls_chembls:
            for method in ls_methods:
                path = datadir + f"/models_{chembl}_200/{model}/model_*"
                ls_paths = glob.glob(path)
                ls_Rp, ls_Rs, ls_R2, ls_RMSE, ls_MSE = [], [], [], [], []
                for path in ls_paths:
                    ls_Rp, ls_Rs, ls_R2, ls_RMSE, ls_MSE = get_ls_R_and_R2(path, method, ls_Rp, ls_Rs, ls_R2, ls_RMSE,
                                                                           ls_MSE)
                if len(ls_Rp) > 0:
                    mean_Rp = round(mean(ls_Rp), 2)
                    mean_Rs = round(mean(ls_Rs), 2)
                    mean_R2 = round(mean(ls_R2), 2)
                    mean_RMSE = round(mean(ls_RMSE), 2)
                    stdev_Rp = round(stdev(ls_Rp), 2)
                    stdev_Rs = round(stdev(ls_Rs), 2)
                    stdev_R2 = round(stdev(ls_R2), 2)
                    stdev_RMSE = round(stdev(ls_RMSE), 2)
                    max_Rp = round(max(ls_Rp), 2)
                    max_Rs = round(max(ls_Rs), 2)
                    max_R2 = round(max(ls_R2), 2)
                    min_RMSE = round(min(ls_RMSE), 2)
                else:
                    mean_Rp = 0
                    mean_Rs = 0
                    mean_R2 = 0
                    mean_RMSE = 0
                    mean_MSE = 0
                    stdev_Rp = 0
                    stdev_Rs = 0
                    stdev_R2 = 0
                    stdev_RMSE = 0
                    stdev_MSE = 0
                    max_Rp = 0
                    max_Rs = 0
                    max_R2 = 0
                    min_RMSE = 0
                    min_MSE = 0
                stat = {'ChEMBL version': [chembl],
                        "method": [method],
                        'model': [model],
                        "n_models": [len(ls_paths)],
                        'Rp(mean)': [mean_Rp],
                        'Rs(mean)': [mean_Rs],
                        'R2(mean)': [mean_R2],
                        'RMSE(mean)': [mean_RMSE],
                        'Rp(stdev)': [stdev_Rp],
                        'Rs(stdev)': [stdev_Rs],
                        'R2(stdev)': [stdev_R2],
                        'RMSE(stdev)': [stdev_RMSE],
                        'Rp(max)': [max_Rp],
                        'Rs(max)': [max_Rs],
                        'R2(max)': [max_R2],
                        'RMSE(min)': [min_RMSE]}
                df_row = pd.DataFrame(stat)
                df = pd.concat([df, df_row], ignore_index=True, sort=False)
    df.to_csv(save2png)

    return df


def add_value_labels(ax, spacing=5):
    '''Please see https://stackoverflow.com/questions/59143306/add-label-values-to-bar-chart-and-line-chart-in-matplotlib
    '''
    space = spacing
    va = 'bottom'
    for i in ax.patches:
        y_value = i.get_height()
        x_value = i.get_x() + i.get_width() / 2

        label = "{:.2f}".format(y_value)
        ax.annotate(label, (x_value, y_value), xytext=(0, space),
                    textcoords="offset points", ha='center', va=va)

    # https://matplotlib.org/stable/gallery/color/named_colors.html


def bar_plot(df, model, metric, y, colors, save2png):
    ymin, ymax = y
    df = df[df["model"] == model]
    figwidth = 2
    figheight = 4
    col_mean = metric + "(mean)"
    col_stdev = metric + "(stdev)"
    yerr = df[col_stdev].to_numpy().T
    ax = df.plot.bar(x='method', y=col_mean, yerr=yerr,color=colors, figsize=(figwidth, figheight), width=0.85);
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='dotted', linewidth=0)
    ax.legend().set_visible(False)
    if metric == "Rp":
        ax.tick_params(top=False, bottom=False, left=True, right=False, labelleft=True, labelbottom=True, pad=0)
    elif metric == "Rs":
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=True, pad=0)
    elif metric == "R2":
        ax.tick_params(top=False, bottom=False, left=False, right=True, labelleft=True, labelbottom=True, pad=0)
    plt.axhline(y=0, color='black', linestyle='-')
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_ticks(np.arange(ymin, ymax, 0.2))

    if save2png:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('', fontsize=10)
        ax.set_ylabel('', fontsize=10)
        plt.savefig(save2png, dpi=600, transparent=True, bbox_inches="tight")

    else:
        add_value_labels(ax)
        ax.set_title(f"{model} ({metric})")

    return df


#def internal_selectivity_calculation():
# n_model = 10

# ls_R2, ls_Rp, ls_Rs, ls_MSE, ls_RMSE = [], [], [], [], []
# df = pd.DataFrame()

# for chembl in ['C31', 'C32', 'C33']:
#     for i in range(n_model):
#         datadir = f"{HOME}/repositories/ai-DR/selectivity_D2D3/selectivity_models/models_{chembl}_200/"
#         path1 = datadir+f"models_DR2_overlap/model_{i}/reg_xgb_0.15/pred_864.csv"
#         path2 = datadir+f"models_DR3_overlap/model_{i}/reg_xgb_0.15/pred_864.csv"

#         df1 = pd.read_csv(path1, sep = '\t', header = None)
#         df1.columns = ['compound', 'pred']
#         df2 = pd.read_csv(path2, sep = '\t', header = None)
#         df2.columns = ['compound', 'pred']

#         dataset = f"{HOME}/repositories/ai-DR/selectivity_D2D3/selectivity_datasets/{chembl}_200/dataset_overlap_D2_antagonist_Ki/pubdata{i}.act"
#         df_exp = pd.read_csv(dataset, sep = '\t', header = None)
#         df_exp.columns = ['compound', 'exp']
#         df1 = df1.merge(df_exp, on = 'compound')

#         dataset = f"{HOME}/repositories/ai-DR/selectivity_D2D3/selectivity_datasets/{chembl}_200/dataset_overlap_D3_antagonist_Ki/pubdata{i}.act"
#         df_exp = pd.read_csv(dataset, sep = '\t', header = None)
#         df_exp.columns = ['compound', 'exp']
#         df2 = df2.merge(df_exp, on = 'compound')

#         df_diff = pd.DataFrame()
#         df_diff = df1.merge(df2, on = 'compound')
#         df_diff['pred_diff'] = df_diff['pred_x'] - df_diff['pred_y']
#         df_diff['exp_diff'] = df_diff['exp_x'] - df_diff['exp_y']
#         y_pred = df_diff['pred_diff']
#         y_true = df_diff['exp_diff']
#         R2, Rp, Rs, MSE, RMSE = stats(y_true, y_pred, verbose = False)
#         ls_R2.append(R2)
#         ls_Rp.append(Rp)
#         ls_Rs.append(Rs)
#         ls_MSE.append(MSE)
#         ls_RMSE.append(RMSE)

#     mean_Rp = round(mean2(ls_Rp), 2)
#     mean_Rs = round(mean2(ls_Rs), 2)
#     mean_R2 = round(mean2(ls_R2), 2)
#     mean_RMSE = round(mean2(ls_RMSE), 2)
#     mean_MSE = round(mean2(ls_MSE), 2)
#     model = 'D2 overlap - D3 overlap'
#     stat = {'ChEMBL version': [chembl],
#         'model': [model],
#         'Rp(mean)': [mean_Rp],
#         'Rs(mean)': [mean_Rs],
#         'R2(mean)': [mean_R2],
#         'RMSE(mean)': [mean_RMSE],
#         'MSE(mean)': [mean_MSE]}
#     df_row = pd.DataFrame(stat)
#     df = pd.concat([df, df_row], ignore_index=True, sort=False)
#     df
