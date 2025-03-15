"""
This code prepares selectivity datasets. It generates six total datasets:
    1/2) protein 1 and protein 2 directly from ChEMBL database
    3/4) finds overlapping compounds between protein 1 and protein 2, then displays those pKi values from each
    5/6) protein 1 pKi - protein 2 pKi is used as data. Both regression and classification datasets are generated.

This code assumes that your primary datasources are formatted as and located in
    {datadir}/{chembl_version}/dataset_{target1}_{assaydefinition}_{measurement}/pubdata
Please add in your specific target information in def get_target_information(target) in selectivity.py.

Using this shell script, the dataset should be outputted at your current label and named as "new_selectivity_datasets".
Please rename this parent directory name to fit your needs. However, do not rename the folders inside this.

Written by SungJoon (Jason) Won
"""

import os, sys
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
sys.path.insert(0, core_dir)
selectivity_dir = core_dir + "/selectivity"
sys.path.insert(0, selectivity_dir)

from selectivity import *

# command line arguments
chembl_version = sys.argv[1]
n = int(sys.argv[2]) # 200
num_of_val = int(sys.argv[3]) # 10
target = sys.argv[4] # e.g. DR, SERT

target1, target2, assaydefinition1, assaydefinition2, \
        measurement1, measurement2, target1_source, target2_source = get_target_information(target)

# source data
path1 = target1_source
path2 = target2_source
dict_dataset = get_dataset_dict(target1, assaydefinition1, measurement1, target2, assaydefinition2, measurement2)
ls_datasets = list(dict_dataset.values())

# Prepare output directories
datadir = f"new_selectivity_datasets/{chembl_version}_{n}/"
check_output_dir(datadir, keep_old = False)
for dataset in ls_datasets:
    check_output_dir(datadir + dataset, keep_old = False)
    #touch_files(datadir + dataset)

target1 = get_df(path1)  # get df with columns containing chembl, pki, and smile string
target2 = get_df(path2)  # get df with columns containing chembl, pki, and smile string
target1 = check_similarity_within_df(target1)  # checks if there are any similar compounds within its own df
target2 = check_similarity_within_df(target2)  # checks if there are any similar compounds within its own df
mol_dup_pairs = get_mol_dup_pairs(target1, target2)  # a paired set, referring to similar indexes between d2 and d3
df_overlap = get_overlap(target1, target2, mol_dup_pairs)  # combines protein1 and protein2 (horizontally), only returns similar compounds

for i in range(num_of_val):
    np.random.seed(i)
    suffix = str(i)
    print(suffix + " validation index #")

    # dataset 1 and 2) split from original chembl dataset
    ls_target1_indexes, ls_target2_indexes, ls_overlap_indexes = get_similar_indexes(mol_dup_pairs, n)  # get n number of similar indexes
    target1_training, target1_validation = training_validation_datasplit(target1, ls_target1_indexes)  # split into training and validation
    target2_training, target2_validation = training_validation_datasplit(target2, ls_target2_indexes)  # split into training and validation
    # save datasets
    save_one_dataset(f"{datadir}/{ls_datasets[0]}/", target1_training, target1_validation, suffix)  # first dataset
    save_one_dataset(f"{datadir}/{ls_datasets[1]}/", target2_training, target2_validation, suffix)  # second dataset
    print(f"(2) Protein 1 & 2 datasets created")

    # dataset 3 and 4) two overlapping datasets.
    # Find overlapping compounds between target 1 and target 2, concatenate them, then save them separately.
    df_training, df_validation = training_validation_datasplit(df_overlap, ls_overlap_indexes)
    print(df_training.columns)
    print(df_validation.columns)
    df_train_overlap1, df_val_overlap1, df_train_overlap2, df_val_overlap2 = split_overlapped_df(df_training, df_validation)

    # save datasets
    save_one_dataset(f"{datadir}/{ls_datasets[2]}/", df_train_overlap1, df_val_overlap1, suffix)  # first dataset
    save_one_dataset(f"{datadir}/{ls_datasets[3]}/", df_train_overlap2, df_val_overlap2, suffix)  # second dataset
    print(f"(2) Protein 1 & 2 Overlapping datasets created")

    # dataset 5) ratios - reg and class
    save_ratios(f"{datadir}/{ls_datasets[4]}/", df_training, df_validation, suffix)
    print("(2) Ratio of protein 1 & 2 - regression and classification datasets created")
