"""
This code prepares scripts to build XGB models and predict on the selectivity issue.
Example of a shell script you can run:
    #!/bin/bash
    set -x
    module load python/anaconda3-2020.02-py3.7.6-ai

    DATASET_DIR="selectivity_datasets"
    COREPATH=$HOME/repositories/ai-x/core
    python $COREPATH/prep_selectivity_models.py C33 200 D2 D3 antagonist Ki ${DATASET_DIR}

Please modify "DATASET_DIR" to where your selectivity dataset is located.

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

####################################### VARIABLES ###################################################
chembl_version = sys.argv[1]
n = int(sys.argv[2])
target = sys.argv[3]
selectivity_datadir = sys.argv[4]
####################################### VARIABLES ###################################################

model_dir = f"selectivity_models/models_{chembl_version}_{n}/"
n_models = 10

target1, target2, assaydefinition1, assaydefinition2, \
        measurement1, measurement2, target1_source, target2_source = get_target_information(target)
dict_dataset = get_dataset_dict(target1, assaydefinition1, measurement1, target2, assaydefinition2, measurement2)
ls_models = list(dict_dataset.keys())

if n == 200:
    for model in ls_models:
        for i in range(n_models):
            path = f"{model_dir}{model}/model_{i}"
            check_output_dir(path, keep_old = False)
            datadir = f"DATASET={selectivity_datadir}/{chembl_version}_{n}/{dict_dataset[model]}"
            train_dataset = f"{datadir}/pubdata\n"
            val_dataset = f"{datadir}/val\n"
            # script to do each individual model
            write_script_do_one_200(path, "buildmodel", train_dataset, i)
            write_script_do_one_200(path, "prediction", val_dataset, i)

        # script to do all at once (per each set of models) 
        path = f"{model_dir}{model}"
        write_script_do_all_200(path, "buildmodel")
        write_script_do_all_200(path, "prediction")

        # script to predict every single models
        write_script_do_all_predictions(model_dir, ls_models)
