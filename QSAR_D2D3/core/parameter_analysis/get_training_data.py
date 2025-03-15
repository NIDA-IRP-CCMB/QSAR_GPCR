#!/home/wons2/Shilab/deep/anaconda3-2020.02-py3.7.6/bin/python
import sys, os
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
conf_dir = core_dir+"/conf"
sys.path.insert(0, core_dir)
sys.path.insert(0, conf_dir)
DR_dir = home+'/repositories/ai-DR/core'
sys.path.insert(0, DR_dir)
import os.path
from deeplearning import *
import pickle


# initial values
split = sys.argv[1]
# assert target == 'DR2' or target == 'DR3'
stage = 'buildmodel'
mode = 'reg'
method = 'dnn'
tp = 0.15
num_splits = 10
n_rand_stat = 10


# prepare training data
pubdata_dir = '../dataset/pubdata'
train_dataset_prefix = pubdata_dir + "_class" if mode is 'class' else pubdata_dir
output_dir = get_output_dir(mode, method, tp)  # reg_dnn_0.15
check_misc(get_dir(train_dataset_prefix))  # checking and adding appropriate directories
check_output_dir(output_dir, keep_old=False)
rand_splits = gen_random_splits(control_seed=2020, num_splits=num_splits)
rand_states = gen_random_splits(control_seed=501, num_splits=n_rand_stat)
rand_state = rand_states[0]
random_split = rand_splits[int(split)]
train_names, test_names, train_descs, train_acts, test_descs, test_acts, topo_names, phore_names\
                        = get_training_dataset(mode, tp, stage, train_dataset_prefix, output_dir, output_ext, rand_state,
                                               random_split, all_descs=True, extra_features = True, remove_ipc = True)
train_descs_norm, test_descs_norm = convert_to_zscore2(train_descs, test_descs)
feature_names = topo_names + phore_names

dict_data = {'train_descs_norm': train_descs_norm, 'test_descs_norm': test_descs_norm,
             'train_descs': train_descs, 'test_descs': test_descs, "test_names": test_names,
             'train_acts': train_acts, 'test_acts': test_acts, 'train_names': train_names,
             'rand_state': rand_state, 'random_split': random_split, "topo_names": topo_names,
             "phore_names": phore_names, "feature_names": feature_names
             }

with open('../dict_data.pickle', 'wb') as handle:
    pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)