import os, sys
import pandas as pd

# initial values
num_of_jobs = int(sys.argv[1])
gpu_card = 'a100'
days = 1

filename_results = 'results_list.txt'
if os.path.isfile(filename_results):
    df_results = pd.read_csv(filename_results, header = None, sep = ' ')
    indexes = df_results[0].values.tolist()
    indexes = sorted(set(indexes))
    remaining_indexes = list(range(num_of_jobs))
    for i in indexes:
        remaining_indexes.remove(i)
else:
    remaining_indexes = list(range(num_of_jobs))
# prep swarm files
f = open(f"do_params.swarm", "w")
f.write('#!/bin/bash\n')
f.write(f'#SWARM --job-name=do_params\n')
f.write('#SWARM --partition=gpu\n')
f.write(f'#SWARM --time={days}-00:00:00\n')
f.write(f'#SWARM --gres=gpu:{gpu_card}:1\n\n')
for i in remaining_indexes:
    f.write(f'/data/Shilab/apps/anaconda3-2020.02-py3.7.6/bin/python ~/repositories/ai-x/core/parameter_analysis/parameter_modelfit.py {i}\n')
f.close()

