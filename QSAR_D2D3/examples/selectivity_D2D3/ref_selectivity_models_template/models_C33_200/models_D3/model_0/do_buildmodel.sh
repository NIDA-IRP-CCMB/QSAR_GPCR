#!/bin/bash
set -x
module load python/anaconda3-2020.02-py3.7.6-ai

DATASET=new_selectivity_datasets/C33_200/dataset_D3_antagonist_Ki/pubdata
COREPATH=$HOME/repositories/ai-x/core
i=0

echo $DATASET
echo $COREPATH

echo $i
python $COREPATH/run_buildmodel.py -s buildmodel -m reg -x xgb -t 0 -r 1 -n 1 -i ${DATASET}$i 
python $COREPATH/run_buildmodel.py -s buildmodel -m reg -x xgb -t 0.15 -r 1 -n 1 -i ${DATASET}$i 
