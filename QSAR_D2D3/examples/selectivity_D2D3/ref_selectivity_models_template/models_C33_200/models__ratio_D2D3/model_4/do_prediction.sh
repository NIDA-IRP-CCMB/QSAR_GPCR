#!/bin/bash
set -x
module load python/anaconda3-2020.02-py3.7.6-ai

DATASET=new_selectivity_datasets/C33_200/dataset__ratio_D2_antagonist_Ki_D3_antagonist_Ki/val
COREPATH=$HOME/repositories/ai-x/core
i=4

echo $DATASET
echo $COREPATH

echo $i
python $COREPATH/run_buildmodel.py -s prediction -m reg -x xgb -t 0 -r 1 -n 1 -d ${DATASET}$i 
