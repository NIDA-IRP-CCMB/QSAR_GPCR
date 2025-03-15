#!/bin/bash
set -x

module load python/anaconda3-2020.02-py3.7.6-ai 

SELECTIVITY_DATADIR="new_selectivity_datasets"
COREPATH=$HOME/repositories/ai-x/core/selectivity

python $COREPATH/prep_selectivity_models.py C33 200 DR ${SELECTIVITY_DATADIR}

