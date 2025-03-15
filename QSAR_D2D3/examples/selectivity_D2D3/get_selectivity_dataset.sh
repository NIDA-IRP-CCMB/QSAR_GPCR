set -x
module load python/anaconda3-2020.02-py3.7.6-ai

COREPATH=$HOME/repositories/ai-x/core/selectivity
python $COREPATH/get_selectivity_dataset.py C33 200 10 DR
