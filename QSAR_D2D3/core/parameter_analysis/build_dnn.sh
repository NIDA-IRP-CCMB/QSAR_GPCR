#!/bin/bash

target=$1
start=$2
end=$3
split=$4
interval="$((end-start))"


set -x
dir_name=${target}_${start}_${end}_split${split}
module load python

mkdir ${dir_name}
cd ${dir_name}

python ../scripts/parameter_list.py ${start} ${end}
<<<<<<< HEAD
~/Shilab/deep/anaconda3-2020.02-py3.7.6/bin/python ../scripts/get_training_data.py ${target} ${split}
=======
/data/Shilab/apps/anaconda3-2020.02-py3.7.6/bin/python ../scripts/get_training_data.py ${target} ${split}
>>>>>>> 5a8257cedd1301bd999c5091d5837dc3bfb40aee


python ../scripts/prep_swarm.py ${interval}
swarm --module CUDA/11.2.2,cuDNN/8.1.0.77/CUDA-11.2.2 --logdir uptime_output --file do_params.swarm 
