python ~/repositories/ai-x/core/parameter_analysis/parameter_list.py 3000 4000
/data/Shilab/apps/anaconda3-2020.02-py3.7.6/bin/python ~/repositories/ai-x/core/parameter_analysis/prep_swarm.py 1000
swarm --module CUDA/11.2.2,cuDNN/8.1.0.77/CUDA-11.2.2 -g 40 --logdir uptime_output --file do_params.swarm 
