# Selectivity

There are key related files in core/selectivity/

		selectivity.py 			----> all parent functions

		get_selectivity_dataset.py	----> functions used to generate selectivity datasets


In this folder, there are the following files:

	Relevant files:

		get_selectivity_dataset.sh 	----> bash script to generate selectivity dataset 
  
 		prep_selectivity_models.py	----> preparing scripts to build and predict XGB 

 		ref_* 				----> References that are expected outputs from the *.sh files
   
		outputs   			---->  how the model hierarchy should look like. 
  
  Outputs has the output files from running DNN models for 85% data using the D3-selective dataset. Includes all outputs for the first model except the model itself (too large to be included in Git)




## Instructions

	Please check the py file at each steps for further instructions.

	1. Generate dataset first.

		$ bash get_selectivity_datasets.sh
	
	2. Prepare the scripts that can be used to build XGB models. This script can also be used to predict against the 200 validation datapoints

		$ bash prep_selectivity_models.sh


## Data set information

This would generate 10 datasets (10 different rand_states) for both training and validation datasets.
There are four types of datasets:

1) D2 only (original chembl dataset), 200 validation extracted
        --> build D2 models

2) D3 only (original chembl dataset), 200 validation extracted
        --> build D3 models

	----> compare 1 and 2 models from (1) and (2)


3) Overlapping D2 and D3 - 1758, 200 validation extracted --> two datasets - D2 and D3
        --> build D2 models
        --> build D3 models
--> compare D2 and D3 models from (3) alone

4) Same as (3), but ratio between D2 and D3 - one dataset
