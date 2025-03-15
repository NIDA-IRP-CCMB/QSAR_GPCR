## Parameter Analysis

	The following files should be here:

	parameter_list.py		<------		Generates list of parameter combinations into a text file.

	parameters_list.txt		<------		Generated from the parameter_list.py file above

	parameter_modelfit.py		<------		Runs the DNN models using model.fit, given the parameter set in .txt file. Deletes the set in the txt file upon model completion, then outputs results to DR2_results.txt and DR3_results.txt

	parameter_analysis.ipynb	<-----		Notebook to do parameter analysis. Scatterplot, histogram, and heatmap of the parameter results.

	prep_swarm.py			<-----		Prepares the swarm files, do_DR2_params.swarm and do_DR3_params.swarm to run parameter_modelfit.py. 	

	do_params.swarm 		<-----		Swarm files that uses parameter_modelfit.py that sends job scripts to Biowulf to run models.

## Instruction

**1. Soft link the dataset to prepare model building.**
	
		$ ln -sf<dataset directory location (the directory of pubdata)> dataset
		
		
**2. Prepare folder name, e.g. DR2_0_1000**

		$ mkdir DR2_0_1000_split0
		
		$ cd DR2_0_1000_split0
		
		
**3. Pre-generate the parameter combinations ahead of time.**
		
		$ python ~/repositories/ai-x/core/parameter_analysis/parameter_list.py <starting index> <ending index>
	
For example, if you want to generate the first 1000 random combinations, 
 
		$ python ~/repositories/ai-x/core/parameter_analysis/parameter_list.py 0 1000 (0 to 999, 1000 is not included)
		
If you want to generate the next 1000 random combinations, you can use 1000 and 2000.
This generates 'parameters_list.txt,' which contains list of parameter combinations to build models from. This is done by comparing the list to results_list.txt seen later. 


**4. Prepare the training and testing descriptor data into a pickle file.**

```
$ /data/Shilab/apps/anaconda3-2020.02-py3.7.6/bin/python ~/repositories/ai-x/core/parameter_analysis/get_training_data.py <split #>
```		
where split # is 0 to 10 (index of rand_split). 


**5. Generate swarm files.**
	
		$ /data/Shilab/apps/anaconda3-2020.02-py3.7.6/bin/python ~/repositories/ai-x/core/parameter_analysis/prep_swarm.py <num_of_jobs>
		
where num_of_jobs is the number of jobs you want to submit to biowulf server and run models for, e.g. 1000. 

		If you missed any models or failed models and need to rerun, you can rerun this swarm generating file and it will scan the results_list.txt for an updated do_params.swarm file to see which new models to run.

**6. Run the DNN models individually using model.fit function. You MUST have "parameters_list.txt" in the corresponding folder to run these commands.**
	
		$ swarm --module CUDA/11.2.2,cuDNN/8.1.0.77/CUDA-11.2.2 -g 40 --logdir uptime_output --file do_params.swarm  # note that 40 refers to GB used. This is important for job to not crash via memory issue
	
**7. Do analysis of the results using parameter_analysis.ipynb.**

## Notes

If some of the models do not finish running or if some models fail, you can update the parameters_list.txt file. This file is compared with the results_list.txt to see which combinations still need to be done. 


## Analysis 

Analysis can be found in notebooks/parameter_analysis.ipynb. 

