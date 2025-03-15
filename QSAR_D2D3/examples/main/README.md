# Utility

This notebook provides utility scripts for carrying out the large tasks in this project. For some of the corresponding notebooks, you will notice "n", which corresponds to the number of data points desired in the validation dataset. For this project, set n = 200 or 0 to reserve data for validation dataset. If n = 0, it will still produce 10 (or desired number of datasets) for training dataset but they would all be the same. Same thing for validation dataset.


## Notebooks

### Generating Regression and Classification Datasets & Reserving Validation Datasets

	training_dataset.ipynb 		  <--- creating training dataset using TSV files. Using unique set of filters that is not used in other proteins

	get_selectivity_datasets.ipynb    <--- Generating selectivity data set (which includes the original training dataset). This also reserves 200 data points if desired. The final model production uses this dataset, not the one created above. If no data reservation is desired, use n = 0

	get_classification_dataset.ipynb  <--- creating classification dataset. Can set n = 200 or 0 


### Running XGBoost, RF, DNN, and SHAP & Monitoring Job Progress

	prep_XGB_RF_biowulf.ipynb	  <--- script to generate scripts for running xgb and rf in biowulf, in parallel. set stage to "same_buildmodel" or "prediction"

	prep_DNN.ipynb 			  <--- script to generate scripts to run dnn jobs

	prep_SHAP.ipynb			  <--- script to generate scripts to run shap for whichever methods needed. Not necessary since the job scripts above will generate them, so this is only used to check DNN shaps before DNN models are finished. 


