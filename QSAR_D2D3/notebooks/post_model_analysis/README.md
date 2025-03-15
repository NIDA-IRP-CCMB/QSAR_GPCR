# Post-model analysis

This section includes SHAP analysis, analysis of the representative compounds (includes compounds used for docking, and clusterization analysis), and Ph2D and non-Ph2D feature analysis. 

## SHAP
	
	shap_features_individual.ipynb --> generating xgb, rf, and dnn shap values
	
	shap_features_consensus.ipynb --> average of xgb, rf, and dnn shap values (consensus)


## Post-SHAP analysis

	Ph2D_distributions.ipynb --> generates pickle training datasets then uses those to generate histograms for Ph2D features
	
	Non-Ph2D_distributions.ipynb --> generates pickle training datasets then uses those to generate histograms for non-Ph2D features
	
	pickle files are named train_descs_D2.pkl, train_descs_D3.pkl, train_descs_D2D3.pkl


## Visualizing Compounds
	
	Pharm2D_Visualization_DR.ipynb --> visualizes the Ph2D features in the compound. If Ph2D feature is missing, outputs the compound without the ph2d feature.

