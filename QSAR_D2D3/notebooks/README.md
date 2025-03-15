## Notebooks

### Folder Hierarchy
        notebooks
            ├── D2D3_external                 Prediction analysis work on Dr. Newman's validation set (from JCCao and JiaLing), see /D2D3_external folder
            ├── benchmarks                    Internal and External Benchmark results, including the consensus and classification results. 
            ├── database                      Investigating DrugBank database. Understanding DrugBank compounds in relation to our filters. 
            ├── database_PDSP                 Investigating PDSP database.
            ├── database_bindingDB            Investigating BindingDB database.
            ├── dataset_characteristics       Manuscript figures generated showing the characteristics of the D2 and D3 dataset.
            ├── dataset_quality               Testing dataset quality. Comparing between 6 different datasets for the D2 and 3 different datasets for the D3 using XGBoost and RF models.
            ├── dataset_training              Creating training dataset (overlaps with some files in /datasets folder). Investigating the filtering process in our training dataset, including selectivity dataset.
            ├── dnn_analysis                  DNN hotfixing issues with normalization and ICP feature. Also some DNN versus tree-based algorithm prediction comparison plots are done here. 
            ├── drugbank                      More extensive analysis of the DrugBank compounds through our filtering system. 
            ├── epoch_evaluation              Evaluating which epoch overtraining curve to see where epoch is being overtrained in relation to its performance against the validation loss. 
            ├── linear_regression             Linear regression model analysis
            ├── parameter_analysis            Analysis of the hyperparameterization tuning protocol. Original models for these are found in /models/parameter_analysis. 
            ├── prediction_plots              Obsolete files used for plotting model benchmarks and showing other visualization for presentation purposes. 
            ├── screening                     Hotfixing NCI screening process. Testing failed compounds and debugging. 
            ├── top_representatives_shap      Analysis on: SHAP values, visualizing Ph2D features in compounds, seeing the feature trends in the training dataset
            └── util                          Scripts to either create directly or create the scripts necessary for: regression and classification datasets, selectivity dataset; scripts for running XGB, RF, and DNN models; and monitoring DNN model building process. 
