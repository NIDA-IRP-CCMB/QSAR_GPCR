<a name="top"></a>
# QSAR modeling of Dopamine D2, D3, and D3-selective receptors
## Table of Contents
- [Introduction](#introduction)
- [Data preparation](#data_prep)
- [Model building](#model_build)
- [Model prediction](#model_prediction)
- [Post-model analysis](#analysis)
- [Runtime environment](#environment)
- [Structure of the repository](#structure)

- [Contributors](#contributors)


<a name="introduction"></a>
## Introduction
The code in this repository is the chemoinformatics part of our project in developing an infrastrature for developing QSAR models targeting dopamine receptors (DR), specifically Dopamine D2 and Dopamine D3 receptors (D2 and D3), and the selectivity for the D3 over D2. 

### Workflow

<img width="468" alt="image" src="https://github.com/user-attachments/assets/ee41fbd2-fb37-420c-823a-6081dcb601cf">


The process is separated into four phases: 

1) Data Preparation
2) Model Building
3) Model Prediction
4) Post-model Analysis

<a name="data_prep"></a>
### 1) Data Preparation
The data is retrieved from the ChEMBL database. We isolated the dataset to "antagonist-mode", determining the compound's binding affinity for the receptors in an antagonist-mode. Note that within the antagonist-mode dataset, there may be agonist compounds found as well. Furthermore, we included many additional filters to enhance the quality of the dataset to improve the performance of the models, which is further noted in datasets/README.md.

The scripts to create regression and classification datasets that includes selectivity dataset for both can be found in notebooks/main/get_selectivity_datasets.ipynb and notebooks/main/get_classification_dataset.ipynb.

<a name="model_build"></a>
### 2) Model Building
The machine learning algorithms we used include XGBoost (XGB), RandomForest (RF), and Deep Neural Network (DNN). The predictions from the models generated in each of these three individual algorithms were combined and averaged to give the "consensus" predictions.

#### Running XGBoost and RF 

The scripts to run XGBoost and RF  models can be found in examples/main/prep_XGB_RF_biowulf.ipynb.ipynb. This notebook generates the hierarchy of folders and scripts to run the XGBoost and RF models.

#### Running DNN
Deep Learning algorithm was applied in our NIH Biowulf cluster. The code can be found in core/deeplearning.py and core/run_deeplearning.py. The protocol for hyperparameter tuning can be found in models/parameter_analysis, with the corresponding analysis in notebooks/parameter_analysis.

The scripts to run DNN models can be found in examples/main/prep_DNN.ipynb. This notebook generates the hierarchy of folders and scripts to run the DNN models.

<a name="model_prediction"></a>
### 3) Model Prediction

The scripts for making model predictions should have already been generated through the scripts used to run the associated machine/deep learning models.

For analysis, the benchmarks for the regression and classification models of the D2-affinity, D3-affinity, and selectivity models can be found in notebooks/benchmarks. This includes the consensus predictions. 

<a name="analysis"></a>
### 4) Post-model Analysis

The SHAP analysis is concurrently being ran with the model building script (after model is built), but if it is not done or needs to be reran, please see the notebooks/main/prep_SHAP.ipynb, which would generate the scripts to run the SHAP analysis for the models. 

* SHAP analysis can be found in notebooks/post_model_analysis/shap_features_consensus.ipynb. 

* The visualization of Ph2D features mapped onto the representative compounds can be found in notebooks/post_model_analysis/Pharm2D_Visualization_DR.ipynb. These need to be manually changed with the CHEMBL ID and desired mapped Ph2D features.

* The trend of the Ph2D and non-Ph2D values within our training dataset can be found in notebooks/post_model_analysis/Ph2D_distributions.ipynb and notebooks/post_model_analysis/Non-Ph2D_distributions.ipynb, respectively.

* Other post-model analysis can be found in the same folder, notebooks/post_model_analysis. Please see the README there.

<a name="environment"></a>
## Runtime environment
### Python version and Dependencies 
Everything in this repository, including the model generation, were created using Python version 3.7.6 (installer from Anaconda Python [Anaconda3-2020.02-*.sh](https://repo.anaconda.com/archive/)). Additionally, these were created using the major modules listed below. Xgboost, Scikit-learn, RDKit and prerequisites were installed from the anaconda repository (using the channel rdkit for rdkit and its prerequisites). MolVS was installed under anaconda using pip. Scikit-learn was installed by default with the full anaconda distribution (not miniconda). The commands to install xgboost, RDKit and MolVS into an active python virtual environment are as follows:
      conda install py-xgboost
      conda install -c rdkit rdkit 
      pip install MolVS
      pip install -U scikit-learn


Major modules used and their corresponding versions in this repository are as follows:

      MolVS==0.1.1
      scikit-image==0.16.2
      scikit-learn==1.0.1
      rdkit-pypi==2020.03.3
      xgboost==1.0.2
      shap==0.41.0
      tensorflow==2.5.0              
      tensorflow-gpu==2.5.0


The requirements.txt file is attached. Please run the following:

            pip install -r requirements.txt

Tensorflow 2.5.0 version is being used 

### Unittest

Currently, the unittest was moved to the ai-x repository. This is the unittest for dataset filters.

            python -m unittest -v datasets/scripts/unittest_filters_dop.py -b

<a name="structure"></a>
## Structure of the repository

The major scripts are located in the core directory. 

      QSAR_D2D3
        ├── core                 All major py files
        ├── datasets             Datasets of D2 and D3 antagonists
        ├── examples             Dataset preparation and model production, including hyperparameter tuning protocol
        ├── notebooks            All analysis work
        └── unittest             Unittest




<a name="contributors"></a>
## Contributors
* Sung Joon (Jason) Won
* Benjoe Rey B. Visayas
* Kuo Hao Lee
* Julie Nguyen
* Lei Shi*  
