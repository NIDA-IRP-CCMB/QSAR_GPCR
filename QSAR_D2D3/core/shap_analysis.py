import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
import seaborn as sns

import warnings

from draw_Pharm2D import get_description

warnings.filterwarnings('ignore')


def get_df_SHAP_percent(df):
    total_sum = df['SHAP Values Mean'].sum()
    # Divide each value in the column by the total sum
    df['SHAP Percent'] = df['SHAP Values Mean'] / total_sum
    return df


def plot_shap(df, col_name, title, ylim, y_tick_interval, xfontsize, bar_color, save2png):
    n_head = 10

    # Select the first 30 rows
    df = df.sort_values(by=col_name, ascending=False).reset_index(drop=True)
    df_first_30 = df.head(n_head).copy()
    df_first_30['Features'] = df_first_30['Features'].str.replace('Ph2D_', '')

    # Create a bar plot
    plt.figure(figsize=(6, 6))
    plt.bar(df_first_30['Features'], df_first_30[col_name], width=0.85, color=bar_color)
    # ylim
    plt.ylim(ylim[0], ylim[1])
    y_ticks = np.arange(ylim[0], ylim[1]+0.001, y_tick_interval)
    # plt.tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=True)
    plt.xticks(rotation=90, fontsize=xfontsize, fontweight="semibold")  # Rotate x-axis labels for better visibility
    plt.tight_layout()

    # Show the plot
    if save2png:
        #         plt.yticks(y_ticks, fontsize = 0)
        # plt.yticks(y_ticks, fontsize=0)
        plt.tick_params(axis='y', which='both', left=True, labelleft=False)  # Hide y-ticks and labels
        #plt.xticks(ticks=range(len(df_first_30['Features'])), labels=[""] * len(df_first_30['Features']))
        #plt.xticks(rotation=90, fontsize=xfontsize, fontweight="semibold")  # Rotate x-axis labels for better visibility
        # plt.xticks(x_ticks, fontsize=0)
        plt.yticks(y_ticks, fontsize=0)
        plt.savefig(save2png, bbox_inches='tight', transparent=True, dpi=300)
    else:
        plt.yticks(y_ticks, fontsize=16)
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel(col_name)

    plt.show()


def shap_analysis(datadir, method, model):

    directory = f"{datadir}{model}/reg_{method}_0.00"
    files = os.listdir(directory)
    feat_imp_files = [file for file in files if
                      file.startswith("feature_importances_reg") and file.endswith(".dat") and 'norm' not in file]
    n_models = len(feat_imp_files)

    for i, file in enumerate(feat_imp_files):
        with open(f"{directory}/{file}", 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(list(data.items()), columns=['Features', 'SHAP Values'])
        df_add = df[["Features", f"SHAP Values"]]
        df_add = df_add.rename(columns={"SHAP Values": f"SHAP Values {i}"})
        if i == 0:
            df_frame = df_add
        elif i > 0:
            df_frame = pd.merge(df_frame, df_add, on="Features", how='outer')
    df = df_frame

    shap_columns = [f'SHAP Values {i}' for i in range(n_models)]
    df['SHAP Values Mean'] = df[shap_columns].apply(lambda row: pd.to_numeric(row, errors='coerce').mean(), axis=1)
    df['SHAP Values StdDev'] = df[shap_columns].apply(lambda row: pd.to_numeric(row, errors='coerce').std(), axis=1)
    df['Description'] = df['Features'].apply(get_description)

    df = get_df_SHAP_percent(df)
    df.sort_values(by="SHAP Values Mean", ascending=True, inplace=True)

    df = df[["Features", "Description", "SHAP Values Mean", "SHAP Values StdDev", "SHAP Percent"]]
    #df = df[["Features", "Description", "SHAP Values Mean", "SHAP Values StdDev"]]

    df_shap = df.copy()
    df_shap.sort_values(by="SHAP Values Mean", ascending=False, inplace=True)


    # prioritize ordering by SHAP values
    df = df_shap
    df_with_Ph2D = df[df["Features"].str.contains("Ph2D_")].reset_index(drop=True)
    df_without_Ph2D = df[~df["Features"].str.contains("Ph2D_")].reset_index(drop=True)
    # df_with_Ph2D = get_df_SHAP_percent(df_with_Ph2D)
    # df_without_Ph2D = get_df_SHAP_percent(df_without_Ph2D)

    return df_with_Ph2D, df_without_Ph2D


def get_shap_consensus(datadir_xgb, datadir_rf, datadir_dnn, model):
    """
    Warning: Features in XGB and RF are pruned. DNN is not. DNN is missing feature Ipc (situational)
    We do not exclude any feature information here, so all 200 features are captured in nonPh2D dfs.
    The features that are missing in XGB/RF, and DNN (Ipc) have NaN entry values, which are replaced with 0s.
    Because consensus is the average of the three predictions, I deemed it would be more accurate to take the average
    of the three algorithms for all 200 features even if the features are missing from other algorithms.
    """
    # retrieve dfs per algorithm
    Ph2D_xgb, nonPh2D_xgb = shap_analysis(datadir_xgb, "xgb", model)
    Ph2D_rf, nonPh2D_rf = shap_analysis(datadir_rf, "rf", model)
    Ph2D_dnn, nonPh2D_dnn = shap_analysis(datadir_dnn, "dnn", model)

    # clean up dfs by dropping unnecessary columns
    drop_columns = ['SHAP Values StdDev', 'SHAP Percent']
    Ph2D_xgb.drop(columns=drop_columns, inplace=True)
    Ph2D_rf.drop(columns=drop_columns, inplace=True)
    Ph2D_dnn.drop(columns=drop_columns, inplace=True)
    nonPh2D_xgb.drop(columns=drop_columns, inplace=True)
    nonPh2D_rf.drop(columns=drop_columns, inplace=True)
    nonPh2D_dnn.drop(columns=drop_columns, inplace=True)

    # rename SHAP Values Mean
    Ph2D_xgb.rename(columns={'SHAP Values Mean': 'XGB'}, inplace=True)
    Ph2D_rf.rename(columns={'SHAP Values Mean': 'RF'}, inplace=True)
    Ph2D_dnn.rename(columns={'SHAP Values Mean': 'DNN'}, inplace=True)
    nonPh2D_xgb.rename(columns={'SHAP Values Mean': 'XGB'}, inplace=True)
    nonPh2D_rf.rename(columns={'SHAP Values Mean': 'RF'}, inplace=True)
    nonPh2D_dnn.rename(columns={'SHAP Values Mean': 'DNN'}, inplace=True)

    # Merge the 3 dfs
    Ph2D_consensus = pd.merge(Ph2D_dnn, Ph2D_xgb, on=['Features', 'Description'])
    Ph2D_consensus = pd.merge(Ph2D_consensus, Ph2D_rf, on=['Features', 'Description'])
    Ph2D_consensus['SHAP Values Mean'] = Ph2D_consensus[["XGB", "RF", "DNN"]].mean(axis=1)  # consensus
    Ph2D_consensus = Ph2D_consensus.sort_values(by="SHAP Values Mean", ascending=False).reset_index(drop=True)
    Ph2D_consensus = Ph2D_consensus.rename(columns={'SHAP Values Mean': 'consensus'}).reset_index(drop=True)

    # Merge nonPh2D DataFrames using an outer join to ensure all features are kept.
    # XGB and RF are pruned and missing features. DNN is missing Ipc feature.
    nonPh2D_consensus = pd.merge(nonPh2D_dnn, nonPh2D_xgb, on=['Features', 'Description'], how='outer')
    nonPh2D_consensus = pd.merge(nonPh2D_consensus, nonPh2D_rf, on=['Features', 'Description'], how='outer')
    nonPh2D_consensus[['XGB', 'RF', 'DNN']] = nonPh2D_consensus[['XGB', 'RF', 'DNN']].fillna(0)
    nonPh2D_consensus['SHAP Values Mean'] = nonPh2D_consensus[['XGB', 'RF', 'DNN']].sum(axis=1) / 3
    nonPh2D_consensus = nonPh2D_consensus.sort_values(by="SHAP Values Mean", ascending=False).reset_index(drop=True)
    nonPh2D_consensus = nonPh2D_consensus.rename(columns={'SHAP Values Mean': 'consensus'}).reset_index(drop=True)

    return Ph2D_consensus, nonPh2D_consensus

