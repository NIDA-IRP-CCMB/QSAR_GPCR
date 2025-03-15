import sys, os
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
sys.path.insert(0, core_dir)
from buildmodel import *
from draw_Pharm2D import *

core_dir = home+'/repositories/ai-x/core/keywords'
sys.path.insert(0, core_dir)
from descriptor_setup import dnames, dlist

# from rdkit import Chem
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import Chem

import io
import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets


def get_training_df(target=None, datadir=None, dataset=None):
    def count_bonds(mol):
        return mol.GetNumBonds() if mol else None

    filename = f'train_descs_{target}.pkl'
    if os.path.exists(filename):
        print(f"The file {filename} exists.")
        df = pd.read_pickle(filename)
        return df
    else:
        if not datadir:
            datadir = "~/repositories/ai-DR/models/selectivity_D2D3/selectivity_datasets/C33_0/"
            datadir = os.path.expanduser(datadir)
        else:
            datadir = os.path.expanduser(datadir)
        if not dataset:
            if target == "D2":
                dataset = datadir + "dataset_D2_antagonist_Ki/pubdata0"
            elif target == "D3":
                dataset = datadir + "dataset_D3_antagonist_Ki/pubdata0"
            elif target == "D2D3":
                dataset = datadir + "dataset__ratio_D2_antagonist_Ki_D3_antagonist_Ki/pubdata0"
        else:
             dataset = datadir + dataset
        mode = "reg"
        mols, acts, deletes, changes = read_data4buildmodel(dataset, mode)
        mols, acts = curate_mols(mols, acts, deletes, changes)

        train_mols, train_names, train_acts = all_data(mols, acts)

        topo_names = copy.copy(dnames)
        topo_index = []
        for desc in topo_names:
            i = dnames.index(desc)
            topo_index.append(i)

        train_topo_descs = calc_topo_descs(train_mols, topo_index)
        train_chir_descs = calc_chir_descs(train_mols)
        train_topo_descs = np.concatenate((train_topo_descs, train_chir_descs), axis=1)
        topo_names.append('nChiral')
        topo_names.append('ChirIdx')
        max_index = max(topo_index)
        topo_index.append(max_index + 1)
        topo_index.append(max_index + 2)
        output_ext = ""
        output_dir = ""
        train_phore_descs, phore_sigbits, phore_names = prune_phore_descs(calc_phore_descs(train_mols), output_dir,
                                                                          output_ext)
        train_descs = np.concatenate((train_topo_descs, train_phore_descs), axis=1)

        df = pd.DataFrame(train_descs, columns=topo_names+phore_names)
        df.insert(0, 'pKi', train_acts)
        df.insert(0, 'mol', train_mols)
        df.insert(0, 'CHEMBL', train_names)

        df['Num_Bonds'] = df['mol'].apply(count_bonds)
        df["Phi"] = df["Kappa1"] * df["Kappa2"] / df["HeavyAtomCount"]

        df["Molecular_Flexibility1"] = df["NumRotatableBonds"] / df["HeavyAtomCount"]
        df["Molecular_Flexibility2"] = df["NumRotatableBonds"] / df["Num_Bonds"]

        if target == "D2D3": # D3-selective should be positive
            df["pKi"] = -df["pKi"]

        # save df to pickle file
        df.to_pickle(filename)

    return df


def calculate_proportions(data, bin_edges, total_data_points):
    proportions = []
    for i in range(len(bin_edges) - 1):
        bin_values = data[(data['pKi'] >= bin_edges[i]) & (data['pKi'] < bin_edges[i + 1])]
        # proportion = bin_values.shape[0] / data.shape[0]  # Proportion of values in the bin
        proportion = bin_values.shape[0] / total_data_points  # Proportion of values in the bin
        proportions.append(proportion)
    return proportions


def plot_Ph2D_hist(df, feature, interval, xlim, save2png=False):
    ls_passes = list(df[feature])
    # Data for the first histogram
    data1 = {'pKi': df['pKi'], 'binary': ls_passes}
    df1 = pd.DataFrame(data1)

    # Define the bin edges with a step of 0.10
    bin_edges = [i * interval for i in range(int(df1['pKi'].min() / interval), int(df1['pKi'].max() / interval) + 2)]

    # Calculate proportions for the first histogram
    proportions1 = []
    for i in range(len(bin_edges) - 1):
        bin_values = df1[(df1['pKi'] >= bin_edges[i]) & (df1['pKi'] < bin_edges[i + 1])]
        proportion_1s = bin_values['binary'].mean()
        proportions1.append(proportion_1s)

    # Create a figure and axis with twinx for the second y-axis
    fig, ax1 = plt.subplots()

    # Plot the first histogram as a line plot on the left y-axis
    ax1.plot(bin_edges[:-1], proportions1, marker='o', linestyle='-', color='r', label='Proportion of Present')
    if not save2png:
        ax1.set_xlabel("pKi or delta pKi")
        ax1.set_ylabel('Proportion (for line plot)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
    else:
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    #         ax1.set_yticks([])
    ax1.set_ylim(0, 1.05)

    # Create a twin Axes sharing the xaxis for the second y-axis
    ax2 = ax1.twinx()

    # Data for the second histogram
    df2 = df  # Assuming df is df_d3_selective, adjust if needed
    data2 = {'pKi': df2['pKi'], 'binary': ls_passes}
    df2 = pd.DataFrame(data2)

    # Calculate proportions for the second histogram
    total_data_points = df2.shape[0]
    df_zeros = df2[df2['binary'] == 0]
    df_ones = df2[df2['binary'] == 1]
    proportions_zeros = calculate_proportions(df_zeros, bin_edges, total_data_points)
    proportions_ones = calculate_proportions(df_ones, bin_edges, total_data_points)
    percent0 = round(len(df_zeros) / total_data_points, 2)
    percent1 = round(len(df_ones) / total_data_points, 2)
    if not save2png:
        print(f"Feature {feature} Absent {len(df_zeros)} ({percent0}) Present {len(df_ones)} ({percent1}) Total {total_data_points}")

    # Plot the second histogram as bars on the right y-axis
    ax2.bar(bin_edges[:-1], proportions_ones, width=interval, alpha=0.5, label='Distribution of Present')
    ax2.bar(bin_edges[:-1], proportions_zeros, width=interval, alpha=0.5, label='Distribution of Absent')

    if not save2png:
        ax2.set_ylabel('Proportion (for histogram)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
    else:
        ax2.set_yticklabels([])
    #         ax2.set_yticks([])

    # Set title and legend
    ax1.set_xlim(xlim[0], xlim[1])
    ax2.set_xlim(xlim[0], xlim[1])
    ax2.set_ylim(0, 0.075)
    fig.tight_layout()

    if not save2png:
        title = feature
        plt.title(title)
        fig.legend(bbox_to_anchor=(1.5, 0.5), loc='center right', prop={'size': 12})
        plt.show()
    else:
        plt.savefig(save2png, dpi=300, transparent=True, bbox_inches="tight")
        plt.close()



def plot_NonPh2D_hist(df, feature_name, xlim="", ylim="", y_tick_interval=1, save2png=""):
    if not save2png:
        print("Feature", feature_name, "Min", min(df[feature_name]), "Max", max(df[feature_name]))

    # Round 'pKi' to the nearest "interval"
    interval = 0.2
    df['rounded_pKi'] = np.round(df['pKi'] / interval) * interval
    grouped_df = df.groupby('rounded_pKi').mean().reset_index()

    plt.figure(figsize=(6, 6))
    #plt.scatter(df['pKi'], df[feature_name], s=5)
    plt.plot(grouped_df['rounded_pKi'], grouped_df[feature_name], marker='o', linestyle='-', color="red")

    if xlim == "":
        xlim0 = min(df["pKi"])
        xlim1 = max(df["pKi"])
        plt.xlim(xlim0, xlim1)
    else:
        plt.xlim(xlim[0], xlim[1])
    if ylim == "":
        ylim0 = min(df[feature_name])
        ylim1 = max(df[feature_name])
        plt.ylim(ylim0, ylim1)
        plt.yticks(fontsize=16)
    else:
        plt.ylim(ylim[0], ylim[1])
        y_ticks = np.arange(ylim[0], ylim[1] + 0.1, y_tick_interval)
        plt.yticks(y_ticks, fontsize=16)

    if save2png:
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)
        #         plt.xticks(ticks=plt.gca().get_xticks(), labels=[""] * len(plt.gca().get_xticks()))
        plt.savefig(save2png, bbox_inches='tight', transparent=True, dpi=300)
        plt.close()
    else:
        plt.xlabel('pKi or delta pKi', fontsize=18)
        plt.ylabel('Raw value', fontsize=18)
        plt.xticks(fontsize=16)
        plt.title(feature_name)
        plt.show()
