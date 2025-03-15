import os
import glob
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import math
from statistics import mean
from itertools import product


def pretreatment(df):
    cols = ['epochs', 'hidden_layers', 'neurons', 'learning_rate', 'batch_size', 'dropout']
    df = df.drop_duplicates(subset=cols, keep='first')
    df['avg R'] = (df['Rp'] + df['Rs']) / 2
    df = df.sort_values(by="R2", ascending=False)
    return df


def get_df(filename, pretreat = True):
    cols = ['i', 'epochs', 'hidden_layers', 'neurons', 'learning_rate', 'batch_size', 'dropout', 'time', 'R2', 'Rp',
            'Rs', 'MSE', 'RMSE']
    df = pd.read_csv(filename, header=None, sep=' ')
    df.columns = cols
    if pretreat:
        df = pretreatment(df)
    return df


def get_data(modeldir, pretreat = True):
    files = []
    for subdir, _, _ in os.walk(modeldir):
        if 'feasible' in subdir:
            continue  # Skip the 'feasible' folder
        filepath = os.path.join(subdir, 'results_list.txt')
        if os.path.isfile(filepath):
            files.append(filepath)

    #dfs = [get_df(file) for file in sorted(files)[:run]]
    dfs = [get_df(file, pretreat) for file in sorted(files)[:]]

    if not dfs:
        return pd.DataFrame()  # Return an empty DataFrame if no files were read

    df = pd.concat(dfs)

    if pretreat:
        df = pretreatment(df)

    print(f"Sample of {len(df)}\n")
    return df


def get_subset_df(df, dict_para):
    """ Give dict_para, get a subset of the combinations find in dict_para among df """
    combinations = list(product(*dict_para.values()))
    # Filter the DataFrame for each combination
    df_subset = pd.concat([df[(df[list(dict_para.keys())] == combo).all(axis=1)] for combo in combinations])
    df_subset = pretreatment(df_subset)

    return df_subset


def get_feasible_df(modeldir):
    """modeldir should be called 'feasible'"""
    df = get_df(modeldir + '/feasible/results_list.txt')
    df = pretreatment(df)
    print("feasible df is size ", len(df))

    return df


def plot_parameter(ax, i, df, parameter, y, ylim=None, logx=False):
    df.plot(ax=ax[i], x=parameter, y=y, style='.', ylim=ylim, logx=logx)
    ax[i].set_ylabel(y)
    ax[i].set_xticks(sorted(df[parameter].unique()))
    ax[i].set_xticklabels(sorted(df[parameter].unique()), fontsize=8, rotation=90)
    ax[i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[i].tick_params(top=True, bottom=True, left=True, right=True,
                      labelleft=True, labelbottom=True)
    ax[i].get_legend().remove()


def plot_hist(ax, i, df, parameter):
    df = df.sort_values(by=parameter)
    df[parameter].value_counts().sort_index().plot(ax=ax[i], kind='bar', width=0.8)
    ax[i].tick_params(axis='x', labelrotation=90, labelsize=8)
    ax[i].tick_params(axis='y', labelsize=8)
    ax[i].legend().set_visible(False)
    ax[i].set_xlabel(parameter, fontsize=10)
    ax[i].set_ylabel('Sample size (#)', fontsize=10)
    ax[i].tick_params(top=False, bottom=True, left=True, right=True,
                      labelleft=True, labelbottom=True)


def plot_heatmap(df, parameter, metric, inclements=0.025, save2png = ""):
    '''
    metric can be "R2" or "avg R". Other metrics should work
    inclements are measured from top performing model to bottom
    Change "values" from "proportions" to "freq" to measure frequency
    Code ref: https://stackoverflow.com/questions/43330205/heatmap-from-columns-in-pandas-dataframe
    '''

    figure_size = (10, 10)
    show_top_n_rows = 5
    # to control the heatmap scale legend axis
    vmin = 0.0
    vmax = 0.6

    df2 = df[[parameter, metric]]
    ranges = sorted(np.arange(max(df2[metric]), 0.0, -inclements).tolist()) # 0.7 is the lowest the heatmap will go
    df2["range"] = pd.cut(df2[metric], ranges)
    df2 = pd.DataFrame({'freq': df2.groupby([parameter, "range"]).size()}).reset_index()

    # get dictionary values, mapping from parameter value to its cumulative sum
    ls_parameter = list(df2[parameter].unique())
    dict_sum = {}
    for p in ls_parameter:
        dict_sum[p] = sum(df2[df2[parameter] == p]['freq'])

    # narrowing down df to the wanted y-axis ranges, in the given ymin and ymax
    df["range"] = pd.cut(df[metric], ranges)
    df3 = pd.DataFrame({'freq': df.groupby([parameter, "range"]).size()}).reset_index()
    df3['proportions'] = df3.apply(lambda row: row.freq / dict_sum[row[parameter]], axis=1)

    # create pivot table, days will be columns, hours will be rows
    fig, ax = plt.subplots(figsize=figure_size)
    piv = pd.pivot_table(df3, values='proportions', index=['range'], columns=[parameter], fill_value=0)
    #piv = pd.pivot_table(df3, values='freq', index=['range'], columns=[parameter], fill_value=0)
    piv = piv.tail(show_top_n_rows) # piv df is going from least to greatest
    # plot pivot table as heatmap using seaborn
    sns.heatmap(piv, square=True, cmap='Reds', cbar_kws={"shrink": 0.5}, vmin=vmin, vmax=vmax, cbar=True, annot=True,
                fmt='.2f')
    ax.invert_yaxis()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.tight_layout()
    if save2png:
        plt.savefig(save2png, dpi=300, transparent=True, bbox_inches="tight")
    plt.show()
    return piv.round(3)


def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0
        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];

        list1.remove(max1);
        final_list.append(max1)
    return final_list


def get_list(lsi, lsj, lsij, n):
    idx = []
    score = []
    for i in lsij:
        #print(i, lsi.index(i)+lsj.index(i))
        idx.append(i)
        score.append(lsi.index(i)+lsj.index(i))
        _df = pd.DataFrame()
        _df['name'] = idx
        _df['score'] = score
        _df = _df.sort_values(by="score").reset_index(drop=True)
    return list(_df['name'][0:n])
# _df = get_list(lsi, lsj, lsij, 3)


def heatmap_two_param(df, param1, param2, scoring):
    '''
    Finds the relationship between two parameters, and counts the occurences for above a certain threshold (in a given R choice)
    More information on using 'count' here: https://stackoverflow.com/questions/41464034/summing-multiple-rows-having-duplicate-columns-pandas
    '''
    if scoring == "RMSE":
        tolerance = 1.10
    elif scoring == "R2":
        tolerance = 0.60
    elif scoring == "Rp" or "Rs":
        tolerance = 0.80
    if scoring == "RMSE":
        tolerance = 0.6

    vmin = 0
    df.loc[df[scoring] >= tolerance, 'count'] = 1
    df.loc[df[scoring] < tolerance, 'count'] = 0

    df = df.sort_values(by=scoring, ascending=False)
    df2 = df[[param1, param2, 'count', scoring]]
    # df2 = df2.drop_duplicates(subset = ['hidden_layers', 'neurons'], keep = 'first')
    df2 = df2.groupby([param1, param2], as_index=False)['count'].sum()
    total = sum(df2['count'])
    df2['proportions'] = df2['count'].apply(lambda row: row / total)
    df_piv = pd.pivot_table(df2, values='count', index=param1, columns=param2, fill_value=0)


    vmax = max(df2["count"])

    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(df_piv, square=True, cmap='Reds', cbar_kws={"shrink": 0.5}, vmin=vmin, vmax=vmax, cbar=True, annot=True,
                fmt='.2f')
    ax.set_title(f'tolerance = {tolerance} using {scoring}')

    return df2


def get_total_time_plot(scoring, df, df_subset, save2png):
    if scoring == "RMSE":
        threshold = 1.10
    elif scoring == "R2":
        threshold = 0.60
    elif scoring == "Rp" or "Rs":
        threshold = 0.80
    if scoring == "RMSE":
        df = df.sort_values(by = scoring, ascending = True)
    else:
        df = df.sort_values(by = scoring, ascending = False)

    # Scatter plot
    plt.figure(figsize=(10, 6))

    # Separate data points above and below 0.80 (max/min)
    if scoring == "RMSE":
        above_080 = df[df[scoring] <= threshold] # good models
        below_equal_080 = df[df[scoring] > threshold]
        print("df_subset in red zone: ", len(df_subset[df_subset[scoring] <= threshold]), len(df_subset))
    else:
        above_080 = df[df[scoring] >= threshold] # good models
        below_equal_080 = df[df[scoring] < threshold]
        print("df_subset in red zone: ", len(df_subset[df_subset[scoring] >= threshold]), len(df_subset))

    # Scatter plot with different colors for points above and below 0.80
    s=30
    if scoring == "RMSE":
        plt.scatter(above_080['time'], above_080[scoring], color='red', alpha=0.5, s=s, label=f'Above {threshold}')
        plt.scatter(below_equal_080['time'], below_equal_080[scoring], color='blue', alpha=0.5, s=s, label=f'Below {threshold}')
        plt.scatter(df_subset['time'], df_subset[scoring], color='black', alpha=1, s=s, label="Feasible Starting Hyperparameters")
        plt.ylim(0, 8)
    else:
        plt.scatter(above_080['time'], above_080[scoring], color='red', alpha=0.5, s=s, label=f'Above {threshold}')
        plt.scatter(below_equal_080['time'], below_equal_080[scoring], color='blue', alpha=0.5, s=s, label=f'Below {threshold}')
        plt.scatter(df_subset['time'], df_subset[scoring], color='black', alpha=1, s=s, label="Feasible Starting Hyperparameters")
        if scoring == "Rp" or "Rs":
            plt.ylim(0.0, 0.90)
        if scoring == "R2":
            plt.ylim(0.0, 0.80)

    # Customize the plot
    fontsize = 16
    plt.xscale('log')  # Use a logarithmic scale for the x-axis
    plt.yticks(fontsize=fontsize)

    plt.grid(True)
    max_time = df['time'].max()
    min_time = df['time'].min()
    plt.xlim(min_time, max_time)

    if save2png:
        plt.xticks(fontsize=0)
        plt.savefig(save2png, dpi=300, transparent=True, bbox_inches="tight")
    else:
        plt.title(f'Scatter Plot: Time vs. {scoring}')
#         plt.legend(loc="upper right", bbox_to_anchor=(1.5, 0.6), fontsize=14)
        plt.ylabel(scoring, fontsize=fontsize)
        plt.xlabel('Time (minutes in log scale)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

    plt.show()


def parameter_time(df, parameter, color, save2png = ""):
    # Create a box plot with data subsampling
    plt.figure(figsize=(6, 8))
    ax = sns.boxplot(x=parameter, y='time', data=df, color = color)  # Subsample 1000 points

    # Set y-axis to log scale
    ax.set_yscale('log')
    ax.invert_yaxis()

    # Customize labels and title
    fontsize = 20
    if save2png:
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout()
        ax.tick_params(left = True, right = True, top = True, bottom = True, labelleft = True, labelbottom = True)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize, rotation = 270)  # Set x-axis tick label font size
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize, rotation = 270)  # Set x-axis tick label font size
        ax.set_yticklabels([])
        ax.xaxis.set_ticks_position('top')  # Move x-axis ticks to the top
        if parameter == "learning_rate":
            ax.set_xticklabels([])
#         plt.tick_params(left = True, right = True, top = True, bottom = True, labelleft = False, labelbottom = False)
        plt.savefig(save2png, dpi = 300, transparent = True, bbox_inches = "tight")
    else:
        ax.set_xlabel(parameter)
        ax.set_ylabel('Time (log scale)')
        ax.set_title(f'Box Plot of {parameter} vs Time')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize, rotation = 90)  # Set x-axis tick label font size
        ax.tick_params(left = True, right = True, top = True, bottom = True, labelleft = True, labelbottom = True)
    plt.show()


def get_top_parameters(df, parameter, metric, inclements):
    df2 = df[[parameter, metric]]
    ranges = sorted(np.arange(max(df2[metric]), 0.0, -inclements).tolist())
    df2["range"] = pd.cut(df2[metric], ranges)
    df2 = pd.DataFrame({'freq': df2.groupby([parameter, "range"]).size()}).reset_index()

    # get dictionary values, mapping from parameter value to its cumulative sum
    ls_parameter = list(df2[parameter].unique())
    dict_sum = {}
    for p in ls_parameter:
        dict_sum[p] = sum(df2[df2[parameter] == p]['freq'])

    df["range"] = pd.cut(df[metric], ranges)
    df3 = pd.DataFrame({'freq': df.groupby([parameter, "range"]).size()}).reset_index()
    df3['proportions'] = df3.apply(lambda row: row.freq / dict_sum[row[parameter]], axis=1)

    max_range = max(df3['range'].values.to_list())
    df3 = df3[df3['range'] == max_range]
    ls_top_parameters = list(df3.sort_values(by="proportions", ascending=False)[parameter])

    return ls_top_parameters


def criteria(df, metric, inclements=0.025):
    """
    This criteria finds the top 3 performing parameters within the hyperparameter
    Metric can be "avg R" or "R2"
    """
    ls_parameters = ['epochs', 'hidden_layers', 'neurons', 'learning_rate', 'batch_size', 'dropout']
    dict_para_to_top = {}
    for parameter in ls_parameters:
        ls_top_parameters = get_top_parameters(df, parameter, metric, inclements)  # for pearson R
        dict_para_to_top[parameter] = sorted(ls_top_parameters[0:3])

    return dict_para_to_top

def criteria2(df, metric, inclements=0.025):
    """
       This criteria finds the top performing parameter and returns the surrounding parameters.
    Metric can be "avg R" or "R2"
    """
    ls_parameters = ['epochs', 'hidden_layers', 'neurons', 'learning_rate', 'batch_size', 'dropout']
    dict_para_to_top = {}
    i = 0
    for parameter in ls_parameters:
        n = math.ceil(len(df[parameter].unique()))
        ls_top_parameters = get_top_parameters(df, parameter, metric, inclements)  # for pearson R
        ls_sorted_parameters = sorted(ls_top_parameters)
        if i < 7:
            top_parameter = ls_top_parameters[0]
            index_middle = ls_sorted_parameters.index(top_parameter)  # top performing parameter
            index_bottom = index_middle - 1
            index_top = index_middle + 1

            ls_indexes = []
            if index_middle > 0:
                ls_indexes.append(index_bottom)
            ls_indexes.append(index_middle)
            if index_middle < len(ls_top_parameters) - 1:
                ls_indexes.append(index_top)

            ls_chosen_parameters = []
            for index in ls_indexes:
                ls_chosen_parameters.append(ls_sorted_parameters[index])
                dict_para_to_top[parameter] = sorted(ls_chosen_parameters)
        else:
            dict_para_to_top[parameter] = sorted(ls_top_parameters[0:2])
        i = i + 1

    return dict_para_to_top


def subsample(df, n_sample, n_states, metric, inclements):
    all_dict = []
    for state in range(0, n_states):
        subset_df = df.sample(n_sample, random_state=state)
        dict_para = criteria(subset_df, metric, inclements)
        all_dict.append(dict_para)
    return all_dict


from collections import defaultdict, Counter

def find_mode_hyperparameters(dicts):
    counts = defaultdict(lambda: defaultdict(int))
    for dic in dicts:
        for key, value in dic.items():
            counts[key][tuple(value)] += 1

    modes = {}
    for key, val_dict in counts.items():
        max_count = max(val_dict.values())
        mode = [k for k, v in val_dict.items() if v == max_count]
        modes[key] = {'mode': mode, 'count': max_count}

    return modes


def get_majority_voting(ls_modes):

    def find_mode(ls_batch_size):
        # Convert the list of tuples of tuples into a list of tuples to be compatible with Counter
        tuples_list = [tuple(sublist[0]) for sublist in ls_batch_size]

        # Use Counter to find the frequency of each tuple
        frequency = Counter(tuples_list)

        # Get the highest frequency count
        max_freq = max(frequency.values())

        # Find all tuple elements that have the maximum frequency
        mode_tuples = [key for key, value in frequency.items() if value == max_freq]

        return mode_tuples


    ls_batch_size = []
    ls_dropout = []
    ls_epochs = []
    ls_hidden_layers = []
    ls_learning_rate = []
    ls_neurons = []

    for mode in ls_modes:
        ls_batch_size.append(mode["batch_size"]["mode"])
        ls_dropout.append(mode["dropout"]["mode"])
        ls_epochs.append(mode["epochs"]["mode"])
        ls_hidden_layers.append(mode["hidden_layers"]["mode"])
        ls_learning_rate.append(mode["learning_rate"]["mode"])
        ls_neurons.append(mode["neurons"]["mode"])
    dict_output = {}
    dict_output["batch_size"] = find_mode(ls_batch_size)
    dict_output["dropout"] = find_mode(ls_dropout)
    dict_output["epochs"] = find_mode(ls_epochs)
    dict_output["hidden_layers"] = find_mode(ls_hidden_layers)
    dict_output["learning_rate"] = find_mode(ls_learning_rate)
    dict_output["neurons"] = find_mode(ls_neurons)
    return dict_output


# def majority_sampling(df, metric, inclements):
#     n_states = 100
#     ls_modes = []
#     for n_sample in range(0, len(df), 500):
#         all_dict = subsample(df, n_sample, n_states, metric, inclements)
#         modes_result = find_mode_hyperparameters(all_dict)
#         ls_modes.append(modes_result)
#     dict_output = get_majority_voting(ls_modes)
#     return dict_output


# from collections import Counter
# def get_majority_voting(ls_modes):
#
#     def find_mode(ls_batch_size):
#         # Convert the list of tuples of tuples into a list of tuples to be compatible with Counter
#         tuples_list = [tuple(sublist[0]) for sublist in ls_batch_size]
#
#         # Use Counter to find the frequency of each tuple
#         frequency = Counter(tuples_list)
#
#         # Get the highest frequency count
#         max_freq = max(frequency.values())
#
#         # Find all tuple elements that have the maximum frequency
#         mode_tuples = [key for key, value in frequency.items() if value == max_freq]
#
#         return mode_tuples
#
#
#     ls_batch_size = []
#     ls_dropout = []
#     ls_epochs = []
#     ls_hidden_layers = []
#     ls_learning_rate = []
#     ls_neurons = []
#
#     for mode in ls_modes:
#         ls_batch_size.append(mode["batch_size"]["mode"])
#         ls_dropout.append(mode["dropout"]["mode"])
#         ls_epochs.append(mode["epochs"]["mode"])
#         ls_hidden_layers.append(mode["hidden_layers"]["mode"])
#         ls_learning_rate.append(mode["learning_rate"]["mode"])
#         ls_neurons.append(mode["neurons"]["mode"])
#     dict_output = {}
#     dict_output["batch_size"] = find_mode(ls_batch_size)
#     dict_output["dropout"] = find_mode(ls_dropout)
#     dict_output["epochs"] = find_mode(ls_epochs)
#     dict_output["hidden_layers"] = find_mode(ls_hidden_layers)
#     dict_output["learning_rate"] = find_mode(ls_learning_rate)
#     dict_output["neurons"] = find_mode(ls_neurons)
#     return dict_output
