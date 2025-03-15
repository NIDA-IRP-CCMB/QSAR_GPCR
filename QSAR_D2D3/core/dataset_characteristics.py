import os
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-DR/core'
from misc import *
import time
from statistics import *
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs

# venn diagram
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
# sets
from collections import Counter


def venn_diagram(df1, df2, save2png):
    """
    Outputs an image of venn diagram based on Tanimoto similarity.
    Overlapping compounds are nearly identical compomunds, >0.999.
    :param datadir: str, path to the datadir of dataset
    :param assaydefinition: str, either antagonist or agonist
    :param labels: boolean, either True or False, keeps labels on or off
    :param output_dir: str, output directory name
    :return: image of venn diagram
    """

    color1 = "midnightblue"
    color2 = "maroon"
    mol_dup_pairs = get_mol_dup_pairs(df1, df2)

    A = len(df1['chembl'])
    B = len(df2['chembl'])
    A_U_B = len(mol_dup_pairs)

    df_combined = pd.concat([df1, df2])
    sets = Counter()
    sets['01'] = B - A_U_B
    sets['11'] = A_U_B
    sets['10'] = A - A_U_B

    if save2png:
        A_name = ""
        B_name = ""
        v = venn2(subsets=sets, set_labels=(A_name, B_name), set_colors=(color1, color2))
        for idx, subset in enumerate(v.subset_labels):
            v.subset_labels[idx].set_visible(False)
        plt.title('')
        plt.savefig(save2png, transparent = True, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        A_name = "D2"
        B_name = "D3"
        v = venn2(subsets=sets, set_labels=(A_name, B_name), set_colors=(color1, color2))
        # plt.title("Venn Diagram")
        plt.show()
    plt.close()
#
# def venn_diagram(datadir, chembl_version, assaydefinition, save2png):
#     """
#     Outputs an image of venn diagram based on Tanimoto similarity.
#     Overlapping compounds are nearly identical compomunds, >0.999.
#     :param datadir: str, path to the datadir of dataset
#     :param assaydefinition: str, either antagonist or agonist
#     :param labels: boolean, either True or False, keeps labels on or off
#     :param output_dir: str, output directory name
#     :return: image of venn diagram
#     """
#     df_d2 = get_data_df(f"{datadir}/{chembl_version}/dataset_D2_{assaydefinition}_Ki/pubdata")
#     df_d3 = get_data_df(f"{datadir}/{chembl_version}/dataset_D3_{assaydefinition}_Ki/pubdata")
#     mol_dup_pairs = get_mol_dup_pairs(df_d3, df_d2)
#
#     A = len(df_d2['chembl'])
#     B = len(df_d3['chembl'])
#     A_U_B = len(mol_dup_pairs)
#
#     # label
#     df_d2_d3 = pd.concat([df_d2, df_d3])
#     sets = Counter()
#     sets['01'] = B - A_U_B
#     sets['11'] = A_U_B
#     sets['10'] = A - A_U_B
#
#     if save2png:
#         A_name = ""
#         B_name = ""
#         v = venn2(subsets=sets, set_labels=(A_name, B_name), set_colors=('mediumblue', 'firebrick'))
#         for idx, subset in enumerate(v.subset_labels):
#             v.subset_labels[idx].set_visible(False)
#         plt.title('')
#         plt.savefig(save2png, dpi=300, bbox_inches='tight')
#         plt.show()
#     else:
#         A_name = "D2"
#         B_name = "D3"
#         v = venn2(subsets=sets, set_labels=(A_name, B_name), set_colors=('mediumblue', 'firebrick'))
#         plt.title(f"{chembl_version}_{assaydefinition}")
#         plt.show()
#     plt.close()
#
#     return df_d2, df_d3, A_U_B
#

def get_smooth(x,y):
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(x.min(), x.max(), 300)
    # type: BSpline
    spl = make_interp_spline(x, y, k=1)
    power_smooth = spl(xnew)
    return(xnew, power_smooth)


def get_data_smi(datain):
    df = pd.read_csv(datain, sep='\t', header = None)
    sel_df = df[0]
    sel_df1=sel_df
    return(sel_df1)

def cal_sml(mol_in_i, mol_in_j):
    fp_i = Chem.GetMorganFingerprintAsBitVect(mol_in_i, 2, nBits=2048)
    fp_j = Chem.GetMorganFingerprintAsBitVect(mol_in_j, 2, nBits=2048)
    sml=round(DataStructs.TanimotoSimilarity(fp_i,fp_j),3)
    return sml

# get simle string from a dataframe
def get_smile_df(dfin, i):
    return Chem.MolFromSmiles(dfin['smile'][i])

# get simle string from a list
def get_smile_ls(lsin, i):
    return Chem.MolFromSmiles(lsin[i])

## given a list of smile string
## retrun pairwise smiliarty in list
def get_fp(lsin):
    nmol = len(lsin)
    t1 = time.time()
    fp=[]
    for i in range(nmol):
        for j in range(i+1, nmol):
            fp.append(cal_sml(get_smile_ls(lsin, i),get_smile_ls(lsin, j)))
    t2 = time.time()
    print("TIMER: Function completed time is: %.5f" % (t2 - t1))
    len(fp)
    return(fp)


def get_fp_cross(lsin1, lsin2):
    """
    Given two lists of smile string, return pairwise similarity between lists
    :param lsin1: list
    :param lsin2: list
    :return: list
    """
    nmol1 = len(lsin1)
    nmol2 = len(lsin2)
    t1 = time.time()
    fp=[]
    for i in range(nmol1):
        for j in range(nmol2):
            fp.append(cal_sml(get_smile_ls(lsin1, i),get_smile_ls(lsin2, j)))
    t2 = time.time()
    print("TIMER: Function completed time is: %.5f" % (t2 - t1))
    len(fp)
    return(fp)


def get_max_fp(lsin1, lsin2):
    nmol1 = len(lsin1)
    nmol2 = len(lsin2)
    t1 = time.time()
    maxfp=[]
    for i in range(nmol1):
        fp=[]
        for j in range(nmol2):
            fp.append(cal_sml(get_smile_ls(lsin1, i),get_smile_ls(lsin2, j)))
        maxfp.append(max(fp))
    t2 = time.time()
    print("TIMER: Function completed time is: %.5f" % (t2 - t1))
    return(maxfp)


def get_max_fp2(lsin1, lsin2):
    nmol1 = len(lsin1)
    nmol2 = len(lsin2)
    t1 = time.time()
    maxfp=[]
    fp=[]
    for i in range(nmol1):
        for j in range(nmol2):
            fp.append(cal_sml(get_smile_ls(lsin1, i),get_smile_ls(lsin2, j)))
    t2 = time.time()
    print("TIMER: Function completed time is: %.5f" % (t2 - t1))
    return(fp)

#
# def pKi_vs_act(df_d2, df_d3, A_U_B, chembl_version, assaydefinition, save2png):
#     """
#     Number of overlapping compounds in both D2 and D3. Calculate the delta pKi between D2 and D3 for the same compound
#     :param df_d2: dataframe, based on D2's dataset
#     :param df_d3: dataframe, based on D3's dataset
#     :param A_U_B: int, number of overlapping compounds
#     :param assaydefinition: str, either antagonist or agonist
#     :param save2png: boolean, True or False, to indicate whether to keep labels on or off
#     :param output_dir: str, name of output folder
#     :return:
#     """
#     # get indexes of overlapping compounds for each d2 and d3
#     mol_dup_pairs = get_mol_dup_pairs(df_d2, df_d3)
#     ls_d2, ls_d3 = [], []
#     for (i, j) in mol_dup_pairs:
#         ls_d2.append(i)
#         ls_d3.append(j)
#
#     # using indexes, get df of overlapping compounds
#     df_d2_overlap = df_d2.iloc[ls_d2].reset_index(drop=True)
#     df_d3_overlap = df_d3.iloc[ls_d3].reset_index(drop=True)
#
#     # getting the delta act, D2 pKi - D3 pKi
#     df = pd.DataFrame({'chembl': df_d2_overlap['chembl'], 'd_act': list(df_d2_overlap['pKi'] - df_d3_overlap['pKi'])})
#     df = df.sort_values(by=['d_act'], ascending=True)  # sorting
#     if assaydefinition == "antagonist":
#         assert round(list(df[df['chembl'] == 'CHEMBL42']['d_act'])[0], 2) == 0.49
#         assert round(list(df[df['chembl'] == 'CHEMBL4878826']['d_act'])[0], 2) == -1.42
#
#     ######################## PLOT ########################
#     num = len(df)
#     # figure parameters
#     nrow = 1
#     ncol = 1
#     figheight = 5
#     figwidth = 10
#     linewidth = 0.2
#
#     fig, (ax1) = plt.subplots(nrow, ncol, figsize=(figwidth, figheight))
#     # make a little extra space between the subplots
#     fig.subplots_adjust(hspace=1.5)
#
#     ax1.bar([a + 1 for a in range(num)], df['d_act'], lw=1, label='', color='lightgrey')
#
#     degrees = 90
#     plt.xticks(rotation=degrees)
#     ax1.set_xticks([a + 1 for a in range(num)])
#     ax1.set_xticklabels(df['chembl'], fontsize=8)
#     ax1.tick_params(axis="x", labelsize=8)
#     ax1.set_xlim([0, num + 1])
#     # ax1.set_xticks([])
#     if save2png:
#         ax1.set_xlabel('')
#         ax1.set_ylabel('')
#         ax1.set_title('')
#         ax1.set_yticklabels([])
#         plt.savefig(save2png, dpi=300, bbox_inches='tight')
#     else:
#         ax1.set_xlabel('Overlapping Compounds of D2 and D3')
#         ax1.set_ylabel('Delta pKi (D2 - D3)')
#         ax1.set_title(f"Delta pKi of Overlapped Compounds: {chembl_version}_{assaydefinition}")
#
#
#     plt.show()
#     plt.close()
#     verbose = True
#     if verbose:
#         print("d2 > d3, greater than 1:", round(len(df[df['d_act'] > 1]['d_act']) / A_U_B * 100, 2), "%")
#         print("d2 < d3, less than -1:", round(len(df[df['d_act'] < -1]['d_act']) / A_U_B * 100, 2), "%")
#         print("Percentage greater than 0:", round(len(df[df['d_act'] > 0]['d_act']) / A_U_B * 100, 2), "%")
#         print('min', min(df['d_act']))
#         print('max', max(df['d_act']))
#         print("mean", mean(df['d_act']))
#         print("median", median(df['d_act']))
#
#     return df



def plot_selectivity(df, ylim, save2png):
    num = len(df)
    # figure parameters
    nrow = 1
    ncol = 1
    figheight = 5
    figwidth = 10
    linewidth = 0.2

    fig, (ax1) = plt.subplots(nrow, ncol, figsize=(figwidth, figheight))
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=1.5)

    # ax1.bar([a + 1 for a in range(num)], df['pKi'], lw=1, label='', color='lightgrey')
    colors = ['gold' if y > 1 else 'lightgrey' for y in df['pKi']]
    ax1.bar([a + 1 for a in range(num)], df['pKi'], lw=1, label='', color=colors)

    plt.xticks(rotation=90)
    ax1.set_xticks([a + 1 for a in range(num)])
    ax1.set_xticklabels(df['chembl'], fontsize=8)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_xlim([0, num + 1])
    ax1.set_xticks([])

    if save2png:
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_title('')
        ax1.set_yticklabels([])
        ax1.set_ylim(ylim[0], ylim[1])
        plt.savefig(save2png, transparent = True, dpi=300, bbox_inches='tight')
    else:
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.set_xlabel('Individual Compounds')
        ax1.set_ylabel('pKi Distribution')
        # ax1.set_title("")
    plt.show()
    plt.close()

#
# def activity_probability(df, chembl_version, assay_definition, save2png):
#     min_range = -4.0
#     max_range = 3
#     bins = int((max_range - min_range)/0.50)
#     hist, bins = np.histogram(df['d_act'], bins, range=[min_range, max_range],density=False)
#
#     ## calculate probability instead of using density to aviod sum is not 1
#     hist = hist/sum(hist)
#
#     # manipulate bin
#     bin_shift=(bins[1]-bins[0])/2
#     nbin_center=len(bins)-1
#     xbin=bins+bin_shift
#     xbin=xbin[0:nbin_center]
#
#     # smooth x,y datapoints
#     xbin_smooth,yhist_smooth=get_smooth(xbin,hist)
#     # xbin_smooth=np.append(xbin_smooth, -xbin_smooth[-1:]-(xbin_smooth[-2:-1]-xbin_smooth[-1:]))
#
#     # xbin_smooth=np.append(xbin_smooth,0)
#     # yhist_smooth=np.append(yhist_smooth,0)
#
#     # figure parameters
#     nrow=1
#     ncol=1
#     figheight=4
#     figwidth=4
#     linewidth=1
#     fig, (ax1) = plt.subplots(nrow, ncol,figsize=(figwidth,figheight))
#     fig.subplots_adjust(hspace = 0.3, wspace=0.2)
#     ax1.plot(xbin_smooth, yhist_smooth, lw=linewidth, label='Ki', color='red')
#     ax1.set_xlabel('Delta Activity')
#     ax1.set_ylabel('Probability')
#     ax1.tick_params()
#     ax1.set_title(f"{chembl_version}_{assay_definition}")
#     if save2png:
#         plt.savefig(save2png, dpi=300,bbox_inches='tight')
#     plt.show()
#     plt.close()
#

def tanimoto(datadir, chembl_version, assaydefinition, save2png):
    # get smile string from *.smi
    data_d2 = get_data_smi(f"{datadir}/{chembl_version}/dataset_D2_{assaydefinition}_Ki/pubdata.smi")
    data_d3 = get_data_smi(f"{datadir}/{chembl_version}/dataset_D3_{assaydefinition}_Ki/pubdata.smi")

    sml_d3 = get_fp(data_d3)
    sml_d2 = get_fp(data_d2)
    sml_d3_vs_d2 = get_max_fp(data_d3, data_d2)
    sml_d2_vs_d3 = get_max_fp(data_d2, data_d3)

    # plotting
    ctrl_nbin = 20

    # histogram the raw data
    f_d2, bins = np.histogram(sml_d2, bins=ctrl_nbin, range=[0, 1.01], density=False)
    f_d3, bins = np.histogram(sml_d3, bins=ctrl_nbin, range=[0, 1.01], density=False)
    f_d3_vs_d2, bins = np.histogram(sml_d3_vs_d2, bins=ctrl_nbin, range=[0, 1.01], density=False)
    f_d2_vs_d3, bins = np.histogram(sml_d2_vs_d3, bins=ctrl_nbin, range=[0, 1.01], density=False)

    # calculate probability instead of using density to aviod sum is not 1
    f_d3 = f_d3 / sum(f_d3)
    f_d2 = f_d2 / sum(f_d2)
    f_d3_vs_d2 = f_d3_vs_d2 / sum(f_d3_vs_d2)
    f_d2_vs_d3 = f_d2_vs_d3 / sum(f_d2_vs_d3)

    # manipulate bin
    bin_shift = (bins[1] - bins[0]) / 2
    nbin_center = len(bins) - 1
    xbin = bins + bin_shift
    xbin = xbin[0:nbin_center]

    # smooth x,y datapoints
    x1, y1 = get_smooth(xbin, f_d3)
    x2, y2 = get_smooth(xbin, f_d2)
    x3, y3 = get_smooth(xbin, f_d3_vs_d2)
    x4, y4 = get_smooth(xbin, f_d2_vs_d3)
    x1 = np.append(x1, 1)
    y1 = np.append(y1, 0)
    x2 = np.append(x2, 1)
    y2 = np.append(y2, 0)
    x3 = np.append(x3, 1)
    y3 = np.append(y3, 0)
    x4 = np.append(x4, 1)
    y4 = np.append(y4, 0)

    # figure parameters
    nrow = 1
    ncol = 1
    figheight = 5
    figwidth = 6
    linewidth = 2

    fig, (ax1) = plt.subplots(nrow, ncol, figsize=(figwidth, figheight))

    if save2png:
        ax1.plot(x1, y1, lw=linewidth, label='                         ', color='green')
        ax1.plot(x2, y2, lw=linewidth, label='                         ', color='red')
        ax1.plot(x3, y3, lw=linewidth, label='                         ', color='blue')
        ax1.plot(x4, y4, lw=linewidth, label='                         ', color='cyan')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.legend(loc='upper right', frameon=False)
        ax1.set_title('')
        ax1.tick_params(labelsize=0)
        ax1.tick_params(top=True, bottom=True, left=True, right=True,
                        labelleft=False, labelbottom=False)
        plt.savefig(save2png, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        ax1.plot(x1, y1, lw=linewidth, label='D3 (within pairwise)', color='green')
        ax1.plot(x2, y2, lw=linewidth, label='D2 (within pairwise)', color='red')
        ax1.plot(x3, y3, lw=linewidth, label='D3 vs D2 (fix D3_i; max D2_j)', color='blue')
        ax1.plot(x4, y4, lw=linewidth, label='D2 vs D3 (fix D2_i; max D3_j)', color='cyan')
        ax1.set_xlabel('Similarity')
        ax1.set_ylabel('Probability')
        ax1.legend(loc='upper right')
        ax1.set_title(f'Pairwise Tanimoto Similarity: {chembl_version}_{assaydefinition}')
        plt.savefig(f"{output_dir}/tanimoto_{chembl_version}_{assaydefinition}.png", dpi=300, bbox_inches='tight')
        plt.show()
    plt.close()


def get_data(datain):
    # df = pd.read_csv("/home/khlee/cheminformatics/DAT/1_prepare_dataset_homo/publication_data.act", sep='\t', header = None)
    # datain="/home/khlee/cheminformatics/DAT/1_prepare_dataset_homo/publication_data.act"
    df = pd.read_csv(datain, sep='\t', header=None)
    sel_df = df[1]
    sel_df1 = sel_df
    return (sel_df1)


def get_smooth(x, y):
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(x.min(), x.max(), 300)
    # type: BSpline
    spl = make_interp_spline(x, y, k=2)
    power_smooth = spl(xnew)
    return (xnew, power_smooth)


def count_lines(ls_act_files):
    ls_lines = []
    ls_act_files.sort()
    for file in ls_act_files:
        # print(file)
        file = open(file, "r")
        nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
        line_count = len(nonempty_lines)
        file.close()
        ls_lines.append(line_count)
    return ls_lines


def parse_data(ls_act_files):
    ls_act_names = []  # filename; do this seperately, in order to sort the names
    for i in range(len(ls_act_files)):
        ls_act_names.append(ls_act_files[i].split('/')[-2].replace("dataset_", ""))
    ls_act_names.sort()
    #print(ls_act_names)
    ctrl_nbin = 20
    act_max = 12
    ls_act_data = []  # readin data
    ls_act_data_hist_counts = []  # readin data after binning; y
    ls_act_data_hist_bins = []  # readin data after binning; x
    ls_act_data_peak_x = []  # peak value at x
    ls_act_data_peak_y = []  # peak value at x
    ls_labels = []  # labels for plotting

    for i in range(len(ls_act_files)):
        #print(ls_act_files[i])
        ls_act_data.append(get_data(ls_act_files[i]))
        ## bining
        yhist, bins = np.histogram(ls_act_data[i], bins=ctrl_nbin, range=[0, act_max], density=False)
        ## calculate probability instead of using density to aviod sum is not 1
        yhist = yhist / sum(yhist)
        # manipulate bin
        bin_shift = (bins[1] - bins[0]) / 2
        nbin_center = len(bins) - 1
        xbin = bins + bin_shift
        xbin = xbin[0:nbin_center]
        # smooth x,y datapoints
        xbin_smooth, yhist_smooth = get_smooth(xbin, yhist)
        # return to array
        ls_act_data_hist_counts.append(yhist_smooth)
        ls_act_data_hist_bins.append(xbin_smooth)
        # find the peak
        ls_act_data_peak_x.append(list(xbin_smooth[yhist_smooth == max(yhist_smooth)])[0])
        ls_act_data_peak_y.append(list(yhist_smooth[yhist_smooth == max(yhist_smooth)])[0])
        # parsing labels
        ls_labels.append(ls_act_names[i].replace("_IC50", "").replace("_Ki", ""))

    return ls_act_data, ls_act_data_hist_counts, ls_act_data_hist_bins, ls_act_data_peak_x, ls_act_data_peak_y, ls_labels


def plot_show(ls_act_files, ls_act_data, ls_act_data_hist_counts, ls_act_data_hist_bins,
          ls_act_data_peak_x, ls_act_data_peak_y, ls_labels, save2png = True, maxylim=0.6, linewidth=1):
    nrow = 1
    ncol = 2
    figwidth = 5.4
    figheight = figwidth / 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    axs = axs.ravel()
    ls_lines = count_lines(ls_act_files)
    for i in range(int(len(ls_labels) / 2)):
        j = i * 2
        axs[i].set_ylim(0, maxylim)
        axs[i].plot(ls_act_data_hist_bins[j], ls_act_data_hist_counts[j], lw=linewidth, label='IC50', color='red')
        axs[i].plot(ls_act_data_hist_bins[j + 1], ls_act_data_hist_counts[j + 1], lw=linewidth, label='Ki',
                    color='blue')
        axs[i].plot([ls_act_data_peak_x[j] for a in range(2)], [0, ls_act_data_peak_y[j]], lw=1, color='red',
                    linestyle='dashed')
        axs[i].plot([ls_act_data_peak_x[j + 1] for a in range(2)], [0, ls_act_data_peak_y[j + 1]], lw=1, color='blue',
                    linestyle='dashed')
        if save2png:
            axs[i].tick_params(labelbottom=False, labelleft = False)
            plt.subplots_adjust(wspace=0.05, hspace=0)
            if i == 0:
                axs[i].legend(labels=["         ", ""], loc='upper left')
        else:
            # peak labels
            axs[i].text(ls_act_data_peak_x[j] - 0.7, ls_act_data_peak_y[j] + 0.04, str(round(ls_act_data_peak_x[j], 1)),
                        fontsize=12, color='red')
            axs[i].text(ls_act_data_peak_x[j + 1] - 2.5, ls_act_data_peak_y[j + 1] + 0.01,
                        str(round(ls_act_data_peak_x[j + 1], 1)), fontsize=12, color='blue')
            # label # of compounds
            axs[i].text(9.5, 0.55, ls_lines[2 * i], fontsize=12, color='red')
            axs[i].text(9.5, 0.50, ls_lines[2 * i + 1], fontsize=12, color='blue')
            # title label
            axs[i].text(6.2, 0.625, ls_labels[i*2], fontsize=12, horizontalalignment='center', color='black')
            # axis labels
            fig.text(0.5, 0.00, 'pIC50/pKi', ha='center')
            fig.text(0.04, 0.5, 'Probability', va='center', rotation='vertical')
            if i == 0:
                axs[i].legend(loc='upper left')



def distribution(datadir, chembl_version, target, assaydefinition1, assaydefinition2, save2png = False):
    ls_act_files = glob.glob(datadir+f"{chembl_version}/dataset_{target}_{assaydefinition1}_*/pubdata.act")
    ls_act_files = ls_act_files+glob.glob(datadir+f"{chembl_version}/dataset_{target}_{assaydefinition2}_*/pubdata.act")
    # getting data information
    ls_act_data,ls_act_data_hist_counts,ls_act_data_hist_bins,\
            ls_act_data_peak_x,ls_act_data_peak_y,ls_labels=parse_data(ls_act_files)
    # plotting
    plot_show(ls_act_files, ls_act_data, ls_act_data_hist_counts, ls_act_data_hist_bins, \
              ls_act_data_peak_x, ls_act_data_peak_y, ls_labels, save2png)
    # save as png
    if save2png:
        plt.savefig(f"output_dir/{target}_distribution.png", dpi=300,bbox_inches='tight')



def plot_gaussian_distribution_peak(dataset, ls_colors, selectivity=False, save2png=False):
    datadir = "/home/wons2/repositories/ai-DR/models/selectivity_D2D3/selectivity_datasets_Feb26/C33_0/"
    ls_act_files = [os.path.join(datadir, f"{data}/pubdata0.act") for data in dataset]
    ls_act_names = [os.path.basename(os.path.dirname(act_file)) for act_file in ls_act_files]

    ctrl_nbin = 20
    act_min, act_max = (-4, 4) if selectivity else (2, 12)

    ls_act_data_hist_counts, ls_act_data_hist_bins = [], []
    ls_act_data_peak_x, ls_act_data_peak_y = [], []

    for act_file in ls_act_files:
        act_data = -1 * get_data(act_file) if selectivity else get_data(act_file)
        yhist, bins = np.histogram(act_data, bins=ctrl_nbin, range=[act_min, act_max], density=False)
        yhist = yhist / sum(yhist)
        bin_shift = (bins[1] - bins[0]) / 2
        xbin = bins[:-1] + bin_shift
        xbin_smooth, yhist_smooth = get_smooth(xbin, yhist)
        ls_act_data_hist_counts.append(yhist_smooth)
        ls_act_data_hist_bins.append(xbin_smooth)
        peak_index = np.argmax(yhist_smooth)
        ls_act_data_peak_x.append(xbin_smooth[peak_index])
        ls_act_data_peak_y.append(yhist_smooth[peak_index])

    maxylim = 0.32
    fig, ax = plt.subplots(figsize=(5.4, 5.4), facecolor='w', edgecolor='k')
    ls_lines = count_lines(ls_act_files)

    for i, act_name in enumerate(ls_act_names):
        ax.set_ylim(0, maxylim)
        ax.plot(ls_act_data_hist_bins[i], ls_act_data_hist_counts[i], lw=2, label=act_name, color=ls_colors[i])
        ax.plot([ls_act_data_peak_x[i]] * 2, [0, ls_act_data_peak_y[i]], lw=2, linestyle='dashed', color=ls_colors[i])

        if not save2png:
            ax.text(ls_act_data_peak_x[i] - 0.7, ls_act_data_peak_y[i] + 0.04, str(round(ls_act_data_peak_x[i], 1)),
                    fontsize=12, color=ls_colors[i])
            ax.text(9.5 + i, 0.2, ls_lines[i], fontsize=12, color=ls_colors[i])

    if save2png:
        ax.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labelleft=False)
        plt.savefig(save2png, dpi=300, transparent=True, bbox_inches="tight")
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1), prop={'size': 12})

    plt.show()



def plot_gaussian_distribution_mean(dataset, ls_colors, mode = "median", selectivity=False, save2png=False):
    """

    :param dataset:
    :param ls_colors:
    :param mode: Either "median" or "mean"
    :param selectivity:
    :param save2png:
    :return:
    """
    datadir = "/home/wons2/repositories/ai-DR/models/selectivity_D2D3/selectivity_datasets/C33_0/"
    ls_act_files = [os.path.join(datadir, f"{data}/pubdata0.act") for data in dataset]
    ls_act_names = [os.path.basename(os.path.dirname(act_file)) for act_file in ls_act_files]

    ctrl_nbin = 20
    act_min, act_max = (-4, 4) if selectivity else (2, 12)

    ls_act_data_hist_counts, ls_act_data_hist_bins = [], []
    ls_act_data_mean_x = []

    for act_file in ls_act_files:
        act_data = -1 * get_data(act_file) if selectivity else get_data(act_file)
        act_mean = np.mean(act_data) if mode == "mean" else np.median(act_data)
        #act_median = np.median(act_data) if mode == "median" else
        yhist, bins = np.histogram(act_data, bins=ctrl_nbin, range=[act_min, act_max], density=False)
        yhist = yhist / sum(yhist)
        bin_shift = (bins[1] - bins[0]) / 2
        xbin = bins[:-1] + bin_shift
        xbin_smooth, yhist_smooth = get_smooth(xbin, yhist)
        ls_act_data_hist_counts.append(yhist_smooth)
        ls_act_data_hist_bins.append(xbin_smooth)
        ls_act_data_mean_x.append(act_mean)

    maxylim = 0.32
    fig, ax = plt.subplots(figsize=(5.4, 5.4), facecolor='w', edgecolor='k')
    ls_lines = count_lines(ls_act_files)

    for i, act_name in enumerate(ls_act_names):
        ax.set_ylim(0, maxylim)
        ax.plot(ls_act_data_hist_bins[i], ls_act_data_hist_counts[i], lw=2, label=act_name, color=ls_colors[i])
        mean_y_value = np.interp(ls_act_data_mean_x[i], ls_act_data_hist_bins[i], ls_act_data_hist_counts[i])
        ax.plot([ls_act_data_mean_x[i]] * 2, [0, mean_y_value], lw=2, linestyle='dashed', color=ls_colors[i])

        if not save2png:
            ax.text(ls_act_data_mean_x[i] - 0.7, mean_y_value + 0.02, f"{mode}: {ls_act_data_mean_x[i]:.1f}",
                    fontsize=12, color=ls_colors[i])
            ax.text(11 + i, 0.05, ls_lines[i], fontsize=12, color=ls_colors[i])

    if save2png:
        ax.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labelleft=False)
        plt.savefig(save2png, dpi=300, transparent=True, bbox_inches="tight")
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.2), prop={'size': 12})

    plt.show()



