import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import glob
from statistics import mean, stdev

def add_value_labels(ax, spacing=5):
    '''Please see https://stackoverflow.com/questions/59143306/add-label-values-to-bar-chart-and-line-chart-in-matplotlib
    '''
    space = spacing
    va = 'bottom'
    for i in ax.patches:
        y_value = i.get_height()
        x_value = i.get_x() + i.get_width() / 2

        label = "{:.2f}".format(y_value)
        ax.annotate(label, (x_value, y_value), xytext=(0, space),
                    textcoords="offset points", ha='center', va=va)

    # https://matplotlib.org/stable/gallery/color/named_colors.html

def get_internal_stats(stats):
    '''
    returns list of calculated Rs and R2s for all of models
    '''
    if len(stats) > 0:
        ls_lines = []
        for filename in stats:
            with open(filename) as f:
                lines = f.readlines()
                ls_lines.append(lines)
        ls_Rp, ls_Rs, ls_R2, ls_RMSE, ls_MSE = [], [], [], [], []
        for line in ls_lines:
            line = line[0].replace('\n', '')
            line_split = line.split(' ')
            for i in range(len(line_split)):
                if line_split[i] == 'Rp':
                    ls_Rp.append(float(line_split[i + 1]))
                if line_split[i] == 'Rs':
                    ls_Rs.append(float(line_split[i + 1]))
                if line_split[i] == 'R2':
                    ls_R2.append(float(line_split[i + 1]))
                if line_split[i] == 'RMSE':
                    ls_RMSE.append(float(line_split[i + 1]))
                if line_split[i] == 'MSE':
                    ls_MSE.append(float(line_split[i + 1]))
        mean_Rp = round(mean(ls_Rp), 2)
        mean_Rs = round(mean(ls_Rs), 2)
        mean_R2 = round(mean(ls_R2), 2)
        mean_RMSE = round(mean(ls_RMSE), 2)
        mean_MSE = round(mean(ls_MSE), 2)
        stdev_Rp = round(stdev(ls_Rp), 2)
        stdev_Rs = round(stdev(ls_Rs), 2)
        stdev_R2 = round(stdev(ls_R2), 2)
        stdev_RMSE = round(stdev(ls_RMSE), 2)
        stdev_MSE = round(stdev(ls_MSE), 2)
    else:
        mean_Rp = 0
        mean_Rs = 0
        mean_R2 = 0
        mean_RMSE = 0
        mean_MSE = 0
        stdev_Rp = 0
        stdev_Rs = 0
        stdev_R2 = 0
        stdev_RMSE = 0
        stdev_MSE = 0

    return mean_Rp, mean_Rs, mean_R2, mean_RMSE, mean_MSE, stdev_Rp, stdev_Rs, stdev_R2, stdev_RMSE, stdev_MSE


def bar_plot(df, metric, y, colors, save2png):
    ymin, ymax = y
    figwidth = 2
    figheight = 5
    col_mean = metric + "(mean)"
    col_stdev = metric + "(stdev)"
    yerr = df[col_stdev].to_numpy().T
    ax = df.plot.bar(x='Dataset Version', y=col_mean, yerr = yerr, color=colors, figsize=(figwidth, figheight), width=0.85);
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='dotted', linewidth=0)
    ax.legend().set_visible(False)
    if metric == "Rp":
        ax.tick_params(top=False, bottom=False, left=True, right=False, labelleft=True, labelbottom=True, pad=0)
    elif metric == "Rs":
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=True, pad=0)
    elif metric == "R2":
        ax.tick_params(top=False, bottom=False, left=False, right=True, labelleft=True, labelbottom=True, pad=0)
    plt.axhline(y=0, color='black', linestyle='-')
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_ticks(np.arange(ymin, ymax, 0.2))

    if save2png:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('', fontsize=10)
        ax.set_ylabel('', fontsize=10)
        plt.savefig(save2png, dpi=600, transparent=True, bbox_inches="tight")

    else:
        add_value_labels(ax)
        ax.set_title(f"{metric}")

    return df


def export_legend(colors, filename, expand=[-5, -5, 5, 5]):
    patch1 = mpatches.Patch(color=colors[0], label="             ")
    patch2 = mpatches.Patch(color=colors[1], label="")
    patch3 = mpatches.Patch(color=colors[2], label="")
    legend = plt.legend(handles=[patch1, patch2, patch3], fontsize=16, framealpha=1, frameon=True)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def get_internal_dataset_quality(datadir):
    ls_paths = glob.glob(datadir + "/models_*")
    ls_paths = sorted(ls_paths)
    df = pd.DataFrame()
    for path in ls_paths:
        ls_models = glob.glob(path + "/*")
        ls_models = sorted(ls_models)
        version = path.split('/')[-1]
        for model in ls_models:
            dataset = model.split('/')[-1]
            stats = glob.glob(model + "/reg_xgb_0.15/stat_*")
            n_models = len(stats)
            #         print(version, dataset, n_models, get_internal_stats(stats))
            mean_Rp, mean_Rs, mean_R2, mean_RMSE, mean_MSE, stdev_Rp, stdev_Rs, stdev_R2, stdev_RMSE, stdev_MSE = get_internal_stats(
                stats)

            stat = {'Dataset Version': [version],
                    'model': [dataset],
                    "n_models": [n_models],
                    'Rp(mean)': [mean_Rp],
                    'Rs(mean)': [mean_Rs],
                    'R2(mean)': [mean_R2],
                    'RMSE(mean)': [mean_RMSE],
                    'MSE(mean)': [mean_MSE],
                    'Rp(stdev)': [stdev_Rp],
                    'Rs(stdev)': [stdev_Rs],
                    'R2(stdev)': [stdev_R2],
                    'RMSE(stdev)': [stdev_RMSE],
                    'MSE(stdev)': [stdev_MSE]}
            df_row = pd.DataFrame(stat)
            df = pd.concat([df, df_row], ignore_index=True, sort=False)
    df = df[df["n_models"] > 0]

    return df