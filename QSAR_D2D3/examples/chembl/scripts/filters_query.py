import os, sys
## import all
from pathlib import Path
home = str(Path.home())
core_dir_1 = home + '/repositories/ai-x/core'
sys.path.insert(0, core_dir_1)
core_dir_2 = home + '/repositories/ai-DR/core'
conf_dir = core_dir_2 + "/conf"
sys.path.insert(0, core_dir_2)
sys.path.insert(0, conf_dir)

from filters_dop import *
from buildmodel import *
from misc import *
import io
import pandas as pd
from itertools import product

target = 'DR' # stands for Dopamine
output_dir = 'output_dir_papers'
check_output_dir(output_dir, keep_old=False)


def test_filters_d2d3(chembl_tsv_file):
    buffer = read_data(chembl_tsv_file, Verbose = True)
    buffer = filter_confidence(buffer, Verbose = True)
    buffer = filter_assay_type(buffer, Verbose = True)
    buffer = filter_affinity(buffer, Verbose = True, keepIC50=False, keepKi=True)
    buffer = filter_units(buffer, Verbose = True)
    filtered_out = filter_exact(buffer, Verbose = True)
    filtered_out = filtered_out[filtered_out['assay_type'] == 'B']      # unique to Dopamine Antagonists
    filtered_out = filtered_out[filtered_out['src_short_name'] != 'PATENT']
    return filtered_out


def find_doi(description, in_lines_out):
    ls_des = list(set(in_lines_out['description']))
    _sel = in_lines_out[in_lines_out['description'] == description]
    return list(set('doi.org/'+_sel['doi']))


def uncaught_table(in_lines_out):
    '''entire thing - remaining ones that are not "caught" by the keywords'''
    print("Remaining ones:", len(in_lines_out['description']), ', Unique:', len(list(set(in_lines_out['description']))), '\n')
    ls_description = list(set(in_lines_out['description']))
    ls_doi = []
    for i in range(0, len(ls_description)):
        description = ls_description[i]
        ls_doi.append(find_doi(description, in_lines_out))
    df = pd.DataFrame({"Description": ls_description, "DOI": ls_doi})
#     df.to_excel(f'{output_dir}/DR_uncaught_doi.xlsx', index = False, header=True)
    return df


def filler(word, from_char, to_char):
    options = [(c,) if c != from_char else (from_char, to_char) for c in word]
    return (''.join(o) for o in product(*options))


def print_in_lines(in_lines):
    print("\nTotal:", len(in_lines['description']),", Unique: ", len(list(set(in_lines['description']))))


def catch_keys(key, in_lines):
    filename = 'assaydefinition_' + target + '_' + key + '.txt'
    filterfile = home+"/repositories/ai-DR/datasets/conf/"+filename
    df = pd.read_table(filterfile, names=['keys', 'text'])
    selection = list(df['text']) # selection = specific definitions that we put for agonist or antagonist

    # "Natural Language Processing": modifying our "selection" list
    ls_add = []
    _selection = selection
    for s in selection:
        _selection = _selection + list(filler(s, "-", " "))
        _selection = _selection + list(filler(s, "-", ""))
        _selection = _selection + list(filler(s, " ", ""))
    selection = list(set(_selection))
    selection = [s.lower() for s in selection]

    in_lines_in = in_lines[in_lines['description'].apply(lambda x: any([s in x.lower() for s in selection]))].reset_index(drop=True)
    in_lines_out = in_lines[in_lines['description'].apply(lambda x: all([s not in x.lower() for s in selection]))].reset_index(drop=True)
    print(key, ":", len(in_lines_in['description']), ', Unique:', len(list(set(in_lines_in['description']))))
    return in_lines_in, in_lines_out


def save_to_excel(df, filename = '', version = 'short', save = False):
    if version == 'short':
        df = df[['description', 'doi']].drop_duplicates()
    elif version == 'long':
        df = df[['description', 'doi', 'doc_id', 'standard_value', 'standard_units', 'pchembl_value', 'chembl_id', 'canonical_smiles']].drop_duplicates()
    df['doi'] = 'doi.org/' + df['doi'].astype(str)
    if save:
        df.to_excel(filename)
    return df


def filter_information(subtarget, version, save = False):
    '''
    :param subtarget: "D2" or "D3", your target protein
    :param version: "short" or "long", indicates the version of columns you want; Small or Big dataframe
    :return: 4 dataframes, xx is everything combined, xx_antagonist is antagonists only (filtered using keyterms),
                xx_agonist is agonists only, xx_uncaught are the rows that were NOT caught by the keyterms.
    '''
    chembl_tsv_file = home+f"/repositories/ai-DR/datasets/pgsql/all_pgsql/chembl33_{subtarget}.tsv"

    xx = test_filters_d2d3(chembl_tsv_file)
    print_in_lines(xx)

    assay = 'antagonist'
    in_lines_in, in_lines_out = catch_keys(assay, xx)
    xx_antagonist = in_lines_in
    filename = output_dir+f'/{subtarget}_{assay}.xlsx'
    df2 = save_to_excel(in_lines_in, filename, version, save)

    assay = 'agonist'
    in_lines_in, in_lines_out = catch_keys(assay, in_lines_out)
    xx_agonist = in_lines_in
    xx_uncaught = in_lines_out
    filename = output_dir+f'/{subtarget}_{assay}.xlsx'
    df = save_to_excel(in_lines_in, filename, version, save)

    assay = 'others'
    in_lines_in, in_lines_out = catch_keys(assay, in_lines_out)
    filename = output_dir+f'/{subtarget}_{assay}.xlsx'
    df = save_to_excel(in_lines_in, filename, version, save)

    # uncaught
    uncaught_df = uncaught_table(in_lines_out)
    return xx, xx_antagonist, xx_agonist, xx_uncaught


def filter_selections(in_lines, selection):
    # modifying our "selection" list
    ls_add = []
    _selection = selection
    for s in selection:
        _selection = _selection + list(filler(s, "-", " "))
        _selection = _selection + list(filler(s, "-", ""))
        _selection = _selection + list(filler(s, " ", ""))
    selection = list(set(_selection))
    selection = [s.lower() for s in selection]

    in_lines_in = in_lines[in_lines['description'].apply(lambda x: any([s in x.lower() for s in selection]))].reset_index(drop=True)
    in_lines_out = in_lines[in_lines['description'].apply(lambda x: all([s not in x.lower() for s in selection]))].reset_index(drop=True)
    print("before:", len(in_lines_in['description']), len(list(set(in_lines_in['description']))))
    return in_lines_in, in_lines_out


def papers_1990(xx, year):
    xx2 = xx[xx['year'] <= year]
    df = xx2[['year', 'description', 'doi']].sort_values(by = 'year').drop_duplicates()
    #df.to_excel(output_dir+'/1990.xlsx')
    return df


def get_pieChart(_xx, title):
    dict_BAO = {}
    dict_format = {}
    dict_BAO_to_format = {'BAO_0000219': 'cell-based', 'BAO_0000221': 'tissue-based',
                          'BAO_0000357': 'single protein', 'BAO_0000249': 'cell membrane', 'BAO_0000251': 'microsome',
                         'BAO_0000019': 'assay'}
    dict_colors = {'cell-based': 'lightskyblue', 'tissue-based': 'lightcoral', 'single protein': 'dodgerblue',
                   'cell membrane': 'limegreen', 'microsome': 'gold', 'assay': 'orange'}
    for BAO in _xx['bao_format'].unique():
        num = len(_xx[_xx['bao_format']==BAO])
        dict_BAO[BAO] = num
        formats = dict_BAO_to_format[BAO]
        dict_format[formats] = num
    labels = list(dict_format.keys())
    values = list(dict_format.values())

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.0f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct
    colours = dict(zip(labels, plt.cm.tab10.colors[:len(labels)]))

    ls_colors = []
    for label in labels:
        ls_colors.append(dict_colors[label])
    plt.pie(values, labels=labels, colors = ls_colors, autopct=make_autopct(values), textprops={'fontsize': 13})
    plt.title(title+': BAO Proportions')
    fig = plt.gcf()
    size = 6
    fig.set_size_inches(size, size)
    plt.legend(loc = 'upper right', bbox_to_anchor= (1.35,1))
    plt.tight_layout()
    plt.show()
