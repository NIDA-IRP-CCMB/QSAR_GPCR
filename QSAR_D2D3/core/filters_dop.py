## define enviroment
import sys, os
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
conf_dir = core_dir+"/keywords"
sys.path.insert(0, core_dir)
sys.path.insert(0, conf_dir)

from filters import *
from misc import check_output_dir

import io
from contextlib import redirect_stdout
import datetime
from itertools import product

def filter_assaydefinition(in_lines, target, key, Verbose=False):
    def filler(word, from_char, to_char):
        options = [(c,) if c != from_char else (from_char, to_char) for c in word]
        return (''.join(o) for o in product(*options))
    # filter for displacement assay data
    if target == "D2" or target == "D3":
        filterfile = f'{conf_dir}/assaydefinition_DR_{key}.txt'
        df = pd.read_table(filterfile, names=['keys', 'text'])
        selection = list(df['text'])  # selection = specific definitions that we put for agonist or antagonist

        # modifying our "selection" list
        _selection = selection
        ls_fillers = [(')', '-'), (']', '-'), ('-', '- '), ('-', ' '), ('(', ''), (')', ''), ('[', ''), (']', ''),
            ('-', ''), (',', ''), (', ', ''), (' ', '')]

        for fill_key, fill_val in ls_fillers:
            for s in selection:
                _selection = _selection + list(filler(s, fill_key, fill_val))

        # change everything to lower case
        selection = list(set(_selection))
        selection = [s.lower() for s in selection]

        in_lines_in = in_lines[
            in_lines['description'].apply(lambda x: any([s in x.lower() for s in selection]))].reset_index(drop=True)
        in_lines_out = in_lines[
            in_lines['description'].apply(lambda x: all([s not in x.lower() for s in selection]))].reset_index(drop=True)

    else:
        filterfile = conf_dir+'/assaydefinition_'+target+'_'+key+'.txt'
        df = pd.read_table(filterfile, names=['keys', 'text'])
        selection = list(df['text'])

        in_lines_in = in_lines[
            in_lines['description'].apply(lambda x: any([s in x for s in selection]))]
        in_lines_out = in_lines[
            in_lines['description'].apply(lambda x: all([s not in x for s in selection]))]
    
    if Verbose:
        print('Number of compounds in ' + key, len(in_lines_in))
        in_lines_in[['description', 'pubmed_id', 'doi']].to_csv("hERG_data_" + key + ".dat",
                                                                sep='\t', index=False)
        in_lines_in.to_csv("hERG_data_" + key + ".tsv", sep='\t', index=False)

        print('Number of compounds out', len(in_lines_out))

    return in_lines_in.reset_index(drop=True), in_lines_out.reset_index(drop=True)


def filter_assay_type(in_lines, target = 'others', assaydefinition = 'others', Verbose=False):
    # Remove entries that are not binding or functional studies
    # if this filter is used, it should be done early in the pipeline

    # one line version below, takes longer
    #in_lines = in_lines[in_lines['confidence_score'].apply(lambda x: np.any(s in x for s in ['B','F']))]

    if "D2" in target or "D3" in target:
        if assaydefinition == 'antagonist':
            in_lines = in_lines[in_lines['assay_type'] == 'B']
    else:
        in_lines1 = in_lines[in_lines['assay_type'] == 'B']
        in_lines2 = in_lines[in_lines['assay_type'] == 'F']
        in_lines = pd.concat([in_lines1, in_lines2])

    if Verbose:
        print('Number of compounds after assay type filter: ', len(in_lines))

    return in_lines.reset_index(drop=True)


def filter_year(in_lines, target, year = 1990, Verbose=False):
    # Remove the entries with year 1990 and before
    # Dopamine 2 (1990 and before) may have mixed entries with DR2 and DR3. DR3 was discovered in fall 1990.
    if target == 'D2':
        in_lines = in_lines[in_lines['year'] > year]
    if Verbose:
        print(f'Number of compounds after {year} year filter: ', len(in_lines))

    return in_lines.reset_index(drop=True)


def filter_bao_format(in_lines, target, assaydefinition, Verbose=False):
    # Remove tissue-based; Keep: cell-based; others: need to test
    dict_bao = {'cell-based': 'BAO_0000219', 'tissue-based': 'BAO_0000221', 'single protein': 'BAO_0000357',
                          'cell membrane': 'BAO_0000249', 'microsome': 'BAO_0000251', 'assay': 'BAO_0000019'}
    ls_remove_format = []
    if assaydefinition == "antagonist":
        if "D2" in target:
            #ls_remove_format = ['assay']
            #ls_remove_format = ['tissue-based']
            ls_remove_format = ['tissue-based', 'assay']     # items that we want to remove/filter out
        elif "D3" in target:
            ls_remove_format = ['tissue-based']
    ls_remove_bao = [dict_bao[format_] for format_ in ls_remove_format]
    in_lines = in_lines[~in_lines['bao_format'].isin(ls_remove_bao)]

    if Verbose:
        print(f'Number of compounds after BAO_FORMAT filter: ', len(in_lines))

    return in_lines.reset_index(drop=True)


def filter_selected(in_lines, target, assaydefinition, Verbose=False):
    """
    Remove 'PATENT' papers
    Hand select to remove entries based on paper quality (and any redundancies)
    """
    # remove patent papers
    in_lines = in_lines[in_lines['src_short_name'] != 'PATENT']

    # remove specific papers
    ls_remove_doc_id = [71409, 81789, 119044, 48827, 77073]
    if "D2" in target or "D3" in target:
        if assaydefinition == 'antagonist':
            ls_remove_doc_id.extend([98345, 98610])     # GTP assay S ... these should be agonists
    in_lines = in_lines[~in_lines['doc_id'].isin(ls_remove_doc_id)]

    # remove specific compounds
    ls_chembl_id = ["CHEMBL198174"]
    if "D3" in target and assaydefinition == "antagonist":
            in_lines = in_lines[~in_lines['chembl_id'].isin(ls_chembl_id)]

    if Verbose:
        print(f'Number of compounds after patent & hand selecting (paper) filter: ', len(in_lines))

    return in_lines.reset_index(drop=True)


############ everything above is a modified version of ~/repositories/ai-x/core/filters.py ###############

# def read_data(file_name, Verbose=False):
#     '''
#     Added in one more column: "bao_format"
#     labels = {0:'pref_name', 1:'organism', 2:'assay_id', 3:'assay_type',
#                    4:'relationship_type', 5:'relationship_desc', 6:'confidence_score',
#                    7'curated_by', 8:'description', 9:'activity_id', 10:'relation',
#                    11:'value', 12:'units', 13:'type', 14:'standard_relation', 15:'standard_value',
#                    16:'standard_units', 17:'standard_flag', 18:'standard_type', 19:'pchembl_value',
#                    20:'activity_comment', 21:'data_validity_comment', 22:'potential_duplicate',
#                    23:'text_value', 24:'standard_text_value', 25:'molregno', 26:'chembl_id',
#                    27:'canonical_smiles', 28:'pref_name', 29:'parent_molregno', 30:'active_molregno',
#                    31:'doc_id', 32:'pubmed_id', 33:'doi', 34:'journal', 35:'year', 36:'volume',
#                    37:'first_page', 38:'src_short_name', 39:'bao_format'}
#     '''
#     labels = ['pref_name_target', 'organism', 'assay_id', 'assay_type', 'relationship_type', 'relationship_desc',
#               'confidence_score', 'curated_by', 'description', 'activity_id', 'relation', 'value', 'units', 'type',
#               'standard_relation', 'standard_value', 'standard_units', 'standard_flag', 'standard_type',
#               'pchembl_value', 'activity_comment', 'data_validity_comment', 'potential_duplicate', 'text_value',
#               'standard_text_value', 'molregno', 'chembl_id', 'canonical_smiles', 'pref_name', 'parent_molregno',
#               'active_molregno', 'doc_id', 'pubmed_id', 'doi', 'journal', 'year', 'volume', 'first_page',
#                'src_short_name', 'bao_format']
#     raw_data = pd.read_csv(file_name, names=labels, header=None, sep="\t")
#
#     if Verbose:
#         print('Number of compounds at starting: ', len(raw_data))
#
#     return raw_data

def get_dataframe(buffer, df, target, standard_type, assaydefinition):
    # extract the data from 'buffer'
    ls_line = buffer.split("\n")
    ls_filter_data = []
    for line in ls_line:
        idx = line.find(':')
        filter_data = line[idx + 1:len(line)].replace(" ",
                                                      "")  # takes the values after ":" substring and removes spaces
        ls_filter_data.append(filter_data)
    ls_filter_data = [i for i in ls_filter_data if i]  # removes empty string from list

    # ls_output has all the data WITHIN ONE COLUMN
    ls_output = []
    ls_output.append(target)
    for i in range(0, 3):
        ls_output.append(ls_filter_data[i])
    ls_output.append(standard_type)
    for i in range(3, 6):
        ls_output.append(ls_filter_data[i])
    ls_output.append("")
    ls_output.append("")
    ls_output.append(target)
    ls_output.append(assaydefinition)
    ls_output.append(standard_type)
    for i in range(6, len(ls_filter_data)):
        ls_output.append(ls_filter_data[i])

    # combine with PREVIOUS dataframe (appending onto df)
    df_one = pd.DataFrame(ls_output)  # dataframe with ONE column only

    df = df.reset_index(drop=True)
    df_one = df_one.reset_index(drop=True)
    concatenated_df = pd.concat([df, df_one], axis=1)

    return concatenated_df


def save_to_excel(df, chembl, xlsx_dir):
    # saving excel spreadsheet
    # column names are all '0'. Change to 0, 1, 2, ...
    ls_col = []
    for i in range(0, len(df.columns)):
        ls_col.append(i)
    df.columns = ls_col

    # index and label columns
    df.index = ['', 'Starting data', 'After confidence score filter', 'After assay type filter', '', \
                'After Ki / IC50 filter', 'After standard units filter', 'After activity relationship type fixes', \
                '', '', '', '', '', 'After assay description filter', 'Reserving hERG calibration compounds',
                "After year 1990 filter", "After bioassay ontology filter", "After patent and manual curation filter",\
                'After data set size filter', 'Desalting pass', 'After oddball element filter', \
                'After molecular weight filter', 'After pChEMBL value filter', 'After edge case filter', \
                'After deduplication pass', 'Excluding 5-6 pKi/pIC50', 'Binders', 'Non-binders']

    check_output_dir(xlsx_dir, keep_old=False)
    df.to_excel(f'{xlsx_dir}/chembl{chembl}_filtering_statistics.xlsx', sheet_name='Table', header=True, index=True)
