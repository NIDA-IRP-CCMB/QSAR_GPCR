import pandas as pd
from .data_to_pickle import data_to_pickle

def filter_bao_format(in_lines, assaydefinition, output_dir=None, save=False, Verbose=False):
    """
    BAO ID Reference: https://www.ebi.ac.uk/ols/ontologies/bao
    """
    if save:
        raw = in_lines
    # Remove tissue-based; Keep: cell-based; others: need to test
    dict_bao = {'cell-based': 'BAO_0000219', 'tissue-based': 'BAO_0000221', 'single protein': 'BAO_0000357',
                'cell membrane': 'BAO_0000249', 'microsome': 'BAO_0000251', 'assay': 'BAO_0000019',
                'organism-based': 'BAO_0000218', 'subcellular': 'BAO_0000220'}

    if assaydefinition == "antagonist":
        ls_remove_format = ['tissue-based']     # items that we want to remove/filter out
    else:
        ls_remove_format = []

    ls_remove_bao = [dict_bao[format_] for format_ in ls_remove_format]
    in_lines = in_lines[~in_lines['bao_format'].isin(ls_remove_bao)]

    if Verbose:
        print(f'Number of pharmacological activity after BAO_FORMAT filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)
