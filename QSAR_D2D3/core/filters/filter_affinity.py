import pandas as pd
from .data_to_pickle import data_to_pickle

def filter_affinity(in_lines, output_dir=None, save=False, keepKi=False, keepIC50=False, keeppKi=False, keepEC50=False, Verbose=False):
    # Remove entries that are not Ki, IC50 values, or EC50 (for agonists)
    # if this filter is used, it should be placed early in the filter pipeline
    if save:
        raw = in_lines

    in_lines1 = pd.DataFrame()
    in_lines2 = pd.DataFrame()
    in_lines3 = pd.DataFrame()
    in_lines4 = pd.DataFrame()

    if keepKi:
        in_lines1 = in_lines[in_lines['standard_type'] == 'Ki']
    if keepIC50:
        in_lines2 = in_lines[in_lines['standard_type'] == 'IC50']
    if keeppKi:
       in_lines3 = in_lines[in_lines['standard_type'] == 'pKi']
    if keepEC50:
        in_lines4 = in_lines[in_lines['standard_type'] == 'EC50']

    in_lines = pd.concat([in_lines1, in_lines2, in_lines3, in_lines4])

    if Verbose:
        print(f'Number of pharmacological activity after Ki / IC50 / EC50 filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)