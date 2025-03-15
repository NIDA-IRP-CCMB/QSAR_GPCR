import pandas as pd
from .data_to_pickle import data_to_pickle

def filter_assay_type(in_lines, output_dir=None, save=False, target=None, assaydefinition=None, Verbose=False):
    # Remove entries that are not binding or functional studies
    # if this filter is used, it should be done early in the pipeline
    if save:
        raw = in_lines

    if target == 'mOPR':
        if assaydefinition == 'antagonist':
            in_lines = in_lines[in_lines['assay_type'] == 'B']
        else:
            in_lines1 = in_lines[in_lines['assay_type'] == 'B']
            in_lines2 = in_lines[in_lines['assay_type'] == 'F']
            in_lines = pd.concat([in_lines1,in_lines2])

    if target == 'D2' or target == 'D3':
        if assaydefinition == 'antagonist':
            in_lines = in_lines[in_lines['assay_type'] == 'B']
        else:
            in_lines1 = in_lines[in_lines['assay_type'] == 'B']
            in_lines2 = in_lines[in_lines['assay_type'] == 'F']
            in_lines = pd.concat([in_lines1, in_lines2])

    if target is None:
        in_lines1 = in_lines[in_lines['assay_type'] == 'B']
        in_lines2 = in_lines[in_lines['assay_type'] == 'F']
        in_lines = pd.concat([in_lines1,in_lines2])

    if Verbose:
        print(f'Number of pharmacological activity after assay type filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)