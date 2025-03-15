import pandas as pd
from .add_doc_cmpd_count import add_doc_cmpd_count
from .data_to_pickle import data_to_pickle

def filter_small_sets(in_lines, output_dir=None, save=False, Verbose=False, threshold=5):
    # Remove compounds that come from sets of less than threshold compounds
    # this filter is generally only applicable for regression models

    # Add number of molecules per document to data first, so that the filter can be done
    if save:
        raw = in_lines

    in_lines = add_doc_cmpd_count(in_lines, Verbose=False)

    # I think this line causes some warning message
    in_lines = in_lines[in_lines['doc_cmpd_count'] > threshold]

    if Verbose:
        print(f'Number of pharmacological activity after data set size filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)
    # khlee add drop(columns=['index']), we don't need this column
    return in_lines.drop(columns=['index']).reset_index(drop=True)