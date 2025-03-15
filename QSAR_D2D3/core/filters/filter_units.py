import pandas as pd
from .data_to_pickle import data_to_pickle

def filter_units(in_lines, output_dir=None, save=False, Verbose=False):
    # Remove compounds without a standard unit of nM
    # this filter should be placed after the filter_affinity filter is used to limit the number of false positives
    if save:
        raw = in_lines
    in_lines = in_lines[in_lines['standard_units'] == 'nM']
    # TODO: using only nM may not be good. we can address this in the future

    if Verbose:
        print(f'Number of pharmacological activity after standard units filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)