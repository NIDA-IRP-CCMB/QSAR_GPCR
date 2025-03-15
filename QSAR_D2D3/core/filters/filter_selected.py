import pandas as pd
from .data_to_pickle import data_to_pickle

def filter_selected(in_lines, output_dir=None, save=False, Verbose=False):
    """
    Remove 'PATENT' papers
    Hand select to remove entries based on paper quality (and any redundancies)
    """
    if save:
        raw = in_lines

    in_lines = in_lines[in_lines['src_short_name'] != 'PATENT']

    if Verbose:
        print(f'Number of pharmacological activity after patent & hand selecting (paper) filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)