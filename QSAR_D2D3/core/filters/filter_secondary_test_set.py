import pandas as pd
from .filter_config import test_set_2_compounds
from .data_to_pickle import data_to_pickle

def filter_secondary_test_set(in_lines, output_dir=None, save=False, Verbose=False):
    if save:
        raw = in_lines
    # Remove compounds present in secondary test set
    # this filter only applies to andy_hERG models

    # maybe we should break this line into 2 to make it easier to read
    in_lines = in_lines.drop(
        in_lines[in_lines['chembl_id'].apply(lambda x: any([s == str(x) for s in test_set_2_compounds]))].index)

    if Verbose:
        print(f'Number of pharmacological activity after removing testset 2 compounds:', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)