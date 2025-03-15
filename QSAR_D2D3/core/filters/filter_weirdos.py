import pandas as pd
from .data_to_pickle import data_to_pickle

def filter_weirdos(in_lines, output_dir=None, save=False, kickout=False, Verbose=False):
    # filter out some odd edge cases we occasionall see in  a full data dump
    # this may not be necessary to use routinely.
    if save:
        raw = in_lines

    drop_1 = in_lines[in_lines['standard_value'].isnull()]
    drop_2 = in_lines[in_lines['standard_units'] != 'nM']
    drops = in_lines.iloc[list(drop_1.index + list(set(drop_2.index) - set(drop_1.index)))]
    in_lines = in_lines.drop(drops.index)

    if kickout:
        drops = drops.reset_index(drop=True)
        for i in range(len(drops)):
            kickouts.writerow(drops[i])

    if Verbose:
        print(f'Number of pharmacological activity after edge case filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)