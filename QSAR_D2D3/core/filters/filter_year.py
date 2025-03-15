import pandas as pd
from .data_to_pickle import data_to_pickle

def filter_year(in_lines, target = None, year = 1990, Verbose=False, output_dir = None, save = False):
    # Remove the entries with year 1990 and before
    # Dopamine 2 (1990 and before) may have mixed entries with DR2 and DR3. DR3 was discovered in fall 1990.
    if save:
        raw = in_lines
    if target == 'D2':
        in_lines = in_lines[in_lines['year'] > year]
    if Verbose:
        print(f'Number of compounds after {year} year filter: ', len(in_lines))
    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)