import pandas as pd
from .data_to_pickle import data_to_pickle

def filter_exact(in_lines, output_dir=None, save=False, include_ceilings=False, include_drugmatrix=False, Verbose=False):
    # Process compounds with inexact relationships
    # this should probably be run early in the filtering pipeline, but after values that are not Ki or IC50 are removed
    if save:
        raw = in_lines

    in_lines_sel = in_lines[in_lines['pchembl_value'].isnull()]
    in_lines1 = in_lines[in_lines['standard_relation'] == '=']
    in_lines2 = pd.DataFrame()
    in_lines3 = pd.DataFrame()

    # the code in this if statement works, but results in a red message
    if include_ceilings:
        in_lines_sel[in_lines_sel['standard_relation'] == '>']['pchembl_value'] = 1.0
        in_lines_sel[in_lines_sel['standard_relation'] == '>=']['pchembl_value'] = 1.0
        in_lines_sel[in_lines_sel['standard_relation'] == '<']['pchembl_value'] = 11.0
        in_lines_sel[in_lines_sel['standard_relation'] == '<=']['pchembl_value'] = 11.0
        in_lines2 = in_lines_sel[in_lines_sel['standard_relation'] != '=']
        in_lines2 = in_lines2.dropna(thresh=1)

    if include_drugmatrix:
        in_lines_sel2 = in_lines_sel[in_lines_sel['src_short_name'] == 'DRUGMATRIX']
        in_lines_sel2[in_lines_sel2.activity_comment.str.startswith('Not Active')]['chembl_value'] = 1.0
        in_lines3 = in_lines_sel2

    in_lines = pd.concat([in_lines1, in_lines2, in_lines3])

    if Verbose:
        print(f'Number of pharmacological activity after activity relationship type fixes: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)