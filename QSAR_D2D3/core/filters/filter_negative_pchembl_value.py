from .data_to_pickle import data_to_pickle

def filter_negative_pchembl_value(in_lines, output_dir=None, save=False, Verbose=False):

    if save:
        raw = in_lines

    df_neg = in_lines[in_lines['standard_value'] <= 0]
    df_clean = in_lines.drop(df_neg.index).reset_index(drop=True)

    if Verbose:
        print('Number of pharmacological activity after removal of negative pChEMBL values:',
              len(df_clean))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return df_clean