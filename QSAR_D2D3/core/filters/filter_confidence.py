import pandas as pd
from .data_to_pickle import data_to_pickle

def filter_confidence(in_lines, output_dir=None, save=False, broad=False, Verbose=False):

    # Remove compounds with a confidence score less than 9
    # this should be run early in the filter pipeline
    if save:
        raw = in_lines

    if broad:
        #one line version below, takes 0.5 seconds (longer)
        #in_lines = in_lines[in_lines['confidence_score'].apply(lambda x: np.any(np.in1d(x, [8,9])))]
        in_lines1=in_lines[in_lines['confidence_score']== 8]
        in_lines2=in_lines[in_lines['confidence_score']== 9]
        in_lines = pd.concat([in_lines1,in_lines2])
    else:
        in_lines=in_lines[in_lines['confidence_score']==9]

    if Verbose:
        print(f'Number of pharmacological activity after confidence score filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)