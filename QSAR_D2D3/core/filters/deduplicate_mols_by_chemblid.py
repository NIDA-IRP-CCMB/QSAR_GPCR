import pandas as pd

def deduplicate_mols_by_chemblid(in_lines, Verbose=False):
    '''
    Exclude compounds based on chemblid
    '''
    out_lines = in_lines.drop_duplicates(subset = 'chembl_id', keep = 'first').reset_index(drop=True)
    if Verbose:
        print('Number of unique chemblid after deduplication (using chemblid): ', len(out_lines))
    return out_lines