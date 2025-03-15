from .pairwise_comparison import pairwise_comparison

def deduplicate_mols_by_similarity(in_lines, Verbose=False, useIsomer=False):
    '''
    Exclude compounds that are similar within the DataFrame.
    The 'ls_remove' contains the 'j' indexes, which correspond to the repeated indexes later on.
    '''

    ls_remove = [pair[1] for pair in pairwise_comparison(in_lines, thresh=0.999, useIsomer=useIsomer)]

    out_lines = in_lines.drop(index=ls_remove).reset_index(drop=True)

    assert len(in_lines) - len(set(ls_remove)) == len(out_lines)

    if Verbose:
        print('Number of compounds after deduplication (using pairwise similarity > 0.999): ', len(out_lines))

    return out_lines