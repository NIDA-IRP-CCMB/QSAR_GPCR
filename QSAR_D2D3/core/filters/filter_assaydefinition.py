import pandas as pd
from itertools import product  # For the filler function
from .data_to_pickle import data_to_pickle
import string

def filter_assaydefinition(in_lines, conf_dir, target, key, output_dir=None, save=False, Verbose=False):
    if save:
        raw = in_lines

    """ New NLP code incorporated. The unittest for DAT works but hERG fails (new code captures more compounds"""

    def filler(word, from_char, to_char):
        options = [(c,) if c != from_char else (from_char, to_char) for c in word]
        return (''.join(o) for o in product(*options))

    def count_special_characters(s):
        '''Counting the special characters for a given string'''
        special_characters = string.punctuation + " "
        return sum(1 for char in s if char in special_characters)

    # filter for displacement assay data
    if target == "D2" or target == "D3":
        filterfile = f'{conf_dir}/assaydefinition_DR_{key}.txt'
    if target == "mOPR":
        filterfile = f'{conf_dir}/assaydefinition_mOPR_{key}.txt'
    else:
        filterfile = f'{conf_dir}/assaydefinition_{target}_{key}.txt'

    df = pd.read_table(filterfile, names=['keys', 'text'])
    selection = list(df['text'])  # selection = specific definitions that we put for agonist or antagonist

    # modifying our "selection" list
    _selection = selection
    ls_fillers = [(')', '-'), (']', '-'), ('-', '- '), ('-', ' '), ('(', ''), (')', ''), ('[', ''), (']', ''),
                  ('-', ''), (',', ''), (', ', ''), (' ', ''), (']', ']-'), (']', ']- '), (']', '] '), (']', ']-S-'),
                  (']', ']-(+/-)-'),(']', '](R,S)-'), (']',']-N-methyl-'),(']', '](R,S)'),(']', '](R,S) ')]

    # for fill_key, fill_val in ls_fillers:
    #     for s in selection:
    #         _selection = _selection + list(filler(s, fill_key, fill_val))

    for s in selection:
        '''
        to avoid forever waiting in generating keyword combination, use a hard cutoff to skip generating pattern
        '''
        if count_special_characters(s) < 10:
            for fill_key, fill_val in ls_fillers:
                _selection = _selection + list(filler(s, fill_key, fill_val))

    # change everything to lower case
    selection = list(set(_selection))
    selection = [s.lower() for s in selection]

    in_lines_in = in_lines[
        in_lines['description'].apply(lambda x: any([s in x.lower() for s in selection]))].reset_index(drop=True)
    in_lines_out = in_lines[
        in_lines['description'].apply(lambda x: all([s not in x.lower() for s in selection]))].reset_index(
        drop=True)

    if Verbose:
        print(f'Number of pharmacological activity in ' + key, len(in_lines_in))
        in_lines_in[['description', 'pubmed_id', 'doi']].to_csv(target + "_data_" + key + ".dat",
                                                                sep='\t', index=False)
        in_lines_in.to_csv(target + "_data_" + key + ".tsv", sep='\t', index=False)

        print(f'Number of pharmacological activity out', len(in_lines_out))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines_in.reset_index(drop=True), in_lines_out.reset_index(drop=True)