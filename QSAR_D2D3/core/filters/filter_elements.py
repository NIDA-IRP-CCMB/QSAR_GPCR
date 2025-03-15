import pandas as pd
from rdkit.Chem import AllChem as Chem
from .data_to_pickle import data_to_pickle

def filter_elements(in_lines, output_dir=None, save=False, kickout=False, Verbose=False):
    # remove entries with oddball elements
    # this needs to be run after the desalting and molecular standardization step
    if save:
        raw = in_lines

    element_filter = Chem.MolFromSmarts('[!C&!c&!N&!n&!O&!o&!S&!s&!P&!p&!F&!Cl&!Br&!I]')
    for i in range(len(in_lines)):
        curr_mol = Chem.MolFromSmiles(in_lines['canonical_smiles'][i])
        if not curr_mol.HasSubstructMatch(element_filter):
            continue
        else:
            in_lines = in_lines.drop(i)
            if kickout:
                kickouts.writerow(in_lines[i])

    if Verbose:
        print(f'Number of pharmacological activity after oddball element filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)