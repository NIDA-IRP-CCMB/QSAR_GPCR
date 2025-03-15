import pandas as pd
from rdkit.Chem import AllChem as Chem
from .data_to_pickle import data_to_pickle

def filter_size(in_lines, output_dir=None, save=False, maxweight=650, Verbose=False):
    if save:
        raw = in_lines

    # remove compounds with a MW that is greater than the maximum
    # this needs to be run after the structure standardization and desalting step

    for i in range(len(in_lines)):
        molweight = Chem.CalcExactMolWt(Chem.MolFromSmiles(in_lines['canonical_smiles'][i]))
        if molweight >= maxweight:
            in_lines = in_lines.drop(i)

    if Verbose:
        print(f'Number of pharmacological activity after molecular weight filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)