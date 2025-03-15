import pandas as pd
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import SaltRemover
from molvs import Standardizer
from .data_to_pickle import data_to_pickle

def filter_salts(in_lines, conf_dir, output_dir=None, save=False, kickout=False, Verbose=False, useIsomer=False):
    # standardize structures and remove salts
    #
    # This should be called before any other filters having to do with molecular structures as it
    # affects both the molecular structure and the molecular weight of many compounds that come out of ChEMBL
    if save:
        raw = in_lines

    s = Standardizer()
    salt_file = conf_dir + '/Salts.txt'     #TODO: Make conf_dir global?
    remover = SaltRemover.SaltRemover(defnFilename=salt_file)

    for i in range(len(in_lines)):
        mol_in = Chem.MolFromSmiles(in_lines['canonical_smiles'][i])
        mol_out = s.standardize(mol_in)
        smiles_out = Chem.MolToSmiles(remover(mol_out), isomericSmiles=useIsomer)
        if '.' in smiles_out:
            in_lines = in_lines.drop(i)
            if kickout:
                kickouts.writerow(in_lines[i])
        else:
            in_lines.loc[i, 'canonical_smiles'] = smiles_out

    if Verbose:
        print(f'Number of pharmacological activity after desalting pass: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)