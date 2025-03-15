import pandas as pd
import numpy as np

def write_smi_act_class(buffer, base_name, output_dir, inact_val=5.0, act_val=6.0, Verbose=False):
    active_mols = buffer[buffer.pchembl_value >= act_val]
    active_mols['pchembl_value_class'] = np.ones(len(active_mols), dtype=int)
    inactive_mols = buffer[buffer.pchembl_value <= inact_val]
    inactive_mols['pchembl_value_class'] = np.zeros(len(inactive_mols), dtype=int)
    last_list = pd.concat([active_mols, inactive_mols])

    if Verbose:
        print("Total number of molecules in training set: ", len(last_list))
        print("Number of active molecules: ", len(active_mols))
        print("Number of inactive molecules: ", len(inactive_mols))

    df_struct = last_list[['canonical_smiles', 'chembl_id']]
    df_act = last_list[['chembl_id', 'pchembl_value_class']]
    df_csv = last_list[['chembl_id', 'pchembl_value', 'pchembl_value_class', 'canonical_smiles']]

    smiles_file = base_name + '_class.smi'
    activity_file = base_name + '_class.act'
    output_csv_file = base_name + '_class.csv'

    df_struct.to_csv(output_dir + '/' + smiles_file, sep='\t', index=False, header=False)
    df_act.to_csv(output_dir + '/' + activity_file, sep='\t', index=False, header=False)
    df_csv.to_csv(output_dir + '/' + output_csv_file, sep='\t', index=False, header=False)