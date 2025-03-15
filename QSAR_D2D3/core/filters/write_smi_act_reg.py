import pandas as pd
from .calc_pscale import calc_pscale
from .filter_low_pchembl_values import filter_low_pchembl_values


def write_smi_act_reg(final_data, base_name, output_dir='./', pchem_cutoff = 1, add_extra=False):
    final_data = filter_low_pchembl_values(final_data, cutoff = pchem_cutoff, Verbose = True)
    nans = final_data[final_data['pchembl_value'].isnull()]
    if len(nans) > 0:
        final_data = final_data.drop(list(nans.index))
        ki = nans[nans['standard_type'] == 'Ki'].reset_index()
        # pki = nans[nans['standard_type'].apply(lambda x: any(x in s for s in ['pKi','Log Ki']))].reset_index()
        pki = nans[nans['standard_type'] == 'pKi'].reset_index()
        for i in range(len(ki)):
            ki['pchembl_value'][i] = calc_pscale(ki['standard_value'][i],
                                                 ki['standard_units'][i])
        for i in range(len(pki)):
            pki['pchembl_value'][i] = pki['standard_value'][i]
        final_data = pd.concat([final_data, ki, pki]).reset_index(drop=True)

    sort = []
    for i in range(len(final_data['chembl_id'])):
        sort.append(int(final_data['chembl_id'][i][6:]))
    final_data['sort'] = sort
    final_data = final_data.sort_values(by='sort')

    smiles_file = base_name + '.smi'
    activity_file = base_name + '.act'

    df_struct = final_data[['canonical_smiles', 'chembl_id']]
    df_act = final_data[['chembl_id', 'pchembl_value']]

    # confirm whether or not we want to have the headers written to the files
    df_struct.to_csv(output_dir + '/' + smiles_file, sep='\t', index=False, header=False)
    df_act.to_csv(output_dir + '/' + activity_file, sep='\t', index=False, header=False)

    return