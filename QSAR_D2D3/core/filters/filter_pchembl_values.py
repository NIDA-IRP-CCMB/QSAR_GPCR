import pandas as pd
from .calc_pscale import calc_pscale
from .data_to_pickle import data_to_pickle

def filter_pchembl_values(in_lines, output_dir=None, save=False, replace=False, kickout=False, Verbose=False):
    # remove compounds that don't have a pChEMBL value associated with them unless we want to
    # calculate it manually.  This should probably be run after filter_exact so that floor and
    # ceiling values get added before they would be eliminated.
    if save:
        raw = in_lines

    nans = in_lines[in_lines['pchembl_value'].isnull()]
    in_lines = in_lines.drop(list(nans.index))

    if replace:
        # drop_14 = list(nans[nans['standard_relation'] != '='].index)
        # drop_15 = list(nans[nans['standard_value'].isnull()].index)
        # drops = nans[nans['index'].apply(lambda x: any(s in x for s in
        #                                                list(drop_14 + list(set(drop_15) - set(drop_14)))))]
        # nans = nans.drop(list(drops.index))
        # nans = nans.reset_index()
        # for i in range(len(nans)):
        #     if nans['standard_units'][i] in ['M', 'mM', 'uM', 'nM', 'pM', 'fM']:
        #         nans.loc[i, 'pchembl_value'] = calc_pscale(nans['standard_value'][i],
        #                                                    nans['standard_units'][i])
        #     else:
        #         drops = pd.concat([drops, nans.iloc[list(i)]])
        # nans = nans.drop(list(drops.index), errors='ignore')
        # in_lines = pd.concat([in_lines, nans])

        #khlee do the above changes
        drop_14 = list(nans[nans['standard_relation'] != '='].index)
        drop_15 = list(nans[nans['standard_value'].isnull()].index)
        _combined_drop = list(drop_14 + list(set(drop_15) - set(drop_14)))
        nans = nans.drop(_combined_drop).reset_index(drop=True)
        for i in range(len(nans)):
            if nans['standard_units'][i] in ['M', 'mM', 'uM', 'nM', 'pM', 'fM']:
                nans.loc[i, 'pchembl_value'] = calc_pscale(nans['standard_value'][i],
                                                           nans['standard_units'][i])
            else:
                nans = nans.drop(i)
        in_lines = pd.concat([in_lines, nans])
    else:
        drops = nans

    if kickout:
        drops = drops.reset_index(drop=True)
        for i in range(len(drops)):
            kickouts.writerow(drops[i])

    if Verbose:
        print(f'Number of pharmacological activity after pChEMBL value filter: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)